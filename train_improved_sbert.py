"""
Training script for the improved SBERT model on the IndoNLI dataset.
Incorporates best practices for NLI training with careful hyperparameter settings.
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from datasets import load_dataset

# Import the improved model
from indo_sbert_improved_model import ImprovedSBERTModel
# Import the NLILogger for unified logging
from nli_logger import NLILogger, create_logger

class NLIDataset(Dataset):
    """
    Dataset class for the IndoNLI dataset.
    """
    
    def __init__(self, dataset, tokenizer, max_length=128):
        """
        Initialize dataset.
        
        Args:
            dataset: IndoNLI dataset split
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Map NLI labels
        self.label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        
        # Log label distribution to check for issues
        self._log_label_distribution()
    
    def _log_label_distribution(self):
        """Log the distribution of labels to help with debugging"""
        labels = []
        for item in self.dataset:
            # Check if the label is already numeric
            if isinstance(item["label"], int):
                label = item["label"]
            # Handle string labels
            elif isinstance(item["label"], str):
                label = self.label_map.get(item["label"].lower(), 1)
            else:
                label = 1
            labels.append(label)
        
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logging.info(f"Label distribution: {label_counts}")
        if 1 not in label_counts:
            logging.warning(f"No neutral labels found!")
            # Print sample items with invalid labels
            invalid_examples = []
            count = 0
            for item in self.dataset:
                if count >= 5:  # Limit to 5 examples
                    break
                if isinstance(item["label"], str) and item["label"].lower() not in self.label_map:
                    invalid_examples.append({"label": item["label"], "premise": item["premise"][:50]})
                    count += 1
            logging.warning(f"Sample invalid examples: {invalid_examples}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get premise, hypothesis and label
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        
        # Handle different label formats
        if isinstance(item["label"], int) and 0 <= item["label"] <= 2:
            # Already numeric and in the right range
            label = item["label"]
        elif isinstance(item["label"], str):
            # Convert string label to numeric
            label = self.label_map.get(item["label"].lower(), 1)  
        else:
            label = 1  
        
        # Tokenize premise
        premise_encoding = self.tokenizer(
            premise,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize hypothesis
        hypothesis_encoding = self.tokenizer(
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "premise_input_ids": premise_encoding["input_ids"].squeeze(),
            "premise_attention_mask": premise_encoding["attention_mask"].squeeze(),
            "hypothesis_input_ids": hypothesis_encoding["input_ids"].squeeze(),
            "hypothesis_attention_mask": hypothesis_encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(predictions, labels):
    """
    Compute metrics for evaluation.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    conf_matrix = confusion_matrix(labels, predictions)
    
    # Per-class F1 scores
    class_f1 = f1_score(labels, predictions, average=None)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_f1": class_f1,
        "confusion_matrix": conf_matrix
    }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train for one epoch.
    
    Args:
        model: Model
        dataloader: DataLoader
        optimizer: Optimizer
        scheduler: Scheduler
        device: Device
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        premise_input_ids = batch["premise_input_ids"].to(device)
        premise_attention_mask = batch["premise_attention_mask"].to(device)
        hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
        hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Check for invalid labels and skip batch if found
        if torch.any(labels < 0) or torch.any(labels >= 3):
            logging.warning(f"Invalid label detected in batch: {labels}. Skipping batch.")
            continue
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            premise_input_ids=premise_input_ids,
            premise_attention_mask=premise_attention_mask,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask
        )
        
        # Ensure logits and labels have compatible shapes
        if logits.shape[0] != labels.shape[0]:
            logging.warning(f"Shape mismatch: logits {logits.shape}, labels {labels.shape}")
            continue
        
        try:
            # Compute loss with error handling
            loss = criterion(logits, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                logging.warning("NaN loss detected. Skipping batch.")
                continue
                
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update scheduler
            scheduler.step()
            
            # Update loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        except RuntimeError as e:
            logging.error(f"Error in batch: {e}")
            # Print the actual label values for debugging
            logging.error(f"Labels in batch: {labels.cpu().tolist()}")
            continue
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluate model.
    
    Args:
        model: Model
        dataloader: DataLoader
        device: Device
        
    Returns:
        Metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Skip batches with invalid labels
            if torch.any(labels < 0) or torch.any(labels >= 3):
                logging.warning(f"Invalid label detected during evaluation: {labels}. Skipping batch.")
                continue
            
            # Forward pass
            logits = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Add to lists
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir, is_best=False):
    """
    Save checkpoint.
    
    Args:
        model: Model
        optimizer: Optimizer
        scheduler: Scheduler
        epoch: Current epoch
        metrics: Metrics dictionary
        output_dir: Output directory
        is_best: Whether this is the best model
    """
    # Create directory for checkpoints
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics
    }, os.path.join(checkpoint_dir, "training_state.bin"))
    
    # If this is the best model, save to best_model directory
    if is_best:
        best_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        
        # Save metrics for best model
        import json
        with open(os.path.join(best_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


def train_with_nli_logger(args, logger):
    """
    Train the improved SBERT model with the NLILogger.
    
    Args:
        args: Arguments
        logger: NLILogger instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log hyperparameters
    hparams = vars(args)
    logger.log_hyperparameters(hparams)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    logger.logger.info(f"Loaded tokenizer from {args.base_model}")
    
    # Load model
    model = ImprovedSBERTModel(
        model_name=args.base_model,
        pooling_mode=args.pooling_mode,
        classifier_hidden_size=args.classifier_hidden_size,
        dropout=args.dropout,
        use_cross_attention=args.use_cross_attention
    )
    model.to(device)
    logger.logger.info(f"Created model: {args.base_model}")
    logger.logger.info(f"Pooling mode: {args.pooling_mode}")
    logger.logger.info(f"Classifier hidden size: {args.classifier_hidden_size}")
    logger.logger.info(f"Dropout: {args.dropout}")
    logger.logger.info(f"Using cross-attention: {args.use_cross_attention}")
    
    # Load dataset
    try:
        logger.logger.info("Loading IndoNLI dataset from Hugging Face")
        train_dataset = load_dataset("indonli", split="train")
        val_dataset = load_dataset("indonli", split="validation")
        test_lay_dataset = load_dataset("indonli", split="test_lay")
        test_expert_dataset = load_dataset("indonli", split="test_expert")
        
        # Log dataset statistics
        logger.log_dataset_statistics({
            "train": train_dataset,
            "validation": val_dataset,
            "test_lay": test_lay_dataset,
            "test_expert": test_expert_dataset
        })
    except Exception as e:
        logger.logger.error(f"Error loading dataset from Hugging Face: {e}")
        try:
            # Try local loading
            logger.logger.info("Attempting to load dataset from local files")
            train_dataset = load_dataset("json", data_files="data/indonli/train.json", split="train")
            val_dataset = load_dataset("json", data_files="data/indonli/valid.json", split="train")
            test_lay_dataset = load_dataset("json", data_files="data/indonli/test_lay.json", split="train")
            test_expert_dataset = load_dataset("json", data_files="data/indonli/test_expert.json", split="train")
            
            # Log dataset statistics
            logger.log_dataset_statistics({
                "train": train_dataset,
                "validation": val_dataset,
                "test_lay": test_lay_dataset,
                "test_expert": test_expert_dataset
            })
        except Exception as e2:
            logger.logger.error(f"Error loading dataset from local files: {e2}")
            raise RuntimeError(f"Failed to load dataset: {e}, {e2}")
    
    # Create datasets
    max_seq_length = args.max_seq_length if hasattr(args, 'max_seq_length') else args.max_length
    train_data = NLIDataset(train_dataset, tokenizer, max_length=max_seq_length)
    val_data = NLIDataset(val_dataset, tokenizer, max_length=max_seq_length)
    test_lay_data = NLIDataset(test_lay_dataset, tokenizer, max_length=max_seq_length)
    test_expert_data = NLIDataset(test_expert_dataset, tokenizer, max_length=max_seq_length)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_lay_dataloader = DataLoader(
        test_lay_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_expert_dataloader = DataLoader(
        test_expert_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.logger.info(f"Number of training examples: {len(train_data)}")
    logger.logger.info(f"Number of validation examples: {len(val_data)}")
    logger.logger.info(f"Number of test_lay examples: {len(test_lay_data)}")
    logger.logger.info(f"Number of test_expert examples: {len(test_expert_data)}")
    logger.logger.info(f"Batch size: {args.batch_size}")
    logger.logger.info(f"Total steps: {total_steps}")
    logger.logger.info(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = None
    
    for epoch in range(args.num_epochs):
        logger.logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Train epoch
        train_loss = 0.0
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Check for invalid labels and skip batch if found
            if torch.any(labels < 0) or torch.any(labels >= 3):
                logger.logger.warning(f"Invalid label detected in batch: {labels}. Skipping batch.")
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update loss
            train_loss += loss.item()
            
            # Log training step
            if step % 100 == 0:
                logger.log_train_step(
                    epoch=epoch+1,
                    step=step,
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0]
                )
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                premise_input_ids = batch["premise_input_ids"].to(device)
                premise_attention_mask = batch["premise_attention_mask"].to(device)
                hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
                hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                # Skip batches with invalid labels
                if torch.any(labels < 0) or torch.any(labels >= 3):
                    continue
                
                # Forward pass
                outputs = model(
                    premise_input_ids=premise_input_ids,
                    premise_attention_mask=premise_attention_mask,
                    hypothesis_input_ids=hypothesis_input_ids,
                    hypothesis_attention_mask=hypothesis_attention_mask
                )
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                
                # Move to CPU
                preds = preds.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                
                # Append to lists
                val_preds.extend(preds)
                val_labels.extend(batch_labels)
        
        # Calculate validation metrics
        val_metrics = logger.log_evaluation(
            dataset_name="validation",
            predictions=val_preds,
            labels=val_labels,
            checkpoint_name=f"epoch-{epoch+1}"
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Log epoch results
        logger.log_epoch(
            epoch=epoch+1,
            train_loss=avg_train_loss,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path
        )
        
        # Check if this is the best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_model_path = checkpoint_path
            logger.logger.info(f"New best model: {best_model_path} (accuracy: {best_val_accuracy:.4f})")
    
    # Load best model for final evaluation
    logger.logger.info(f"Training complete. Loading best model from {best_model_path}")
    model = ImprovedSBERTModel.from_pretrained(best_model_path)
    model.to(device)
    
    # Evaluate on test sets
    logger.logger.info("Evaluating on test_lay dataset")
    
    model.eval()
    test_lay_preds = []
    test_lay_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_lay_dataloader, desc="Test Lay"):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Skip batches with invalid labels
            if torch.any(labels < 0) or torch.any(labels >= 3):
                continue
            
            # Forward pass
            outputs = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Move to CPU
            preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            # Append to lists
            test_lay_preds.extend(preds)
            test_lay_labels.extend(batch_labels)
    
    # Log test_lay metrics
    logger.log_evaluation(
        dataset_name="test_lay",
        predictions=test_lay_preds,
        labels=test_lay_labels,
        checkpoint_name="best"
    )
    
    # Evaluate on test_expert
    logger.logger.info("Evaluating on test_expert dataset")
    
    test_expert_preds = []
    test_expert_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_expert_dataloader, desc="Test Expert"):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Skip batches with invalid labels
            if torch.any(labels < 0) or torch.any(labels >= 3):
                continue
            
            # Forward pass
            outputs = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Move to CPU
            preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            # Append to lists
            test_expert_preds.extend(preds)
            test_expert_labels.extend(batch_labels)
    
    # Log test_expert metrics
    logger.log_evaluation(
        dataset_name="test_expert",
        predictions=test_expert_preds,
        labels=test_expert_labels,
        checkpoint_name="best"
    )
    
    # Generate visualizations and report
    logger.plot_training_curve()
    logger.compare_checkpoints()
    logger.generate_summary_report()
    
    logger.logger.info("Training and evaluation complete")
    return best_model_path


def train(args):
    """
    Train the improved SBERT model.
    
    Args:
        args: Arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Add this line for more debug info on CUDA errors
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load IndoNLI dataset
    logger.info("Loading IndoNLI dataset...")
    try:
        dataset = load_dataset("indonli")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        test_lay_dataset = dataset["test_lay"]
        test_expert_dataset = dataset["test_expert"]
        
        # Log dataset structure to debug
        logger.info(f"Dataset structure: {dataset}")
        logger.info(f"Train dataset features: {train_dataset.features}")
        logger.info(f"Sample example: {train_dataset[0]}")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Try loading from local directories
        try:
            dataset = load_dataset("json", data_files={
                "train": "data/indonli/train.json",
                "validation": "data/indonli/valid.json",
                "test_lay": "data/indonli/test_lay.json",
                "test_expert": "data/indonli/test_expert.json"
            })
            train_dataset = dataset["train"]
            validation_dataset = dataset["validation"]
            test_lay_dataset = dataset["test_lay"]
            test_expert_dataset = dataset["test_expert"]
            
            # Log dataset structure to debug
            logger.info(f"Local dataset features: {train_dataset.features}")
            logger.info(f"Sample example: {train_dataset[0]}")
            
        except Exception as e2:
            logger.error(f"Error loading local dataset: {e2}")
            raise RuntimeError(f"Failed to load dataset from either HF or local: {e}, {e2}")
    
    # Create datasets
    max_seq_length = args.max_seq_length if hasattr(args, 'max_seq_length') else args.max_length
    train_dataset = NLIDataset(train_dataset, tokenizer, max_length=max_seq_length)
    validation_dataset = NLIDataset(validation_dataset, tokenizer, max_length=max_seq_length)
    test_lay_dataset = NLIDataset(test_lay_dataset, tokenizer, max_length=max_seq_length)
    test_expert_dataset = NLIDataset(test_expert_dataset, tokenizer, max_length=max_seq_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_lay_dataloader = DataLoader(
        test_lay_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_expert_dataloader = DataLoader(
        test_expert_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info(f"Initializing model from {args.base_model}...")
    model = ImprovedSBERTModel(
        model_name=args.base_model,
        pooling_mode=args.pooling_mode,
        classifier_hidden_size=args.classifier_hidden_size,
        dropout=args.dropout,
        use_cross_attention=args.use_cross_attention
    )
    
    # Move to device
    model = model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon
    )
    
    # Calculate total steps
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info("Starting training...")
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        validation_metrics = evaluate(model, validation_dataloader, device)
        logger.info(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
        logger.info(f"Validation F1: {validation_metrics['macro_f1']:.4f}")
        
        # Check for improvement
        if validation_metrics['macro_f1'] > best_f1:
            best_f1 = validation_metrics['macro_f1']
            is_best = True
            patience_counter = 0
            logger.info("New best model!")
        else:
            is_best = False
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch+1,
            metrics=validation_metrics,
            output_dir=args.output_dir,
            is_best=is_best
        )
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info("Early stopping!")
            break
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "best_model")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model = ImprovedSBERTModel.from_pretrained(best_model_path)
        model = model.to(device)
    
    # Evaluate on test sets
    logger.info("Evaluating on test_lay dataset...")
    test_lay_metrics = evaluate(model, test_lay_dataloader, device)
    logger.info(f"Test lay accuracy: {test_lay_metrics['accuracy']:.4f}")
    logger.info(f"Test lay F1: {test_lay_metrics['macro_f1']:.4f}")
    
    logger.info("Evaluating on test_expert dataset...")
    test_expert_metrics = evaluate(model, test_expert_dataloader, device)
    logger.info(f"Test expert accuracy: {test_expert_metrics['accuracy']:.4f}")
    logger.info(f"Test expert F1: {test_expert_metrics['macro_f1']:.4f}")
    
    # Save final metrics
    import json
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump({
            "test_lay": test_lay_metrics,
            "test_expert": test_expert_metrics
        }, f, indent=2)
    
    # Save confusion matrices as text files
    np.savetxt(
        os.path.join(args.output_dir, "test_lay_confusion_matrix.txt"),
        test_lay_metrics["confusion_matrix"],
        fmt="%d"
    )
    np.savetxt(
        os.path.join(args.output_dir, "test_expert_confusion_matrix.txt"),
        test_expert_metrics["confusion_matrix"],
        fmt="%d"
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved SBERT model on IndoNLI")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="firqaaa/indo-sentence-bert-base",
                        help="Base model to use")
    parser.add_argument("--pooling_mode", type=str, default="mean_pooling",
                        choices=["mean_pooling", "max_pooling", "cls"],
                        help="Pooling mode")
    parser.add_argument("--classifier_hidden_size", type=int, default=256,
                        help="Hidden size of classifier")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--use_cross_attention", action="store_true",
                        help="Use cross-attention between sentences")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/indo-sbert-improved",
                        help="Output directory")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloader")
    
    # Add new logger parameter
    parser.add_argument("--use_new_logger", action="store_true",
                        help="Use the new NLILogger")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train with either the new logger or the traditional approach
    if args.use_new_logger:
        logger = create_logger("improved-sbert", args.output_dir)
        train_with_nli_logger(args, logger)
    else:
        train(args)
