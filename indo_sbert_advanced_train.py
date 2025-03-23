"""
Training script for Advanced SBERT model on IndoNLI dataset.
This script incorporates the unified logging system for better results tracking.
"""

import os
import argparse
import logging
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Import the advanced SBERT model
from indo_sbert_advanced_model import SBERTAdvancedModel
from nli_logger import create_logger, NLILogger
from utils import set_seed


class AdvancedNLIDataset(Dataset):
    """Advanced dataset class for NLI task."""
    
    def __init__(self, examples, tokenizer, max_length=128, dynamic_padding=True):
        """
        Initialize dataset.
        
        Args:
            examples: List of examples with premise, hypothesis, and label
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            dynamic_padding: Whether to use dynamic padding
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding
        
        # Convert labels to IDs
        self.label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract premise, hypothesis, and label
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        
        # Handle unknown labels properly - use neutral as default if label not found
        if "label" in example and example["label"] in self.label_map:
            label = self.label_map[example["label"]]
        else:
            # Use neutral (1) as default instead of -1 to avoid batch skipping
            label = 1  # neutral
        
        # Tokenize premise and hypothesis separately
        premise_tokens = self.tokenizer(
            premise,
            max_length=self.max_length,
            padding=False if self.dynamic_padding else "max_length",
            truncation=True,
            return_tensors=None
        )
        
        hypothesis_tokens = self.tokenizer(
            hypothesis,
            max_length=self.max_length,
            padding=False if self.dynamic_padding else "max_length",
            truncation=True,
            return_tensors=None
        )
        
        # Combine into a single item
        return {
            "premise_input_ids": premise_tokens["input_ids"],
            "premise_attention_mask": premise_tokens["attention_mask"],
            "hypothesis_input_ids": hypothesis_tokens["input_ids"],
            "hypothesis_attention_mask": hypothesis_tokens["attention_mask"],
            "label": label
        }
    
    @staticmethod
    def collate_fn(batch, tokenizer, max_length):
        """
        Collate function for dynamic padding.
        
        Args:
            batch: Batch of examples
            tokenizer: Tokenizer for padding
            max_length: Maximum sequence length
            
        Returns:
            Batched and padded tensors
        """
        # Separate the items
        premise_input_ids = [item["premise_input_ids"] for item in batch]
        premise_attention_mask = [item["premise_attention_mask"] for item in batch]
        hypothesis_input_ids = [item["hypothesis_input_ids"] for item in batch]
        hypothesis_attention_mask = [item["hypothesis_attention_mask"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Pad the sequences
        premise_input_ids = tokenizer.pad(
            {"input_ids": premise_input_ids},
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"]
        
        premise_attention_mask = tokenizer.pad(
            {"input_ids": premise_attention_mask},
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"]
        
        hypothesis_input_ids = tokenizer.pad(
            {"input_ids": hypothesis_input_ids},
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"]
        
        hypothesis_attention_mask = tokenizer.pad(
            {"input_ids": hypothesis_attention_mask},
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"]
        
        # Convert to tensors
        labels = torch.tensor(labels)
        
        return {
            "premise_input_ids": premise_input_ids,
            "premise_attention_mask": premise_attention_mask,
            "hypothesis_input_ids": hypothesis_input_ids,
            "hypothesis_attention_mask": hypothesis_attention_mask,
            "label": labels
        }


def load_indonli_data(split="train"):
    """
    Load IndoNLI dataset.
    
    Args:
        split: Dataset split
        
    Returns:
        Dataset examples
    """
    try:
        # Try loading from Hugging Face
        dataset = load_dataset("indonli", split=split)
        return dataset
    except Exception as e:
        # Try loading from local files
        try:
            if split == "train":
                data_path = "data/indonli/train.json"
            elif split == "validation":
                data_path = "data/indonli/valid.json"
            elif split == "test_lay":
                data_path = "data/indonli/test_lay.json"
            else:  # test_expert
                data_path = "data/indonli/test_expert.json"
            
            dataset = load_dataset("json", data_files=data_path, split="train")
            return dataset
        except Exception as e2:
            raise RuntimeError(f"Error loading dataset: {e}, {e2}")


def train_with_nli_logger(args, logger):
    """
    Train advanced SBERT model with the NLILogger.
    
    Args:
        args: Command-line arguments
        logger: NLILogger instance
        
    Returns:
        Path to best model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    logger.logger.info(f"Loaded tokenizer from {args.base_model}")
    
    # Create model
    model = SBERTAdvancedModel(
        model_name=args.base_model,
        pooling_mode=args.pooling_mode,
        classifier_hidden_sizes=[int(size) for size in args.classifier_hidden_sizes.split(",")],
        dropout=args.dropout,
        use_cross_attention=args.use_cross_attention
    )
    model.to(device)
    logger.logger.info(f"Created model with architecture: {model.__class__.__name__}")
    
    # Log model architecture
    logger.logger.info(f"Pooling mode: {args.pooling_mode}")
    logger.logger.info(f"Classifier hidden sizes: {args.classifier_hidden_sizes}")
    logger.logger.info(f"Using cross-attention: {args.use_cross_attention}")
    
    # Load dataset
    try:
        # Try loading using Hugging Face
        logger.logger.info("Loading IndoNLI dataset from Hugging Face")
        train_dataset = load_dataset("indonli", split="train")
        validation_dataset = load_dataset("indonli", split="validation")
        test_lay_dataset = load_dataset("indonli", split="test_lay")
        test_expert_dataset = load_dataset("indonli", split="test_expert")
        
        # Log dataset statistics
        logger.log_dataset_statistics({
            "train": train_dataset,
            "validation": validation_dataset,
            "test_lay": test_lay_dataset,
            "test_expert": test_expert_dataset
        })
    except Exception as e:
        logger.logger.error(f"Error loading dataset from Hugging Face: {e}")
        
        try:
            # Try loading from local files
            logger.logger.info("Attempting to load dataset from local files")
            train_dataset = load_indonli_data("train")
            validation_dataset = load_indonli_data("validation")
            test_lay_dataset = load_indonli_data("test_lay")
            test_expert_dataset = load_indonli_data("test_expert")
            
            # Log dataset statistics
            logger.log_dataset_statistics({
                "train": train_dataset,
                "validation": validation_dataset,
                "test_lay": test_lay_dataset,
                "test_expert": test_expert_dataset
            })
        except Exception as e2:
            logger.logger.error(f"Error loading dataset from local files: {e2}")
            raise RuntimeError(f"Failed to load dataset: {e}, {e2}")
    
    # Create datasets
    train_data = AdvancedNLIDataset(
        train_dataset,
        tokenizer,
        max_length=args.max_length,
        dynamic_padding=True
    )
    
    validation_data = AdvancedNLIDataset(
        validation_dataset,
        tokenizer,
        max_length=args.max_length,
        dynamic_padding=True
    )
    
    test_lay_data = AdvancedNLIDataset(
        test_lay_dataset,
        tokenizer,
        max_length=args.max_length,
        dynamic_padding=True
    )
    
    test_expert_data = AdvancedNLIDataset(
        test_expert_dataset,
        tokenizer,
        max_length=args.max_length,
        dynamic_padding=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, args.max_length)
    )
    
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, args.max_length)
    )
    
    test_lay_loader = DataLoader(
        test_lay_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, args.max_length)
    )
    
    test_expert_loader = DataLoader(
        test_expert_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, args.max_length)
    )
    
    # Define optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Log training information
    logger.logger.info(f"Number of training examples: {len(train_data)}")
    logger.logger.info(f"Number of validation examples: {len(validation_data)}")
    logger.logger.info(f"Number of test_lay examples: {len(test_lay_data)}")
    logger.logger.info(f"Number of test_expert examples: {len(test_expert_data)}")
    logger.logger.info(f"Batch size: {args.batch_size}")
    logger.logger.info(f"Total steps: {total_steps}")
    logger.logger.info(f"Warmup steps: {warmup_steps}")
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize best model tracking
    best_val_accuracy = 0
    best_model_path = None
    patience_counter = 0
    
    # Training loop
    logger.logger.info(f"Starting training for {args.num_epochs} epochs")
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Skip batches with invalid labels
            if (labels < 0).any():
                logger.logger.warning(f"Skipping batch with invalid labels: {labels}")
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
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Apply gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update training loss
            train_loss += loss.item()
            
            # Log every 100 steps
            if step % 100 == 0:
                logger.log_train_step(
                    epoch=epoch+1,
                    step=step,
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0]
                )
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc="Validation"):
                # Move batch to device
                premise_input_ids = batch["premise_input_ids"].to(device)
                premise_attention_mask = batch["premise_attention_mask"].to(device)
                hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
                hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                # Skip batches with invalid labels
                if (labels < 0).any():
                    continue
                
                # Forward pass
                logits = model(
                    premise_input_ids=premise_input_ids,
                    premise_attention_mask=premise_attention_mask,
                    hypothesis_input_ids=hypothesis_input_ids,
                    hypothesis_attention_mask=hypothesis_attention_mask
                )
                
                # Calculate loss
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Move predictions and labels to CPU
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                
                # Append predictions and labels
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(validation_loader)
        
        # Calculate validation metrics
        val_metrics = logger.log_evaluation(
            dataset_name="validation",
            predictions=all_preds,
            labels=all_labels,
            checkpoint_name=f"epoch-{epoch+1}"
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Log epoch
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
            logger.logger.info(f"New best model: {best_model_path}")
    
    # Load best model for evaluation
    if best_model_path is None:
        logger.logger.warning("No best model saved. Using the last checkpoint instead.")
        best_model_path = os.path.join(args.output_dir, f"checkpoint-{args.num_epochs}")
        # If directory doesn't exist, create it and save current model
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
    
    logger.logger.info(f"Loading best model from {best_model_path}")
    try:
        model = SBERTAdvancedModel.from_pretrained(best_model_path)
        model.to(device)
    except Exception as e:
        logger.logger.error(f"Error loading best model: {e}")
        logger.logger.warning("Continuing with current model state")
    
    # Evaluate on test sets
    model.eval()
    
    # Evaluate on test_lay
    logger.logger.info("Evaluating on test_lay dataset")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_lay_loader, desc="Test Lay"):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Skip batches with invalid labels
            if (labels < 0).any():
                continue
            
            # Forward pass
            logits = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Move predictions and labels to CPU
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Append predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Log evaluation results
    logger.log_evaluation(
        dataset_name="test_lay",
        predictions=all_preds,
        labels=all_labels,
        checkpoint_name="best"
    )
    
    # Evaluate on test_expert
    logger.logger.info("Evaluating on test_expert dataset")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_expert_loader, desc="Test Expert"):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Skip batches with invalid labels
            if (labels < 0).any():
                continue
            
            # Forward pass
            logits = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Move predictions and labels to CPU
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Append predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Log evaluation results
    logger.log_evaluation(
        dataset_name="test_expert",
        predictions=all_preds,
        labels=all_labels,
        checkpoint_name="best"
    )
    
    # Generate visualizations and report
    logger.plot_training_curve()
    logger.compare_checkpoints()
    logger.generate_summary_report()
    
    logger.logger.info("Training and evaluation complete")
    return best_model_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train advanced SBERT model on IndoNLI dataset")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="firqaaa/indo-sentence-bert-base",
                        help="Base model to use")
    parser.add_argument("--pooling_mode", type=str, default="mean_pooling",
                        choices=["mean_pooling", "max_pooling", "cls", "attention"],
                        help="Pooling strategy")
    parser.add_argument("--classifier_hidden_sizes", type=str, default="512,256",
                        help="Comma-separated list of classifier hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability")
    parser.add_argument("--use_cross_attention", action="store_true",
                        help="Use cross-attention between premise and hypothesis")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm (0 to disable)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="Warmup ratio")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/indo-sbert-advanced",
                        help="Output directory")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # New logger parameter
    parser.add_argument("--use_new_logger", action="store_true",
                        help="Use the new NLILogger")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup either traditional logging or the new NLILogger
    if args.use_new_logger:
        logger = create_logger("advanced-sbert", args.output_dir)
        logger.log_hyperparameters(vars(args))
        
        # Train with new logger
        train_with_nli_logger(args, logger)
    else:
        # Setup traditional logging
        log_file = os.path.join(args.output_dir, 'train.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Arguments: {args}")
        
        # Warn about not using the new logger
        logger.warning("Training without the new NLILogger. Use --use_new_logger for better logging.")
        
        # Train with traditional logging
        logger.error("Traditional logging not implemented. Please use --use_new_logger")


if __name__ == "__main__":
    main()
