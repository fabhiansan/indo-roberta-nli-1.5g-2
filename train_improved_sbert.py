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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_sbert_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get premise, hypothesis and label
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = self.label_map.get(item["label"], -1)
        
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
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            premise_input_ids=premise_input_ids,
            premise_attention_mask=premise_attention_mask,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask
        )
        
        # Compute loss
        loss = criterion(logits, labels)
        
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
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Try loading from local directories
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
    
    # Create datasets
    train_dataset = NLIDataset(train_dataset, tokenizer, max_length=args.max_length)
    validation_dataset = NLIDataset(validation_dataset, tokenizer, max_length=args.max_length)
    test_lay_dataset = NLIDataset(test_lay_dataset, tokenizer, max_length=args.max_length)
    test_expert_dataset = NLIDataset(test_expert_dataset, tokenizer, max_length=args.max_length)
    
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
    
    # Training loop
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
    
    args = parser.parse_args()
    
    train(args)
