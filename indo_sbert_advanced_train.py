"""
Training script for the advanced SBERT model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Import our custom modules
from indo_sbert_advanced_model import SBERTAdvancedModel
from indo_sbert_advanced_dataset import load_indonli_data, AdvancedNLIDataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model_name,
    output_dir,
    batch_size=16,
    max_length=128,
    learning_rate=2e-5,
    weight_decay=0.01,
    epochs=3,
    warmup_steps=0,
    save_steps=500,
    pooling_mode="mean_pooling",
    classifier_hidden_sizes=[512, 256],
    dropout=0.2,
    use_cross_attention=True,
    seed=42
):
    """
    Train the advanced SBERT model.
    
    Args:
        model_name: Pretrained model name
        output_dir: Output directory
        batch_size: Batch size
        max_length: Maximum sequence length
        learning_rate: Learning rate
        weight_decay: Weight decay
        epochs: Number of epochs
        warmup_steps: Number of warmup steps
        save_steps: Number of steps between saving
        pooling_mode: Pooling strategy
        classifier_hidden_sizes: Hidden layer sizes for classifier
        dropout: Dropout probability
        use_cross_attention: Whether to use cross-attention
        seed: Random seed
    """
    # Set random seed
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = SBERTAdvancedModel(
        model_name=model_name,
        pooling_mode=pooling_mode,
        classifier_hidden_sizes=classifier_hidden_sizes,
        dropout=dropout,
        use_cross_attention=use_cross_attention
    )
    
    # Log model architecture
    logging.info(f"Model architecture: {model.__class__.__name__}")
    logging.info(f"Pooling mode: {pooling_mode}")
    logging.info(f"Classifier hidden sizes: {classifier_hidden_sizes}")
    logging.info(f"Using cross-attention: {use_cross_attention}")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Load tokenizer from model
    tokenizer = model.bert.tokenizer if hasattr(model.bert, "tokenizer") else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load training data
    logging.info("Loading training data...")
    train_examples = load_indonli_data("train")
    val_examples = load_indonli_data("validation")
    
    # Create datasets
    train_dataset = AdvancedNLIDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        dynamic_padding=True
    )
    
    val_dataset = AdvancedNLIDataset(
        examples=val_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        dynamic_padding=True
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, max_length)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, max_length)
    )
    
    # Log dataset sizes
    logging.info(f"Training on {len(train_dataset)} examples")
    logging.info(f"Validating on {len(val_dataset)} examples")
    
    # Initialize optimizer with weight decay
    # Apply weight decay only to weight matrices, not biases or LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize best metrics
    best_accuracy = 0.0
    best_epoch = 0
    
    # Training loop
    logging.info("Starting training...")
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        logging.info(f"Starting epoch {epoch}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            
            # Increment global step
            global_step += 1
            
            # Save checkpoint if needed
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                logging.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Calculate training metrics
        train_loss /= len(train_dataloader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1_macro = f1_score(train_labels, train_preds, average="macro")
        train_f1_weighted = f1_score(train_labels, train_preds, average="weighted")
        
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, "
                    f"F1 Macro: {train_f1_macro:.4f}, F1 Weighted: {train_f1_weighted:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                premise_input_ids = batch["premise_input_ids"].to(device)
                premise_attention_mask = batch["premise_attention_mask"].to(device)
                hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
                hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                logits = model(
                    premise_input_ids=premise_input_ids,
                    premise_attention_mask=premise_attention_mask,
                    hypothesis_input_ids=hypothesis_input_ids,
                    hypothesis_attention_mask=hypothesis_attention_mask
                )
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Update metrics
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
        
        # Calculate validation metrics
        val_loss /= len(val_dataloader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1_macro = f1_score(val_labels, val_preds, average="macro")
        val_f1_weighted = f1_score(val_labels, val_preds, average="weighted")
        
        logging.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
                    f"F1 Macro: {val_f1_macro:.4f}, F1 Weighted: {val_f1_weighted:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        logging.info(f"Confusion Matrix:\n{cm}")
        
        # Save if best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            
            # Save best model
            model.save_pretrained(output_dir)
            logging.info(f"Saved best model to {output_dir} with accuracy {best_accuracy:.4f}")
        
        # Save checkpoint for current epoch
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
        model.save_pretrained(checkpoint_dir)
        logging.info(f"Saved checkpoint to {checkpoint_dir}")
    
    logging.info(f"Training complete. Best model from epoch {best_epoch} with accuracy {best_accuracy:.4f}")
    
    return model


def evaluate_model(model_path, split="test_lay", batch_size=16, max_length=128):
    """
    Evaluate a trained model on a specific dataset split.
    
    Args:
        model_path: Path to model
        split: Dataset split to evaluate on
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of metrics
    """
    # Load model
    model = SBERTAdvancedModel.from_pretrained(model_path)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load tokenizer from model
    tokenizer = model.bert.tokenizer if hasattr(model.bert, "tokenizer") else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load data
    examples = load_indonli_data(split)
    
    # Create dataset
    dataset = AdvancedNLIDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        dynamic_padding=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, max_length)
    )
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    test_loss = 0.0
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation loop
    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Evaluating on {split}")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
            hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
            
            # Calculate loss
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for i in range(3):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_correct = (np.array(all_preds)[class_mask] == i).sum()
            class_total = class_mask.sum()
            per_class_acc[i] = class_correct / class_total
    
    # Log metrics
    logging.info(f"Evaluation on {split}:")
    logging.info(f"Loss: {test_loss/len(dataloader):.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Macro: {f1_macro:.4f}")
    logging.info(f"F1 Weighted: {f1_weighted:.4f}")
    logging.info(f"Per-class accuracy: {per_class_acc}")
    logging.info(f"Confusion Matrix:\n{cm}")
    
    # Return metrics
    metrics = {
        "loss": test_loss / len(dataloader),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train advanced SBERT model for NLI")
    
    # Input/output arguments
    parser.add_argument("--model_name", type=str, default="firqaaa/indo-sentence-bert-base",
                       help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./outputs/indo-sbert-advanced",
                       help="Output directory")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps between saving checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model parameters
    parser.add_argument("--pooling_mode", type=str, default="mean_pooling",
                       choices=["mean_pooling", "max_pooling", "cls", "attention"],
                       help="Pooling strategy")
    parser.add_argument("--hidden_sizes", type=str, default="512,256",
                       help="Comma-separated list of hidden layer sizes for classifier")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--no_cross_attention", action="store_true",
                       help="Disable cross-attention between sentences")
    
    # Evaluation only
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    parser.add_argument("--eval_split", type=str, default="test_lay",
                       help="Dataset split to evaluate on")
    
    args = parser.parse_args()
    
    # Convert hidden sizes to list
    classifier_hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    
    if args.eval_only:
        # Run evaluation only
        metrics = evaluate_model(
            model_path=args.output_dir,
            split=args.eval_split,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Print metrics
        for k, v in metrics.items():
            if k != "confusion_matrix" and k != "per_class_acc":
                print(f"{k}: {v:.4f}")
    else:
        # Run training
        train_model(
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            pooling_mode=args.pooling_mode,
            classifier_hidden_sizes=classifier_hidden_sizes,
            dropout=args.dropout,
            use_cross_attention=not args.no_cross_attention,
            seed=args.seed
        )
