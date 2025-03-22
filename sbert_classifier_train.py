"""
Fine-tuning script for SBERT with classifier using IndoNLI dataset.
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sbert_classifier_model import SBERTWithClassifier
from utils import set_seed, setup_logging


class IndoNLIDataset(Dataset):
    """Dataset for IndoNLI with premise-hypothesis pairs and labels."""
    
    def __init__(self, split="train"):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation', 'test_lay', 'test_expert')
        """
        self.dataset = load_dataset("afaji/indonli", split=split)
        
        # Map string labels to integers if needed
        self.label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
        
        logging.info(f"Loaded {len(self.dataset)} examples from {split} split")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        
        # Handle both integer and string labels
        if isinstance(item["label"], int):
            label = item["label"]
        else:
            label = self.label_mapping.get(item["label"], 0)
        
        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label
        }


def collate_fn(batch):
    """
    Collate function for the dataloader.
    
    Args:
        batch: Batch of examples
        
    Returns:
        Dictionary with premises, hypotheses, and labels
    """
    premises = [item["premise"] for item in batch]
    hypotheses = [item["hypothesis"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    return {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": labels
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: SBERTWithClassifier model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use for computation
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        premises = batch["premises"]
        hypotheses = batch["hypotheses"]
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(premises, hypotheses)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: SBERTWithClassifier model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use for computation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            premises = batch["premises"]
            hypotheses = batch["hypotheses"]
            labels = batch["labels"].to(device)
            
            logits = model(premises, hypotheses)
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, 
        all_preds,
        target_names=["entailment", "neutral", "contradiction"],
        output_dict=True
    )
    
    metrics = {
        "loss": val_loss / len(dataloader),
        "accuracy": accuracy,
        "f1": f1,
        "report": report
    }
    
    return metrics


def save_classification_report(report, output_dir, split, epoch=None):
    """
    Save classification report to CSV and text files.
    
    Args:
        report: Classification report (dict)
        output_dir: Output directory
        split: Data split name (e.g., 'validation')
        epoch: Current epoch number
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine filename
    if epoch is not None:
        base_filename = f"{split}_report_epoch_{epoch}"
    else:
        base_filename = f"{split}_report_final"
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(report).transpose()
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    df.to_csv(csv_path)
    
    # Save as text
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w') as f:
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"Class: {class_name}\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("\n")
            else:
                f.write(f"{class_name}: {metrics:.4f}\n")
    
    logging.info(f"Classification report saved to {csv_path} and {txt_path}")


def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_history.png"))
    
    # Plot F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_f1"], label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_history.png"))
    
    logging.info(f"Training history plots saved to {output_dir}")


def train_sbert_classifier(
    sbert_model_name="firqaaa/indo-sentence-bert-base",
    output_dir="./outputs/indo-sbert-classifier-nli",
    batch_size=16,
    num_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_seq_length=128,
    early_stopping_patience=3,
    freeze_sbert=False,
    hidden_size=512,
    dropout_prob=0.1,
    combination_mode="concat",
    seed=42
):
    """
    Train SBERT with classifier for NLI tasks.
    
    Args:
        sbert_model_name: Name or path of the pre-trained SBERT model
        output_dir: Directory to save the model and logs
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        max_seq_length: Maximum sequence length
        early_stopping_patience: Number of epochs to wait before early stopping
        freeze_sbert: Whether to freeze the SBERT parameters
        hidden_size: Size of the hidden layer in the classifier
        dropout_prob: Dropout probability for the classifier
        combination_mode: How to combine premise and hypothesis embeddings
        seed: Random seed
    """
    # Set random seed
    set_seed(seed)
    
    # Set up logging
    logger = setup_logging(log_dir=os.path.join(output_dir, "logs"))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = IndoNLIDataset("train")
    val_dataset = IndoNLIDataset("validation")
    test_lay_dataset = IndoNLIDataset("test_lay")
    test_expert_dataset = IndoNLIDataset("test_expert")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_lay_dataloader = DataLoader(
        test_lay_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_expert_dataloader = DataLoader(
        test_expert_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    logging.info(f"Creating model with {sbert_model_name}...")
    model = SBERTWithClassifier(
        sbert_model_name=sbert_model_name,
        freeze_sbert=freeze_sbert,
        hidden_size=hidden_size,
        dropout_prob=dropout_prob,
        combination_mode=combination_mode,
        device=device
    )
    
    # Set max sequence length if needed
    if max_seq_length != model.sbert.get_max_seq_length():
        logging.info(f"Setting max sequence length to {max_seq_length}")
        model.sbert.max_seq_length = max_seq_length
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }
    
    # Training loop
    logging.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        logging.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, criterion, device)
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        
        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(f"Validation accuracy: {val_accuracy:.4f}")
        logging.info(f"Validation F1: {val_f1:.4f}")
        
        # Save classification report
        save_classification_report(
            val_metrics["report"], 
            os.path.join(output_dir, "reports"), 
            "validation", 
            epoch+1
        )
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(checkpoint_dir)
        logging.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # Save best model
            best_model_dir = os.path.join(output_dir, "best_model")
            model.save(best_model_dir)
            logging.info(f"Best model saved to {best_model_dir}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, "plots"))
    
    # Load best model for final evaluation
    best_model = SBERTWithClassifier.load(os.path.join(output_dir, "best_model"), device=device)
    
    # Final evaluation on validation set
    logging.info("Final evaluation on validation set...")
    val_metrics = evaluate(best_model, val_dataloader, criterion, device)
    logging.info(f"Validation loss: {val_metrics['loss']:.4f}")
    logging.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    logging.info(f"Validation F1: {val_metrics['f1']:.4f}")
    
    # Save final validation report
    save_classification_report(
        val_metrics["report"], 
        os.path.join(output_dir, "reports"), 
        "validation_final"
    )
    
    # Evaluate on test sets
    logging.info("Evaluating on test_lay set...")
    test_lay_metrics = evaluate(best_model, test_lay_dataloader, criterion, device)
    logging.info(f"Test Lay accuracy: {test_lay_metrics['accuracy']:.4f}")
    logging.info(f"Test Lay F1: {test_lay_metrics['f1']:.4f}")
    
    # Save test_lay report
    save_classification_report(
        test_lay_metrics["report"], 
        os.path.join(output_dir, "reports"), 
        "test_lay_final"
    )
    
    logging.info("Evaluating on test_expert set...")
    test_expert_metrics = evaluate(best_model, test_expert_dataloader, criterion, device)
    logging.info(f"Test Expert accuracy: {test_expert_metrics['accuracy']:.4f}")
    logging.info(f"Test Expert F1: {test_expert_metrics['f1']:.4f}")
    
    # Save test_expert report
    save_classification_report(
        test_expert_metrics["report"], 
        os.path.join(output_dir, "reports"), 
        "test_expert_final"
    )
    
    # Save final evaluation results
    results = {
        "validation": {
            "loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "f1": val_metrics["f1"]
        },
        "test_lay": {
            "loss": test_lay_metrics["loss"],
            "accuracy": test_lay_metrics["accuracy"],
            "f1": test_lay_metrics["f1"]
        },
        "test_expert": {
            "loss": test_expert_metrics["loss"],
            "accuracy": test_expert_metrics["accuracy"],
            "f1": test_expert_metrics["f1"]
        }
    }
    
    # Save results as JSON
    import json
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Final results saved to {os.path.join(output_dir, 'evaluation_results.json')}")
    
    return best_model, results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SBERT with classifier on IndoNLI dataset")
    
    parser.add_argument("--sbert_model_name", type=str, default="firqaaa/indo-sentence-bert-base",
                        help="Name or path of the pre-trained SBERT model")
    parser.add_argument("--output_dir", type=str, default="./outputs/indo-sbert-classifier-nli",
                        help="Directory to save the model and logs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of epochs to wait before early stopping")
    parser.add_argument("--freeze_sbert", action="store_true",
                        help="Whether to freeze the SBERT parameters")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Size of the hidden layer in the classifier")
    parser.add_argument("--dropout_prob", type=float, default=0.1,
                        help="Dropout probability for the classifier")
    parser.add_argument("--combination_mode", type=str, default="concat",
                        choices=["concat", "diff", "mult", "concat_diff_mult"],
                        help="How to combine premise and hypothesis embeddings")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Train model
    best_model, results = train_sbert_classifier(
        sbert_model_name=args.sbert_model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        early_stopping_patience=args.early_stopping_patience,
        freeze_sbert=args.freeze_sbert,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout_prob,
        combination_mode=args.combination_mode,
        seed=args.seed
    )
    
    # Print final results
    print("\nFinal Results:")
    print(f"Validation Accuracy: {results['validation']['accuracy']:.4f}")
    print(f"Validation F1: {results['validation']['f1']:.4f}")
    print(f"Test Lay Accuracy: {results['test_lay']['accuracy']:.4f}")
    print(f"Test Lay F1: {results['test_lay']['f1']:.4f}")
    print(f"Test Expert Accuracy: {results['test_expert']['accuracy']:.4f}")
    print(f"Test Expert F1: {results['test_expert']['f1']:.4f}")


if __name__ == "__main__":
    main()
