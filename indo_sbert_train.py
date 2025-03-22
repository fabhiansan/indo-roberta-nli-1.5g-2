"""
Fine-tuning script for BERT-based sentence embeddings on IndoNLI dataset,
implemented directly with transformers library (without sentence-transformers).
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from utils import set_seed, setup_logging


class NLIDataset(Dataset):
    """Dataset for NLI task."""
    
    def __init__(self, examples, tokenizer, max_length=128):
        """
        Initialize NLI dataset.
        
        Args:
            examples: List of examples with premise, hypothesis, and label
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Get premise and hypothesis
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        
        # Get label (convert to int if needed)
        label = example["label"]
        if isinstance(label, str):
            label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
            label = label_map.get(label, 0)
        
        # Tokenize inputs
        encoded_premise = self.tokenizer(
            premise,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoded_hypothesis = self.tokenizer(
            hypothesis,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove the batch dimension
        for key in encoded_premise:
            encoded_premise[key] = encoded_premise[key].squeeze(0)
        
        for key in encoded_hypothesis:
            encoded_hypothesis[key] = encoded_hypothesis[key].squeeze(0)
        
        return {
            "premise_input_ids": encoded_premise["input_ids"],
            "premise_attention_mask": encoded_premise["attention_mask"],
            "hypothesis_input_ids": encoded_hypothesis["input_ids"],
            "hypothesis_attention_mask": encoded_hypothesis["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long)
        }


class SBERTModel(nn.Module):
    """BERT model for sentence embeddings."""
    
    def __init__(self, model_name, use_mean_pooling=True):
        """
        Initialize SBERT model.
        
        Args:
            model_name: Pretrained model name or path
            use_mean_pooling: Whether to use mean pooling over token embeddings
        """
        super(SBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.use_mean_pooling = use_mean_pooling
        
        # Add projection layer (optional)
        # self.projection = nn.Linear(self.bert.config.hidden_size, 768)
        
        # Classification layer for NLI
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, 3)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings.
        
        Args:
            token_embeddings: Token embeddings from BERT
            attention_mask: Attention mask
            
        Returns:
            Sentence embedding
        """
        # Expand attention mask to the same shape as token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum the embeddings of tokens with attention
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the number of tokens with attention
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Divide to get mean
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            
        Returns:
            Sentence embedding
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get embeddings
        if self.use_mean_pooling:
            # Mean pooling
            embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            # CLS token
            embeddings = outputs.last_hidden_state[:, 0]
        
        # Optional projection
        # embeddings = self.projection(embeddings)
        
        return embeddings
    
    def predict(self, premise_input_ids, premise_attention_mask, 
                hypothesis_input_ids, hypothesis_attention_mask):
        """
        Predict NLI class.
        
        Args:
            premise_input_ids: Input ids for premise
            premise_attention_mask: Attention mask for premise
            hypothesis_input_ids: Input ids for hypothesis
            hypothesis_attention_mask: Attention mask for hypothesis
            
        Returns:
            Logits for entailment, neutral, contradiction
        """
        # Get embeddings
        premise_embedding = self.forward(premise_input_ids, premise_attention_mask)
        hypothesis_embedding = self.forward(hypothesis_input_ids, hypothesis_attention_mask)
        
        # Concatenate embeddings with element-wise difference and product
        diff = torch.abs(premise_embedding - hypothesis_embedding)
        prod = premise_embedding * hypothesis_embedding
        concatenated = torch.cat([premise_embedding, hypothesis_embedding, diff], dim=1)
        
        # Get logits
        logits = self.classifier(concatenated)
        
        return logits


def load_indonli_data(split="train"):
    """
    Load the IndoNLI dataset for the specified split.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test_lay', 'test_expert')
        
    Returns:
        List of examples
    """
    dataset = load_dataset("afaji/indonli", split=split)
    logging.info(f"Loaded {len(dataset)} examples from {split} split")
    
    return dataset


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model.predict(
            batch["premise_input_ids"],
            batch["premise_attention_mask"],
            batch["hypothesis_input_ids"],
            batch["hypothesis_attention_mask"]
        )
        
        # Calculate loss
        loss = F.cross_entropy(logits, batch["label"])
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluate model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            logits = model.predict(
                batch["premise_input_ids"],
                batch["premise_attention_mask"],
                batch["hypothesis_input_ids"],
                batch["hypothesis_attention_mask"]
            )
            
            # Calculate loss
            loss = F.cross_entropy(logits, batch["label"])
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), accuracy


def train_model(
    model_name="firqaaa/indo-sentence-bert-base",
    output_path="./outputs/indo-sbert-nli-custom",
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    max_seq_length=128,
    warmup_ratio=0.1,
    seed=42,
    use_mean_pooling=True
):
    """
    Train a SBERT model on the IndoNLI dataset.
    
    Args:
        model_name: Pre-trained model name or path
        output_path: Directory to save the model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        warmup_ratio: Ratio of warmup steps
        seed: Random seed
        use_mean_pooling: Whether to use mean pooling
    """
    # Set random seed
    set_seed(seed)
    
    # Set up logging
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load tokenizer
    logging.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    logging.info("Loading datasets")
    train_dataset = load_indonli_data("train")
    validation_dataset = load_indonli_data("validation")
    test_lay_dataset = load_indonli_data("test_lay")
    test_expert_dataset = load_indonli_data("test_expert")
    
    # Create datasets
    train_dataset = NLIDataset(train_dataset, tokenizer, max_length=max_seq_length)
    validation_dataset = NLIDataset(validation_dataset, tokenizer, max_length=max_seq_length)
    test_lay_dataset = NLIDataset(test_lay_dataset, tokenizer, max_length=max_seq_length)
    test_expert_dataset = NLIDataset(test_expert_dataset, tokenizer, max_length=max_seq_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_lay_dataloader = DataLoader(test_lay_dataset, batch_size=batch_size)
    test_expert_dataloader = DataLoader(test_expert_dataset, batch_size=batch_size)
    
    # Load model
    logging.info(f"Loading model: {model_name}")
    model = SBERTModel(model_name, use_mean_pooling=use_mean_pooling)
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total steps and warmup steps
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logging.info(f"Starting training for {num_epochs} epochs")
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        logging.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, validation_dataloader, device)
        logging.info(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_path, f"checkpoint-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        model.bert.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save classifier weights separately
        torch.save(model.classifier.state_dict(), os.path.join(checkpoint_dir, "classifier.pt"))
        
        # Save configuration
        with open(os.path.join(checkpoint_dir, "config.txt"), "w") as f:
            f.write(f"model_name: {model_name}\n")
            f.write(f"use_mean_pooling: {use_mean_pooling}\n")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logging.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
            
            # Save best model
            best_dir = os.path.join(output_path, "best")
            os.makedirs(best_dir, exist_ok=True)
            
            model.bert.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save(model.classifier.state_dict(), os.path.join(best_dir, "classifier.pt"))
    
    # Evaluate on test sets
    logging.info("Evaluating on test_lay split")
    test_lay_loss, test_lay_accuracy = evaluate(model, test_lay_dataloader, device)
    logging.info(f"Test Lay loss: {test_lay_loss:.4f}, accuracy: {test_lay_accuracy:.4f}")
    
    logging.info("Evaluating on test_expert split")
    test_expert_loss, test_expert_accuracy = evaluate(model, test_expert_dataloader, device)
    logging.info(f"Test Expert loss: {test_expert_loss:.4f}, accuracy: {test_expert_accuracy:.4f}")
    
    # Save evaluation results
    with open(os.path.join(output_path, "evaluation_results.txt"), "w") as f:
        f.write(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}\n")
        f.write(f"Test Lay loss: {test_lay_loss:.4f}, accuracy: {test_lay_accuracy:.4f}\n")
        f.write(f"Test Expert loss: {test_expert_loss:.4f}, accuracy: {test_expert_accuracy:.4f}\n")
    
    logging.info(f"Model saved to {output_path}")
    return model


def load_model(model_path, device=None):
    """
    Load a trained SBERT model.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    with open(os.path.join(model_path, "config.txt"), "r") as f:
        config = {}
        for line in f:
            key, value = line.strip().split(": ")
            config[key] = value
    
    # Get model name and use_mean_pooling
    model_name = config.get("model_name", "firqaaa/indo-sentence-bert-base")
    use_mean_pooling = config.get("use_mean_pooling", "True") == "True"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create model
    model = SBERTModel(model_path, use_mean_pooling=use_mean_pooling)
    
    # Load classifier weights
    classifier_path = os.path.join(model_path, "classifier.pt")
    if os.path.exists(classifier_path):
        classifier_state_dict = torch.load(classifier_path, map_location=device)
        model.classifier.load_state_dict(classifier_state_dict)
    
    model.to(device)
    
    return model, tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SBERT-like model on IndoNLI dataset")
    
    parser.add_argument("--model_name", type=str, default="firqaaa/indo-sentence-bert-base",
                        help="Pre-trained model name or path")
    parser.add_argument("--output_path", type=str, default="./outputs/indo-sbert-nli-custom",
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_mean_pooling", action="store_true",
                        help="Use CLS token instead of mean pooling")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    train_model(
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        use_mean_pooling=not args.no_mean_pooling
    )


if __name__ == "__main__":
    main()
