"""
Script to evaluate all checkpoints for both SBERT model types (classifier and similarity)
on the test_lay and test_expert datasets, saving results to text files.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from pathlib import Path
from utils import set_seed, setup_logging


class NLIDataset:
    """Dataset for NLI task, supporting both classifier and similarity models."""
    
    def __init__(self, examples, tokenizer, max_length=128, model_type="classifier"):
        """
        Initialize NLI dataset.
        
        Args:
            examples: List of examples with premise, hypothesis, and label
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            model_type: Type of model ("classifier" or "similarity")
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
    def __len__(self):
        return len(self.examples)
    
    def preprocess_batch(self, examples):
        """
        Preprocess a batch of examples.
        
        Args:
            examples: List of examples
            
        Returns:
            Processed batch
        """
        # Get premises and hypotheses
        premises = []
        hypotheses = []
        labels = []
        
        for example in examples:
            try:
                # Handle different types of examples
                if isinstance(example, dict):
                    # Dictionary
                    premise = example.get("premise", "")
                    hypothesis = example.get("hypothesis", "")
                    label = example.get("label", 1)  # Default to neutral if missing
                elif hasattr(example, "premise") and hasattr(example, "hypothesis"):
                    # Object with attributes
                    premise = example.premise
                    hypothesis = example.hypothesis
                    label = getattr(example, "label", 1)
                elif isinstance(example, (list, tuple)) and len(example) >= 3:
                    # List or tuple with at least 3 elements
                    premise = example[0]
                    hypothesis = example[1]
                    label = example[2]
                else:
                    # Try to parse as a JSON string
                    try:
                        import json
                        if isinstance(example, str):
                            data = json.loads(example)
                            premise = data.get("premise", "")
                            hypothesis = data.get("hypothesis", "")
                            label = data.get("label", 1)
                        else:
                            # Unknown format, use empty strings
                            premise = ""
                            hypothesis = ""
                            label = 1
                    except (json.JSONDecodeError, TypeError):
                        # Not a valid JSON string
                        premise = ""
                        hypothesis = ""
                        label = 1
                
                # Add to lists
                premises.append(premise)
                hypotheses.append(hypothesis)
                
                # Process the label
                if isinstance(label, str):
                    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
                    labels.append(label_map.get(label, 1))  # Default to neutral if unknown
                else:
                    labels.append(label)
            except Exception as e:
                # Log error and use default values
                logging.warning(f"Error processing example: {e}")
                premises.append("")
                hypotheses.append("")
                labels.append(1)  # Default to neutral
        
        # Tokenize inputs
        encoded_premises = self.tokenizer(
            premises,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoded_hypotheses = self.tokenizer(
            hypotheses,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create batch
        batch = {
            "premise_input_ids": encoded_premises["input_ids"],
            "premise_attention_mask": encoded_premises["attention_mask"],
            "hypothesis_input_ids": encoded_hypotheses["input_ids"],
            "hypothesis_attention_mask": encoded_hypotheses["attention_mask"],
            "labels": torch.tensor(labels)
        }
        
        # Add similarity scores if using similarity model
        if self.model_type == "similarity":
            # Convert labels to similarity scores
            similarity_scores = []
            for label in labels:
                # 0=entailment, 1=neutral, 2=contradiction
                similarity_score = 1.0 if label == 0 else (0.5 if label == 1 else 0.0)
                similarity_scores.append(similarity_score)
            
            batch["similarity"] = torch.tensor(similarity_scores, dtype=torch.float)
        
        return batch
    
    def get_dataloader(self, batch_size=16):
        """
        Get dataloader for the dataset.
        
        Args:
            batch_size: Batch size
            
        Returns:
            DataLoader
        """
        # Process the entire dataset at once to create batches
        num_examples = len(self.examples)
        batch_size = min(batch_size, num_examples)
        
        all_batches = []
        for i in range(0, num_examples, batch_size):
            end_idx = min(i + batch_size, num_examples)
            batch_examples = self.examples[i:end_idx]
            batch = self.preprocess_batch(batch_examples)
            all_batches.append(batch)
        
        return all_batches


class SBERTClassifierModel(nn.Module):
    """BERT model for sentence embeddings with classification head."""
    
    def __init__(self, model_name, classifier_path=None, use_mean_pooling=True):
        """
        Initialize SBERT classifier model.
        
        Args:
            model_name: Pretrained model name or path
            classifier_path: Path to the classifier weights
            use_mean_pooling: Whether to use mean pooling over token embeddings
        """
        super(SBERTClassifierModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, 3)
        self.use_mean_pooling = use_mean_pooling
        
        # Load classifier weights if provided
        if classifier_path and os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device("cpu")))
    
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
        Forward pass to get sentence embeddings.
        
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
            Logits for NLI classes
        """
        # Get embeddings
        premise_embedding = self.forward(premise_input_ids, premise_attention_mask)
        hypothesis_embedding = self.forward(hypothesis_input_ids, hypothesis_attention_mask)
        
        # Get absolute difference
        abs_diff = torch.abs(premise_embedding - hypothesis_embedding)
        
        # Concatenate embeddings and abs diff
        concat_embeddings = torch.cat([premise_embedding, hypothesis_embedding, abs_diff], dim=1)
        
        # Get logits
        logits = self.classifier(concat_embeddings)
        
        return logits


class SBERTSimilarityModel(nn.Module):
    """BERT model for sentence embeddings with similarity approach."""
    
    def __init__(self, model_name, use_mean_pooling=True):
        """
        Initialize SBERT similarity model.
        
        Args:
            model_name: Pretrained model name or path
            use_mean_pooling: Whether to use mean pooling over token embeddings
        """
        super(SBERTSimilarityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.use_mean_pooling = use_mean_pooling
    
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
        
        # Normalize embeddings (important for cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity(self, premise_input_ids, premise_attention_mask, 
                         hypothesis_input_ids, hypothesis_attention_mask):
        """
        Compute similarity between premise and hypothesis.
        
        Args:
            premise_input_ids: Input ids for premise
            premise_attention_mask: Attention mask for premise
            hypothesis_input_ids: Input ids for hypothesis
            hypothesis_attention_mask: Attention mask for hypothesis
            
        Returns:
            Cosine similarity between premise and hypothesis embeddings
        """
        # Get embeddings
        premise_embedding = self.forward(premise_input_ids, premise_attention_mask)
        hypothesis_embedding = self.forward(hypothesis_input_ids, hypothesis_attention_mask)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(premise_embedding, hypothesis_embedding)
        
        return similarity


def load_indonli_data(split="train"):
    """
    Load the IndoNLI dataset for the specified split.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test_lay', 'test_expert')
        
    Returns:
        List of examples
    """
    try:
        dataset = load_dataset("afaji/indonli", split=split)
        logging.info(f"Loaded {len(dataset)} examples from {split} split")
        
        # Convert dataset to list - this helps with compatibility
        dataset_list = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                # Create a simple dict with the needed fields
                example = {
                    "premise": item["premise"] if "premise" in item else "",
                    "hypothesis": item["hypothesis"] if "hypothesis" in item else "",
                    "label": item["label"] if "label" in item else 1  # Default to neutral
                }
                dataset_list.append(example)
            except Exception as e:
                logging.warning(f"Error processing dataset item {i}: {e}")
                # Add a default example
                dataset_list.append({
                    "premise": "",
                    "hypothesis": "",
                    "label": 1  # Default to neutral
                })
        
        return dataset_list
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        # Return an empty list
        return []


def evaluate_classifier_model(model, dataloader, device):
    """
    Evaluate a classifier model.
    
    Args:
        model: Model to evaluate
        dataloader: Batched data
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
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
            
            # Get predictions
            predictions = logits.argmax(dim=1)
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")
    f1_weighted = f1_score(all_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate per-class metrics
    class_names = ["entailment", "neutral", "contradiction"]
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = [j for j, label in enumerate(all_labels) if label == i]
        if class_indices:
            class_predictions = [all_predictions[j] for j in class_indices]
            class_labels = [all_labels[j] for j in class_indices]
            per_class_accuracy[class_name] = accuracy_score(class_labels, class_predictions)
        else:
            per_class_accuracy[class_name] = 0.0
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": conf_matrix,
        "per_class_accuracy": per_class_accuracy
    }


def evaluate_similarity_model(model, dataloader, device):
    """
    Evaluate a similarity model.
    
    Args:
        model: Model to evaluate
        dataloader: Batched data
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_similarity_preds = []
    all_similarity_targets = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            similarity = model.compute_similarity(
                batch["premise_input_ids"],
                batch["premise_attention_mask"],
                batch["hypothesis_input_ids"],
                batch["hypothesis_attention_mask"]
            )
            
            # Convert similarities to NLI labels
            # Threshold-based classification
            predicted_labels = []
            for sim in similarity:
                sim_val = sim.item()
                if sim_val >= 0.7:
                    predicted_labels.append(0)  # entailment
                elif sim_val <= 0.3:
                    predicted_labels.append(2)  # contradiction
                else:
                    predicted_labels.append(1)  # neutral
            
            # Store predictions and targets
            all_similarity_preds.extend(similarity.cpu().numpy())
            all_similarity_targets.extend(batch["similarity"].cpu().numpy())
            all_predictions.extend(predicted_labels)
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")
    f1_weighted = f1_score(all_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate Pearson correlation
    pearson = np.corrcoef(all_similarity_preds, all_similarity_targets)[0, 1]
    
    # Calculate per-class metrics
    class_names = ["entailment", "neutral", "contradiction"]
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = [j for j, label in enumerate(all_labels) if label == i]
        if class_indices:
            class_predictions = [all_predictions[j] for j in class_indices]
            class_labels = [all_labels[j] for j in class_indices]
            per_class_accuracy[class_name] = accuracy_score(class_labels, class_predictions)
        else:
            per_class_accuracy[class_name] = 0.0
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "pearson": pearson,
        "confusion_matrix": conf_matrix,
        "per_class_accuracy": per_class_accuracy
    }


def format_metrics(metrics, model_type="classifier"):
    """
    Format metrics as a string.
    
    Args:
        metrics: Dictionary with evaluation metrics
        model_type: Type of model ("classifier" or "similarity")
        
    Returns:
        Formatted string
    """
    result = []
    result.append(f"Accuracy: {metrics['accuracy']:.4f}")
    result.append(f"F1 Macro: {metrics['f1_macro']:.4f}")
    result.append(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    
    if model_type == "similarity":
        result.append(f"Pearson Correlation: {metrics['pearson']:.4f}")
    
    # Add per-class accuracy
    result.append("\nPer-Class Accuracy:")
    for class_name, acc in metrics["per_class_accuracy"].items():
        result.append(f"  {class_name}: {acc:.4f}")
    
    # Add confusion matrix
    result.append("\nConfusion Matrix:")
    conf_matrix = metrics["confusion_matrix"]
    class_names = ["entailment", "neutral", "contradiction"]
    
    # Format confusion matrix
    header = "       " + " ".join(f"{name[:5]:>7}" for name in class_names)
    result.append(header)
    
    for i, row in enumerate(conf_matrix):
        row_str = f"{class_names[i][:5]:>7} " + " ".join(f"{val:>7}" for val in row)
        result.append(row_str)
    
    return "\n".join(result)


def evaluate_checkpoint(
    checkpoint_path,
    model_type,
    output_dir,
    batch_size=16,
    device=None
):
    """
    Evaluate a checkpoint on test_lay and test_expert datasets.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model_type: Type of model ("classifier" or "similarity")
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Evaluating checkpoint: {checkpoint_path}")
    logging.info(f"Model type: {model_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load model
    if model_type == "classifier":
        # Check if classifier.pt exists
        classifier_path = os.path.join(checkpoint_path, "classifier.pt")
        if os.path.exists(classifier_path):
            logging.info(f"Loading classifier from {classifier_path}")
            model = SBERTClassifierModel(checkpoint_path, classifier_path=classifier_path)
        else:
            # Try to find classifier in parent directory
            classifier_path = os.path.join(os.path.dirname(checkpoint_path), "classifier.pt")
            if os.path.exists(classifier_path):
                logging.info(f"Loading classifier from {classifier_path}")
                model = SBERTClassifierModel(checkpoint_path, classifier_path=classifier_path)
            else:
                logging.warning(f"Classifier weights not found, initializing randomly")
                model = SBERTClassifierModel(checkpoint_path)
        
        model.to(device)
        evaluate_func = evaluate_classifier_model
    else:  # similarity model
        model = SBERTSimilarityModel(checkpoint_path)
        model.to(device)
        evaluate_func = evaluate_similarity_model
    
    # Load datasets
    test_lay_dataset = load_indonli_data("test_lay")
    test_expert_dataset = load_indonli_data("test_expert")
    
    # Create datasets and dataloaders
    test_lay_dataset = NLIDataset(test_lay_dataset, tokenizer, model_type=model_type)
    test_expert_dataset = NLIDataset(test_expert_dataset, tokenizer, model_type=model_type)
    
    test_lay_dataloader = test_lay_dataset.get_dataloader(batch_size=batch_size)
    test_expert_dataloader = test_expert_dataset.get_dataloader(batch_size=batch_size)
    
    # Evaluate on test_lay
    logging.info("Evaluating on test_lay split")
    test_lay_metrics = evaluate_func(model, test_lay_dataloader, device)
    
    # Evaluate on test_expert
    logging.info("Evaluating on test_expert split")
    test_expert_metrics = evaluate_func(model, test_expert_dataloader, device)
    
    # Format results
    checkpoint_name = os.path.basename(checkpoint_path)
    results = {
        "checkpoint": checkpoint_name,
        "model_type": model_type,
        "test_lay": test_lay_metrics,
        "test_expert": test_expert_metrics
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a nice formatted text report
    report = f"Evaluation Report for {model_type.title()} Model: {checkpoint_name}\n"
    report += "=" * 80 + "\n\n"
    
    report += "Test Lay Dataset:\n"
    report += "-" * 40 + "\n"
    report += format_metrics(test_lay_metrics, model_type)
    report += "\n\n"
    
    report += "Test Expert Dataset:\n"
    report += "-" * 40 + "\n"
    report += format_metrics(test_expert_metrics, model_type)
    
    # Save report
    report_path = os.path.join(output_dir, f"{model_type}_{checkpoint_name}_eval.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    logging.info(f"Evaluation results saved to {report_path}")
    
    return results


def evaluate_all_checkpoints(
    classifier_dir=None,
    similarity_dir=None,
    output_dir="./evaluation_results",
    batch_size=16,
    seed=42
):
    """
    Evaluate all checkpoints in the given directories.
    
    Args:
        classifier_dir: Directory containing classifier checkpoints
        similarity_dir: Directory containing similarity checkpoints
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        seed: Random seed
    """
    # Set random seed
    set_seed(seed)
    
    # Set up logging
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Evaluate classifier checkpoints
    if classifier_dir and os.path.exists(classifier_dir):
        logging.info(f"Processing classifier checkpoints from {classifier_dir}")
        
        # Get list of checkpoint directories
        checkpoints = [
            d for d in os.listdir(classifier_dir) 
            if os.path.isdir(os.path.join(classifier_dir, d)) and 
            (d.startswith("checkpoint-") or d == "best")
        ]
        
        if not checkpoints:
            logging.warning(f"No checkpoints found in {classifier_dir}")
        else:
            logging.info(f"Found {len(checkpoints)} classifier checkpoints to evaluate")
            
            # Sort checkpoints by step number (put "best" at the end)
            def sort_key(checkpoint):
                if checkpoint == "best":
                    return float('inf')  # Always at the end
                return int(checkpoint.split("-")[-1]) if checkpoint.split("-")[-1].isdigit() else 0
                
            checkpoints = sorted(checkpoints, key=sort_key)
            
            # Create summary file
            summary_path = os.path.join(output_dir, "classifier_summary.txt")
            with open(summary_path, "w") as summary_file:
                summary_file.write("Classifier Checkpoints Evaluation Summary\n")
                summary_file.write("=" * 80 + "\n\n")
                summary_file.write("Checkpoint | Test Lay Accuracy | Test Expert Accuracy | F1 Macro (Lay/Expert)\n")
                summary_file.write("-" * 80 + "\n")
            
            # Evaluate each checkpoint
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(classifier_dir, checkpoint)
                
                try:
                    results = evaluate_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model_type="classifier",
                        output_dir=output_dir,
                        batch_size=batch_size,
                        device=device
                    )
                    
                    # Add to summary
                    with open(summary_path, "a") as summary_file:
                        lay_acc = results["test_lay"]["accuracy"]
                        expert_acc = results["test_expert"]["accuracy"]
                        lay_f1 = results["test_lay"]["f1_macro"]
                        expert_f1 = results["test_expert"]["f1_macro"]
                        
                        summary_file.write(f"{checkpoint:<15} | {lay_acc:.4f} | {expert_acc:.4f} | {lay_f1:.4f}/{expert_f1:.4f}\n")
                
                except Exception as e:
                    logging.error(f"Error evaluating checkpoint {checkpoint}: {e}")
            
            logging.info(f"Classifier summary saved to {summary_path}")
    else:
        logging.info("No classifier directory provided or directory does not exist")
    
    # Evaluate similarity checkpoints
    if similarity_dir and os.path.exists(similarity_dir):
        logging.info(f"Processing similarity checkpoints from {similarity_dir}")
        
        # Get list of checkpoint directories
        checkpoints = [
            d for d in os.listdir(similarity_dir) 
            if os.path.isdir(os.path.join(similarity_dir, d)) and 
            (d.startswith("checkpoint-") or d == "best")
        ]
        
        if not checkpoints:
            logging.warning(f"No checkpoints found in {similarity_dir}")
        else:
            logging.info(f"Found {len(checkpoints)} similarity checkpoints to evaluate")
            
            # Sort checkpoints by step number (put "best" at the end)
            def sort_key(checkpoint):
                if checkpoint == "best":
                    return float('inf')  # Always at the end
                return int(checkpoint.split("-")[-1]) if checkpoint.split("-")[-1].isdigit() else 0
                
            checkpoints = sorted(checkpoints, key=sort_key)
            
            # Create summary file
            summary_path = os.path.join(output_dir, "similarity_summary.txt")
            with open(summary_path, "w") as summary_file:
                summary_file.write("Similarity Checkpoints Evaluation Summary\n")
                summary_file.write("=" * 80 + "\n\n")
                summary_file.write("Checkpoint | Test Lay Accuracy | Test Expert Accuracy | Pearson (Lay/Expert)\n")
                summary_file.write("-" * 80 + "\n")
            
            # Evaluate each checkpoint
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(similarity_dir, checkpoint)
                
                try:
                    results = evaluate_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model_type="similarity",
                        output_dir=output_dir,
                        batch_size=batch_size,
                        device=device
                    )
                    
                    # Add to summary
                    with open(summary_path, "a") as summary_file:
                        lay_acc = results["test_lay"]["accuracy"]
                        expert_acc = results["test_expert"]["accuracy"]
                        lay_pearson = results["test_lay"]["pearson"]
                        expert_pearson = results["test_expert"]["pearson"]
                        
                        summary_file.write(f"{checkpoint:<15} | {lay_acc:.4f} | {expert_acc:.4f} | {lay_pearson:.4f}/{expert_pearson:.4f}\n")
                
                except Exception as e:
                    logging.error(f"Error evaluating checkpoint {checkpoint}: {e}")
            
            logging.info(f"Similarity summary saved to {summary_path}")
    else:
        logging.info("No similarity directory provided or directory does not exist")
    
    logging.info("Finished evaluating all checkpoints")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate all SBERT checkpoints on IndoNLI test sets")
    
    parser.add_argument(
        "--classifier_dir",
        type=str,
        default=None,
        help="Directory containing classifier model checkpoints",
    )
    parser.add_argument(
        "--similarity_dir",
        type=str,
        default=None,
        help="Directory containing similarity model checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    evaluate_all_checkpoints(
        classifier_dir=args.classifier_dir,
        similarity_dir=args.similarity_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
