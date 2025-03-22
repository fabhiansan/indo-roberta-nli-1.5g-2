"""
Evaluate all checkpoints of the advanced SBERT model on test datasets.
"""

import os
import re
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Import the advanced SBERT model and dataset code
from indo_sbert_advanced_model import SBERTAdvancedModel
from indo_sbert_advanced_dataset import load_indonli_data, AdvancedNLIDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


def find_checkpoints(model_dir):
    """
    Find all checkpoint directories.
    
    Args:
        model_dir: Model directory
        
    Returns:
        List of checkpoint paths
    """
    # Check if directory exists
    if not os.path.exists(model_dir):
        logging.error(f"Directory not found: {model_dir}")
        return []
    
    # Find all subdirectories that match checkpoint-XX pattern
    checkpoints = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and re.match(r"checkpoint-\d+", item):
            checkpoints.append(item_path)
            logging.info(f"Found checkpoint: {item_path}")
    
    # Also include the main model directory as a checkpoint
    if os.path.isdir(model_dir):
        checkpoints.append(model_dir)
        logging.info(f"Added main directory as checkpoint: {model_dir}")
    
    # Sort checkpoints by step number
    def get_step(path):
        if path == model_dir:
            return float('inf')  # Main directory last
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_step)
    
    if not checkpoints:
        logging.warning(f"No checkpoints found in {model_dir}")
    
    return checkpoints


def test_model():
    """
    Test model loading and inference to verify it works.
    
    Returns:
        Boolean indicating if test passed
    """
    try:
        logging.info("Running model test to verify functionality")
        
        # Create a small test input
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("firqaaa/indo-sentence-bert-base")
        
        # Create test inputs
        test_premise = "Ini adalah kalimat tes."
        test_hypothesis = "Ini adalah contoh hipotesis."
        
        # Tokenize
        premise_tokens = tokenizer(test_premise, return_tensors="pt", padding=True, truncation=True)
        hypothesis_tokens = tokenizer(test_hypothesis, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        premise_input_ids = premise_tokens["input_ids"].to(device)
        premise_attention_mask = premise_tokens["attention_mask"].to(device)
        hypothesis_input_ids = hypothesis_tokens["input_ids"].to(device)
        hypothesis_attention_mask = hypothesis_tokens["attention_mask"].to(device)
        
        # Create model
        model = SBERTAdvancedModel(
            model_name="firqaaa/indo-sentence-bert-base",
            pooling_mode="mean_pooling",
            classifier_hidden_sizes=[512, 256],
            dropout=0.2,
            use_cross_attention=True
        )
        model.to(device)
        model.eval()
        
        # Run inference
        with torch.no_grad():
            logits = model(
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask
            )
        
        logging.info(f"Model test successful! Output shape: {logits.shape}")
        return True
    
    except Exception as e:
        logging.error(f"Model test failed: {str(e)}")
        return False


def evaluate_checkpoint_standard(checkpoint_path, split, batch_size=16, max_length=128):
    """
    Evaluate a checkpoint using a standard approach (not loading the full model).
    
    Args:
        checkpoint_path: Path to checkpoint
        split: Dataset split
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of metrics
    """
    try:
        logging.info(f"Evaluating checkpoint (standard approach): {checkpoint_path}")
        logging.info(f"Dataset split: {split}")
        
        # Create a new model instance
        model = SBERTAdvancedModel(
            model_name="firqaaa/indo-sentence-bert-base",
            pooling_mode="mean_pooling",
            classifier_hidden_sizes=[512, 256],
            dropout=0.2,
            use_cross_attention=True
        )
        
        # Try to load state dictionary if available
        pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            logging.info(f"Loading model weights from {pytorch_model_path}")
            try:
                state_dict = torch.load(pytorch_model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                logging.info("Successfully loaded model weights")
            except Exception as e:
                logging.error(f"Error loading model weights: {str(e)}")
        else:
            logging.warning(f"No model weights found at {pytorch_model_path}")
        
        model.to(device)
        model.eval()
        
        # Get checkpoint name
        if checkpoint_path.endswith('/'):
            checkpoint_path = checkpoint_path[:-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("firqaaa/indo-sentence-bert-base")
        logging.info("Successfully loaded tokenizer")
        
        # Load dataset
        examples = load_indonli_data(split)
        logging.info(f"Loaded {len(examples)} examples from {split}")
        
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
        
        # Evaluation
        logging.info(f"Starting evaluation on {len(dataset)} examples")
        all_labels = []
        all_preds = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {checkpoint_name} on {split}")):
            try:
                # Move batch to device
                premise_input_ids = batch["premise_input_ids"].to(device)
                premise_attention_mask = batch["premise_attention_mask"].to(device)
                hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
                hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Get predictions
                with torch.no_grad():
                    logits = model(
                        premise_input_ids=premise_input_ids,
                        premise_attention_mask=premise_attention_mask,
                        hypothesis_input_ids=hypothesis_input_ids,
                        hypothesis_attention_mask=hypothesis_attention_mask
                    )
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    # Add to lists
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds)
                
                # Log progress occasionally
                if batch_idx % 10 == 0:
                    logging.info(f"Processed {batch_idx} batches")
            
            except Exception as batch_error:
                logging.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                continue
        
        # Check if we have predictions
        if len(all_preds) == 0:
            logging.error("No predictions were made during evaluation")
            return None
        
        logging.info(f"Collected {len(all_preds)} predictions and {len(all_labels)} labels")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        
        precision_macro = precision_score(all_labels, all_preds, average="macro")
        precision_weighted = precision_score(all_labels, all_preds, average="weighted")
        
        recall_macro = recall_score(all_labels, all_preds, average="macro")
        recall_weighted = recall_score(all_labels, all_preds, average="weighted")
        
        cm = confusion_matrix(all_labels, all_preds)
        
        logging.info(f"Metrics calculated - Accuracy: {accuracy:.4f}, F1 (weighted): {f1_weighted:.4f}")
        
        # Calculate per-class metrics
        per_class_accuracy = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        for i in range(3):
            class_indices = np.where(np.array(all_labels) == i)[0]
            if len(class_indices) > 0:
                class_preds = np.array(all_preds)[class_indices]
                class_labels = np.array(all_labels)[class_indices]
                per_class_accuracy[i] = accuracy_score(class_labels, class_preds)
                
                # Calculate class precision, recall, and F1 using global metrics
                true_positives = np.sum((np.array(all_preds) == i) & (np.array(all_labels) == i))
                false_positives = np.sum((np.array(all_preds) == i) & (np.array(all_labels) != i))
                false_negatives = np.sum((np.array(all_preds) != i) & (np.array(all_labels) == i))
                
                if true_positives + false_positives > 0:
                    per_class_precision[i] = true_positives / (true_positives + false_positives)
                else:
                    per_class_precision[i] = 0.0
                
                if true_positives + false_negatives > 0:
                    per_class_recall[i] = true_positives / (true_positives + false_negatives)
                else:
                    per_class_recall[i] = 0.0
                
                if per_class_precision[i] + per_class_recall[i] > 0:
                    per_class_f1[i] = 2 * per_class_precision[i] * per_class_recall[i] / (per_class_precision[i] + per_class_recall[i])
                else:
                    per_class_f1[i] = 0.0
        
        # Create metrics dictionary
        metrics = {
            "checkpoint": checkpoint_name,
            "split": split,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "per_class_accuracy": per_class_accuracy,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist()
        }
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error evaluating checkpoint {checkpoint_path} on {split}: {str(e)}")
        return None


def evaluate_checkpoint(checkpoint_path, split, batch_size=16, max_length=128):
    """
    Evaluate a checkpoint on a dataset split.
    
    Args:
        checkpoint_path: Path to checkpoint
        split: Dataset split
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of metrics
    """
    try:
        logging.info(f"Evaluating checkpoint: {checkpoint_path}")
        logging.info(f"Dataset split: {split}")
        
        # Check if checkpoint directory exists and has necessary files
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint directory does not exist: {checkpoint_path}")
            return None
        
        # List files in checkpoint directory for debugging
        logging.info(f"Files in checkpoint directory: {os.listdir(checkpoint_path)}")
        
        try:
            # Try to load model with custom class
            logging.info(f"Attempting to load model from {checkpoint_path}")
            model = SBERTAdvancedModel.from_pretrained(checkpoint_path)
            logging.info("Successfully loaded model")
        except Exception as model_error:
            # Fall back to loading with base model and parameters
            logging.warning(f"Error loading SBERTAdvancedModel: {str(model_error)}")
            logging.info("Creating a new model instance and loading components manually")
            
            # Create new model with base parameters
            model = SBERTAdvancedModel(
                model_name="firqaaa/indo-sentence-bert-base",
                pooling_mode="mean_pooling",
                classifier_hidden_sizes=[512, 256],
                dropout=0.2,
                use_cross_attention=True
            )
            
            # Check for classifier weights
            classifier_path = os.path.join(checkpoint_path, "classifier.pt")
            if os.path.exists(classifier_path):
                logging.info(f"Loading classifier from {classifier_path}")
                try:
                    model.classifier.load_state_dict(
                        torch.load(classifier_path, map_location=torch.device("cpu"))
                    )
                    logging.info("Successfully loaded classifier weights")
                except Exception as e:
                    logging.error(f"Error loading classifier weights: {str(e)}")
            else:
                logging.warning(f"No classifier weights found at {classifier_path}")
        
        model.to(device)
        model.eval()
        
        # Get checkpoint name
        if checkpoint_path.endswith('/'):
            checkpoint_path = checkpoint_path[:-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        
        # Get model type (for logging)
        model_type = "advanced"
        logging.info(f"Model type: {model_type}")
        
        # Load tokenizer from the original model name, not from checkpoint
        tokenizer = AutoTokenizer.from_pretrained("firqaaa/indo-sentence-bert-base")
        logging.info("Successfully loaded tokenizer")
        
        # Load dataset
        examples = load_indonli_data(split)
        logging.info(f"Loaded {len(examples)} examples from {split}")
        
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
        
        # Evaluation
        logging.info(f"Starting evaluation on {len(dataset)} examples")
        all_labels = []
        all_preds = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {checkpoint_name} on {split}")):
            try:
                # Move batch to device
                premise_input_ids = batch["premise_input_ids"].to(device)
                premise_attention_mask = batch["premise_attention_mask"].to(device)
                hypothesis_input_ids = batch["hypothesis_input_ids"].to(device)
                hypothesis_attention_mask = batch["hypothesis_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Get predictions
                with torch.no_grad():
                    logits = model(
                        premise_input_ids=premise_input_ids,
                        premise_attention_mask=premise_attention_mask,
                        hypothesis_input_ids=hypothesis_input_ids,
                        hypothesis_attention_mask=hypothesis_attention_mask
                    )
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    # Add to lists
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds)
                
                # Log progress occasionally
                if batch_idx % 10 == 0:
                    logging.info(f"Processed {batch_idx} batches")
            
            except Exception as batch_error:
                logging.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                continue
        
        # Check if we have predictions
        if len(all_preds) == 0:
            logging.error("No predictions were made during evaluation")
            return None
        
        logging.info(f"Collected {len(all_preds)} predictions and {len(all_labels)} labels")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        
        precision_macro = precision_score(all_labels, all_preds, average="macro")
        precision_weighted = precision_score(all_labels, all_preds, average="weighted")
        
        recall_macro = recall_score(all_labels, all_preds, average="macro")
        recall_weighted = recall_score(all_labels, all_preds, average="weighted")
        
        cm = confusion_matrix(all_labels, all_preds)
        
        logging.info(f"Metrics calculated - Accuracy: {accuracy:.4f}, F1 (weighted): {f1_weighted:.4f}")
        
        # Calculate per-class metrics
        per_class_accuracy = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        for i in range(3):
            class_indices = np.where(np.array(all_labels) == i)[0]
            if len(class_indices) > 0:
                class_preds = np.array(all_preds)[class_indices]
                class_labels = np.array(all_labels)[class_indices]
                per_class_accuracy[i] = accuracy_score(class_labels, class_preds)
                
                # Calculate class precision, recall, and F1 using global metrics
                true_positives = np.sum((np.array(all_preds) == i) & (np.array(all_labels) == i))
                false_positives = np.sum((np.array(all_preds) == i) & (np.array(all_labels) != i))
                false_negatives = np.sum((np.array(all_preds) != i) & (np.array(all_labels) == i))
                
                if true_positives + false_positives > 0:
                    per_class_precision[i] = true_positives / (true_positives + false_positives)
                else:
                    per_class_precision[i] = 0.0
                
                if true_positives + false_negatives > 0:
                    per_class_recall[i] = true_positives / (true_positives + false_negatives)
                else:
                    per_class_recall[i] = 0.0
                
                if per_class_precision[i] + per_class_recall[i] > 0:
                    per_class_f1[i] = 2 * per_class_precision[i] * per_class_recall[i] / (per_class_precision[i] + per_class_recall[i])
                else:
                    per_class_f1[i] = 0.0
        
        # Create metrics dictionary
        metrics = {
            "checkpoint": checkpoint_name,
            "split": split,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "per_class_accuracy": per_class_accuracy,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist()
        }
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error evaluating checkpoint {checkpoint_path} on {split}: {str(e)}")
        return None


def format_metrics(metrics):
    """
    Format metrics as a string.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Formatted string
    """
    if metrics is None:
        return "Error evaluating checkpoint"
    
    # Convert class indices to names
    class_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    # Format confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    cm_str = "Confusion Matrix:\n"
    cm_str += "                 Predicted\n"
    cm_str += "                 E     N     C\n"
    cm_str += "Actual E     {:5d} {:5d} {:5d}\n".format(cm[0, 0], cm[0, 1], cm[0, 2])
    cm_str += "       N     {:5d} {:5d} {:5d}\n".format(cm[1, 0], cm[1, 1], cm[1, 2])
    cm_str += "       C     {:5d} {:5d} {:5d}\n".format(cm[2, 0], cm[2, 1], cm[2, 2])
    
    # Format per-class metrics
    per_class_metrics = "Per-Class Metrics:\n"
    for i in range(3):
        per_class_metrics += f"{class_names[i].capitalize():<13}"
        per_class_metrics += f"Accuracy: {metrics['per_class_accuracy'].get(i, 0.0):.4f}  "
        per_class_metrics += f"Precision: {metrics['per_class_precision'].get(i, 0.0):.4f}  "
        per_class_metrics += f"Recall: {metrics['per_class_recall'].get(i, 0.0):.4f}  "
        per_class_metrics += f"F1: {metrics['per_class_f1'].get(i, 0.0):.4f}\n"
    
    # Format overall metrics
    formatted = f"Checkpoint: {metrics['checkpoint']}\n"
    formatted += f"Dataset Split: {metrics['split']}\n\n"
    formatted += f"Overall Metrics:\n"
    formatted += f"Accuracy: {metrics['accuracy']:.4f}\n"
    formatted += f"F1 Score (Macro): {metrics['f1_macro']:.4f}\n"
    formatted += f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n"
    formatted += f"Precision (Macro): {metrics['precision_macro']:.4f}\n"
    formatted += f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n"
    formatted += f"Recall (Macro): {metrics['recall_macro']:.4f}\n"
    formatted += f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n\n"
    formatted += per_class_metrics + "\n"
    formatted += cm_str
    
    return formatted


def create_summary_table(metrics_list):
    """
    Create a summary table from a list of metrics.
    
    Args:
        metrics_list: List of metrics dictionaries
        
    Returns:
        Formatted summary table
    """
    # Filter out None values
    metrics_list = [m for m in metrics_list if m is not None]
    
    if not metrics_list:
        return "No valid metrics to summarize"
    
    # Group by checkpoint and split
    grouped = {}
    for metrics in metrics_list:
        checkpoint = metrics["checkpoint"]
        split = metrics["split"]
        if checkpoint not in grouped:
            grouped[checkpoint] = {}
        grouped[checkpoint][split] = metrics
    
    # Create table header
    header = "Checkpoint            "
    for split in ["test_lay", "test_expert"]:
        header += f"{split:<18} "
    header += "\n"
    header += "                     "
    for _ in range(2):
        header += "Acc    F1     Prec   Rec    "
    
    # Create table rows
    rows = []
    for checkpoint, splits in grouped.items():
        # Extract step number for sorting
        if checkpoint.startswith("checkpoint-"):
            step = int(checkpoint.split("-")[1])
        else:
            step = float('inf')  # Final model
        
        row = f"{checkpoint:<20} "
        for split in ["test_lay", "test_expert"]:
            if split in splits:
                metrics = splits[split]
                row += f"{metrics['accuracy']:.4f} {metrics['f1_weighted']:.4f} "
                row += f"{metrics['precision_weighted']:.4f} {metrics['recall_weighted']:.4f} "
            else:
                row += "N/A    N/A    N/A    N/A    "
        
        rows.append((step, row))
    
    # Sort rows by step number
    rows.sort()
    
    # Combine header and rows
    table = header + "\n" + "\n".join(row for _, row in rows)
    
    return table


def main(args):
    """
    Main function.
    
    Args:
        args: Command-line arguments
    """
    # Get directories
    model_dir = args.model_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run model test first to verify functionality
    test_passed = test_model()
    if not test_passed:
        logging.warning("Model test failed - evaluation may not work properly")
    
    # Find checkpoints
    checkpoints = find_checkpoints(model_dir)
    logging.info(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    # Evaluate each checkpoint on each split
    all_metrics = []
    
    for checkpoint in checkpoints:
        for split in ["test_lay", "test_expert"]:
            try:
                # First try with the standard evaluation
                metrics = evaluate_checkpoint_standard(
                    checkpoint_path=checkpoint,
                    split=split,
                    batch_size=args.batch_size,
                    max_length=args.max_length
                )
                
                # If standard evaluation fails, try with fallback
                if metrics is None:
                    logging.warning("Standard evaluation failed, trying fallback method...")
                    metrics = evaluate_checkpoint(
                        checkpoint_path=checkpoint,
                        split=split,
                        batch_size=args.batch_size,
                        max_length=args.max_length
                    )
                
                if metrics:
                    all_metrics.append(metrics)
                    
                    # Format metrics
                    formatted = format_metrics(metrics)
                    
                    # Save detailed report
                    checkpoint_name = os.path.basename(checkpoint)
                    report_path = os.path.join(output_dir, f"{checkpoint_name}_{split}_report.txt")
                    with open(report_path, "w") as f:
                        f.write(formatted)
                    
                    logging.info(f"Saved report to {report_path}")
            
            except Exception as e:
                logging.error(f"Error evaluating {checkpoint} on {split}: {str(e)}")
    
    # Create summary table
    summary = create_summary_table(all_metrics)
    summary_path = os.path.join(output_dir, "advanced_sbert_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    
    logging.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate advanced SBERT checkpoints")
    
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results_advanced",
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    main(args)
