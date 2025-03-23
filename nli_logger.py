"""
Comprehensive logging and evaluation utilities for IndoNLI models.
This module provides consistent logging, evaluation, and reporting across different model types.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
import torch
from pathlib import Path


class NLILogger:
    """
    Unified logger for all NLI models that provides consistent logging,
    evaluation metrics, and result visualization.
    """
    
    def __init__(self, model_name, output_dir="./logs", log_to_file=True, log_level=logging.INFO):
        """
        Initialize the NLI logger.
        
        Args:
            model_name: Name of the model being trained/evaluated
            output_dir: Directory to save logs and results
            log_to_file: Whether to save logs to a file
            log_level: Logging level
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results", model_name)
        self.figures_dir = os.path.join(output_dir, "figures", model_name)
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger(model_name, output_dir, log_to_file, log_level)
        
        # Store model results
        self.results = {
            "train_loss": [],
            "val_metrics": [],
            "test_lay_metrics": {},
            "test_expert_metrics": {},
            "checkpoints": []
        }
        
        # Label mapping
        self.label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        self.logger.info(f"Initialized NLI logger for model: {model_name}")
    
    def _setup_logger(self, model_name, output_dir, log_to_file, log_level):
        """Set up logging configuration."""
        logger = logging.getLogger(f"nli_logger_{model_name}")
        logger.setLevel(log_level)
        logger.handlers = []  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if requested
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(output_dir, f"{model_name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_hyperparameters(self, hparams):
        """
        Log hyperparameters for the model.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.logger.info("Hyperparameters:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save hyperparameters to file
        with open(os.path.join(self.results_dir, "hyperparameters.json"), "w") as f:
            json.dump(hparams, f, indent=2)
    
    def log_train_step(self, epoch, step, loss, lr=None, additional_metrics=None):
        """
        Log a training step.
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Training loss
            lr: Current learning rate
            additional_metrics: Dictionary of additional metrics to log
        """
        # Build log message
        msg = f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}"
        if lr is not None:
            msg += f", LR: {lr:.6f}"
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, float):
                    msg += f", {key}: {value:.4f}"
                else:
                    msg += f", {key}: {value}"
        
        self.logger.info(msg)
        
        # Store loss for later analysis
        self.results["train_loss"].append({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "lr": lr
        })
    
    def log_epoch(self, epoch, train_loss, val_metrics, checkpoint_path=None):
        """
        Log the results of a training epoch.
        
        Args:
            epoch: Current epoch
            train_loss: Average training loss for the epoch
            val_metrics: Dictionary of validation metrics
            checkpoint_path: Path to the saved checkpoint
        """
        # Log epoch summary
        accuracy = val_metrics.get("accuracy", 0)
        f1 = val_metrics.get("macro_f1", 0)
        
        self.logger.info(f"Epoch {epoch} completed:")
        self.logger.info(f"  Train Loss: {train_loss:.4f}")
        self.logger.info(f"  Validation Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Validation F1: {f1:.4f}")
        
        if checkpoint_path:
            self.logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Store validation metrics
        self.results["val_metrics"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "checkpoint_path": checkpoint_path
        })
        
        if checkpoint_path:
            self.results["checkpoints"].append({
                "epoch": epoch,
                "path": checkpoint_path,
                "metrics": val_metrics
            })
    
    def log_evaluation(self, dataset_name, predictions, labels, checkpoint_name=None):
        """
        Log evaluation results for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "test_lay", "test_expert")
            predictions: Model predictions
            labels: True labels
            checkpoint_name: Name of the checkpoint (optional)
        
        Returns:
            Dictionary of metrics
        """
        # Safety check for empty predictions or labels
        if not predictions or not labels or len(predictions) == 0 or len(labels) == 0:
            self.logger.warning(f"Empty predictions or labels for {dataset_name}. Cannot compute metrics.")
            return {"accuracy": 0, "macro_f1": 0, "precision": 0, "recall": 0}
            
        # Convert to numpy arrays if they are tensors
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        class_f1 = f1_score(labels, predictions, average=None)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        conf_mat = confusion_matrix(labels, predictions)
        
        # Get per-class metrics
        class_precision = precision_score(labels, predictions, average=None)
        class_recall = recall_score(labels, predictions, average=None)
        
        # Log metrics
        self.logger.info(f"Evaluation on {dataset_name}:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Macro F1: {macro_f1:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        
        # Log per-class metrics
        for i in range(len(class_f1)):
            label_name = self.label_map[i]
            self.logger.info(f"  {label_name} - F1: {class_f1[i]:.4f}, "
                             f"Precision: {class_precision[i]:.4f}, "
                             f"Recall: {class_recall[i]:.4f}")
        
        # Create report
        metrics = {
            "dataset": dataset_name,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
            "class_metrics": {
                self.label_map[i]: {
                    "f1": float(class_f1[i]),
                    "precision": float(class_precision[i]),
                    "recall": float(class_recall[i])
                } for i in range(len(class_f1))
            },
            "confusion_matrix": conf_mat.tolist()
        }
        
        # Add checkpoint info if provided
        if checkpoint_name:
            metrics["checkpoint"] = checkpoint_name
        
        # Store metrics
        result_key = f"test_{dataset_name}_metrics" if dataset_name in ["lay", "expert"] else f"{dataset_name}_metrics"
        self.results[result_key] = metrics
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_str = f"_{checkpoint_name}" if checkpoint_name else ""
        results_file = os.path.join(
            self.results_dir, f"{dataset_name}{checkpoint_str}_{timestamp}_metrics.json"
        )
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(
            conf_mat, 
            dataset_name=dataset_name,
            checkpoint_name=checkpoint_name
        )
        
        return metrics
    
    def _plot_confusion_matrix(self, conf_mat, dataset_name, checkpoint_name=None):
        """
        Plot and save confusion matrix.
        
        Args:
            conf_mat: Confusion matrix
            dataset_name: Name of the dataset
            checkpoint_name: Name of the checkpoint (optional)
        """
        # Check if the confusion matrix is empty or has zero size
        if conf_mat is None or conf_mat.size == 0 or np.sum(conf_mat) == 0:
            self.logger.warning(f"Empty confusion matrix for {dataset_name}. Skipping plot.")
            return
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_mat, 
            annot=True, 
            fmt="d",
            cmap="Blues",
            xticklabels=[self.label_map[i] for i in range(len(self.label_map))],
            yticklabels=[self.label_map[i] for i in range(len(self.label_map))]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {dataset_name}")
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_str = f"_{checkpoint_name}" if checkpoint_name else ""
        fig_path = os.path.join(
            self.figures_dir, 
            f"{dataset_name}{checkpoint_str}_{timestamp}_confusion_matrix.png"
        )
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {fig_path}")
    
    def plot_training_curve(self):
        """
        Plot and save training loss and validation metrics curves.
        """
        if not self.results["train_loss"] or not self.results["val_metrics"]:
            self.logger.warning("Not enough data to plot training curves")
            return
        
        # Extract data
        epochs = [x["epoch"] for x in self.results["val_metrics"]]
        train_losses = [x["train_loss"] for x in self.results["val_metrics"]]
        val_accuracies = [x["accuracy"] for x in self.results["val_metrics"]]
        val_f1s = [x.get("macro_f1", 0) for x in self.results["val_metrics"]]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot validation metrics
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, 'g-', marker='o', label='Validation Accuracy')
        plt.plot(epochs, val_f1s, 'r-', marker='s', label='Validation F1')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.figures_dir, f"training_curve_{timestamp}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Training curves saved to {fig_path}")
    
    def compare_checkpoints(self, dataset_name="test_lay"):
        """
        Compare metrics across different checkpoints for a given dataset.
        
        Args:
            dataset_name: Name of the dataset to compare on
        """
        if not self.results["checkpoints"]:
            self.logger.warning("No checkpoints available for comparison")
            return
        
        # Create DataFrame for comparison
        data = []
        for checkpoint in self.results["checkpoints"]:
            metrics = checkpoint["metrics"]
            data.append({
                "Checkpoint": f"Epoch {checkpoint['epoch']}",
                "Accuracy": metrics.get("accuracy", 0),
                "F1": metrics.get("macro_f1", 0),
                "Precision": metrics.get("precision", 0) if "precision" in metrics else 0,
                "Recall": metrics.get("recall", 0) if "recall" in metrics else 0
            })
        
        df = pd.DataFrame(data)
        
        # Create table
        self.logger.info(f"Checkpoint comparison on {dataset_name}:")
        self.logger.info("\n" + df.to_string(index=False))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            self.results_dir, 
            f"{dataset_name}_checkpoint_comparison_{timestamp}.csv"
        )
        df.to_csv(csv_path, index=False)
        
        # Create bar plot for comparison
        plt.figure(figsize=(12, 6))
        
        # Melt DataFrame for seaborn
        df_melted = pd.melt(
            df, 
            id_vars=["Checkpoint"], 
            value_vars=["Accuracy", "F1", "Precision", "Recall"],
            var_name="Metric", 
            value_name="Score"
        )
        
        # Plot
        sns.barplot(x="Checkpoint", y="Score", hue="Metric", data=df_melted)
        plt.title(f"Checkpoint Comparison - {dataset_name}")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(
            self.figures_dir, 
            f"{dataset_name}_checkpoint_comparison_{timestamp}.png"
        )
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Checkpoint comparison saved to {csv_path} and {fig_path}")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of all results.
        
        Returns:
            Summary report as a string
        """
        # Create report
        report = []
        report.append(f"# {self.model_name} - Summary Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add best validation metrics
        if self.results["val_metrics"]:
            best_val = max(self.results["val_metrics"], key=lambda x: x.get("accuracy", 0))
            report.append("## Best Validation Results")
            report.append(f"Epoch: {best_val['epoch']}")
            report.append(f"Accuracy: {best_val.get('accuracy', 0):.4f}")
            report.append(f"F1 Score: {best_val.get('macro_f1', 0):.4f}")
            report.append("")
        
        # Add test metrics
        if self.results["test_lay_metrics"]:
            report.append("## Test Lay Dataset Results")
            metrics = self.results["test_lay_metrics"]
            report.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            report.append(f"F1 Score: {metrics.get('macro_f1', 0):.4f}")
            report.append(f"Precision: {metrics.get('precision', 0):.4f}")
            report.append(f"Recall: {metrics.get('recall', 0):.4f}")
            report.append("")
        
        if self.results["test_expert_metrics"]:
            report.append("## Test Expert Dataset Results")
            metrics = self.results["test_expert_metrics"]
            report.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            report.append(f"F1 Score: {metrics.get('macro_f1', 0):.4f}")
            report.append(f"Precision: {metrics.get('precision', 0):.4f}")
            report.append(f"Recall: {metrics.get('recall', 0):.4f}")
            report.append("")
        
        # Add per-class metrics
        if self.results["test_lay_metrics"] and "class_metrics" in self.results["test_lay_metrics"]:
            report.append("## Per-Class Metrics (Test Lay)")
            for label, metrics in self.results["test_lay_metrics"]["class_metrics"].items():
                report.append(f"### {label}")
                report.append(f"F1 Score: {metrics.get('f1', 0):.4f}")
                report.append(f"Precision: {metrics.get('precision', 0):.4f}")
                report.append(f"Recall: {metrics.get('recall', 0):.4f}")
                report.append("")
        
        # Save report
        report_str = "\n".join(report)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"summary_report_{timestamp}.md")
        
        with open(report_path, "w") as f:
            f.write(report_str)
        
        self.logger.info(f"Summary report saved to {report_path}")
        return report_str
    
    def log_dataset_statistics(self, dataset_splits):
        """
        Log statistics about the datasets.
        
        Args:
            dataset_splits: Dictionary mapping split names to datasets
        """
        self.logger.info("Dataset statistics:")
        
        for split_name, dataset in dataset_splits.items():
            # Get total size
            self.logger.info(f"  {split_name}: {len(dataset)} examples")
            
            # Count labels if dataset has label attribute
            if hasattr(dataset, "features") and "label" in dataset.features:
                label_counts = {}
                for item in dataset:
                    label = item["label"]
                    if isinstance(label, str):
                        label = self.inv_label_map.get(label.lower(), -1)
                    
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                for label, count in label_counts.items():
                    label_name = self.label_map.get(label, f"Unknown ({label})")
                    self.logger.info(f"    {label_name}: {count} examples ({count/len(dataset)*100:.1f}%)")
    
    def create_checkpoint_comparison_table(self, dataset="test_lay", metric="accuracy"):
        """
        Create a formatted comparison table of checkpoints.
        
        Args:
            dataset: Dataset to compare on ("test_lay" or "test_expert")
            metric: Metric to compare ("accuracy", "macro_f1", etc.)
        
        Returns:
            Markdown formatted comparison table
        """
        # Ensure dataset has results
        result_key = f"test_{dataset}_metrics" if dataset in ["lay", "expert"] else f"{dataset}_metrics"
        
        if not self.results["checkpoints"]:
            return "No checkpoints available for comparison"
        
        # Table header
        header = "Checkpoint | " + " | ".join([dataset, metric]) + "\n"
        divider = "--- | " + " | ".join(["---" for _ in range(1)]) + "\n"
        
        rows = []
        for checkpoint in self.results["checkpoints"]:
            metrics = checkpoint["metrics"]
            metric_value = metrics.get(metric, 0)
            row = f"Epoch {checkpoint['epoch']} | {metric_value:.4f}"
            rows.append(row)
        
        # Combine
        table = header + divider + "\n".join(rows)
        
        # Save table
        table_path = os.path.join(
            self.results_dir, 
            f"{dataset}_checkpoint_comparison_{metric}.md"
        )
        with open(table_path, "w") as f:
            f.write(table)
        
        return table


def create_logger(model_name, output_dir="./logs"):
    """
    Convenience function to create a new NLI logger.
    
    Args:
        model_name: Name of the model
        output_dir: Output directory for logs and results
        
    Returns:
        NLILogger instance
    """
    return NLILogger(model_name, output_dir)
