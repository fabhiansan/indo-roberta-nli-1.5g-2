"""
Trainer module for the Indonesian NLI project.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import csv


class Trainer:
    """
    Trainer class for training and evaluating the RoBERTa NLI model.
    """
    
    def __init__(
        self,
        model,
        data_loaders,
        device=None,
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        output_dir="./outputs"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            data_loaders: Dictionary containing DataLoader objects for each split
            device: Device to train on (will use CUDA if available)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            adam_epsilon: Epsilon for Adam optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for gradient clipping
            output_dir: Directory to save outputs
        """
        self.model = model
        self.data_loaders = data_loaders
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics directory
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = None  # Will be initialized during training
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
    def _init_optimizer(self):
        """
        Initialize the optimizer.
        
        Returns:
            AdamW optimizer
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )
    
    def train(self, epochs=3, early_stopping_patience=3):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary containing training history
        """
        print(f"Training on {self.device}")
        
        # Initialize the scheduler
        total_steps = len(self.data_loaders['train']) // self.gradient_accumulation_steps * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = None
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validation
            val_metrics = self.evaluate('validation', epoch + 1)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Save the model checkpoint
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}-{timestamp}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.model.save_pretrained(checkpoint_dir)
            
            # Check for early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save the best model
                best_model_path = os.path.join(self.output_dir, "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                self.model.save_pretrained(best_model_path)
                print(f"Saved best model checkpoint to {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save history plot
        self._plot_training_history()
        
        # Evaluate on test sets
        if best_model_path:
            self.model = self.model.from_pretrained(best_model_path)
            self.model.to(self.device)
        
        print("\nEvaluating on test_lay split:")
        test_lay_metrics = self.evaluate('test_lay', "final")
        
        print("\nEvaluating on test_expert split:")
        test_expert_metrics = self.evaluate('test_expert', "final")
        
        return {
            'history': self.history,
            'test_lay_metrics': test_lay_metrics,
            'test_expert_metrics': test_expert_metrics
        }
    
    def _train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        train_dataloader = self.data_loaders['train']
        epoch_iterator = tqdm(train_dataloader, desc="Training")
        
        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * self.gradient_accumulation_steps
            epoch_iterator.set_postfix({'loss': total_loss / (step + 1)})
        
        return total_loss / len(train_dataloader)
    
    def evaluate(self, split, epoch=""):
        """
        Evaluate the model on a specific data split.
        Public interface for _evaluate method.
        
        Args:
            split: Data split to evaluate on ('validation', 'test_lay', or 'test_expert')
            epoch: Current epoch number or identifier
            
        Returns:
            Dictionary containing evaluation metrics
        """
        return self._evaluate(split, epoch)
    
    def _evaluate(self, split, epoch=""):
        """
        Evaluate the model on a specific data split.
        
        Args:
            split: Data split to evaluate on ('validation', 'test_lay', or 'test_expert')
            epoch: Current epoch number or identifier
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        dataloader = self.data_loaders[split]
        eval_iterator = tqdm(dataloader, desc=f"Evaluating on {split}")
        
        with torch.no_grad():
            for batch in eval_iterator:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Collect loss
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Generate classification report
        target_names = ['entailment', 'neutral', 'contradiction']
        report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
        report_text = classification_report(all_labels, all_preds, target_names=target_names)
        
        # Print the report to console
        print(report_text)
        
        # Save the classification report to files
        self._save_classification_report(report, report_text, split, epoch)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(all_labels, all_preds, split, epoch)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'report': report
        }
    
    def _save_classification_report(self, report_dict, report_text, split, epoch):
        """
        Save classification report to text and CSV files.
        
        Args:
            report_dict: Classification report as dictionary
            report_text: Classification report as text
            split: Data split name
            epoch: Current epoch or identifier
        """
        epoch_str = f"epoch_{epoch}" if isinstance(epoch, int) else epoch
        
        # Save as text file
        txt_path = os.path.join(self.metrics_dir, f"{split}_{epoch_str}_report.txt")
        with open(txt_path, 'w') as f:
            f.write(report_text)
        
        # Reshape the report dictionary for CSV
        rows = []
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):  # Skip non-metric entries
                row = {'class': class_name}
                row.update(metrics)
                rows.append(row)
        
        # Save as CSV file
        csv_path = os.path.join(self.metrics_dir, f"{split}_{epoch_str}_report.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'precision', 'recall', 'f1-score', 'support'])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    'class': row['class'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1-score': row['f1-score'],
                    'support': row['support']
                })
        
        print(f"Saved classification report to {txt_path} and {csv_path}")
        
        # Also save a cumulative CSV with epoch information for validation
        if split == 'validation' and isinstance(epoch, int):
            cumulative_csv_path = os.path.join(self.metrics_dir, "validation_metrics_by_epoch.csv")
            
            # Check if file exists to determine if we need a header
            file_exists = os.path.isfile(cumulative_csv_path)
            
            with open(cumulative_csv_path, 'a', newline='') as f:
                fieldnames = ['epoch', 'class', 'precision', 'recall', 'f1-score', 'support']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for row in rows:
                    writer.writerow({
                        'epoch': epoch,
                        'class': row['class'],
                        'precision': row['precision'],
                        'recall': row['recall'],
                        'f1-score': row['f1-score'],
                        'support': row['support']
                    })
            
            print(f"Updated cumulative metrics in {cumulative_csv_path}")
    
    def _plot_training_history(self):
        """
        Plot training history.
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_accuracy'])
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        # Plot F1 score
        plt.subplot(1, 3, 3)
        plt.plot(self.history['val_f1'])
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_accuracy': self.history['val_accuracy'],
            'val_f1': self.history['val_f1']
        })
        metrics_df.to_csv(os.path.join(self.metrics_dir, 'training_metrics.csv'), index=False)
    
    def _plot_confusion_matrix(self, labels, preds, split, epoch=""):
        """
        Plot confusion matrix.
        
        Args:
            labels: True labels
            preds: Predicted labels
            split: Data split name
            epoch: Current epoch or identifier
        """
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['entailment', 'neutral', 'contradiction'],
            yticklabels=['entailment', 'neutral', 'contradiction']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Add epoch info to the title if provided
        if epoch:
            epoch_str = f"Epoch {epoch}" if isinstance(epoch, int) else epoch.capitalize()
            title = f'Confusion Matrix - {split} ({epoch_str})'
        else:
            title = f'Confusion Matrix - {split}'
            
        plt.title(title)
        plt.tight_layout()
        
        # Create file name with epoch info if provided
        if epoch:
            epoch_str = f"epoch_{epoch}" if isinstance(epoch, int) else epoch
            file_name = f'confusion_matrix_{split}_{epoch_str}.png'
        else:
            file_name = f'confusion_matrix_{split}.png'
            
        plt.savefig(os.path.join(self.metrics_dir, file_name))
        plt.close()
