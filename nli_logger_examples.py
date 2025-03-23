"""
Example usage of the NLILogger for various model training scenarios.
"""

import torch
from nli_logger import NLILogger, create_logger
from transformers import AutoTokenizer
from datasets import load_dataset

def example_training_loop():
    """
    Example of using NLILogger in a training loop for any model type.
    """
    # Initialize logger with model name and output directory
    logger = create_logger(
        model_name="example-sbert-model", 
        output_dir="./logs"
    )
    
    # Log hyperparameters
    hparams = {
        "model_name": "firqaaa/indo-sentence-bert-base",
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "num_epochs": 5,
        "max_grad_norm": 1.0,
        "pooling_mode": "mean_pooling",
        "use_cross_attention": True
    }
    logger.log_hyperparameters(hparams)
    
    # Mock training data
    tokenizer = AutoTokenizer.from_pretrained("firqaaa/indo-sentence-bert-base")
    try:
        dataset = load_dataset("indonli")
        
        # Log dataset statistics
        logger.log_dataset_statistics({
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test_lay": dataset["test_lay"],
            "test_expert": dataset["test_expert"]
        })
        
    except Exception as e:
        logger.logger.error(f"Error loading dataset: {e}")
        return
    
    # Mock training loop
    for epoch in range(5):
        # Log training steps
        for step in range(10):
            loss = 1.0 - (epoch * 0.1 + step * 0.01)
            lr = 2e-5 * (1.0 - epoch / 5)
            logger.log_train_step(
                epoch=epoch,
                step=step,
                loss=loss,
                lr=lr,
                additional_metrics={"grad_norm": 0.8}
            )
        
        # Log epoch results
        val_metrics = {
            "accuracy": 0.7 + epoch * 0.05,
            "macro_f1": 0.65 + epoch * 0.06,
            "precision": 0.68 + epoch * 0.05,
            "recall": 0.72 + epoch * 0.04
        }
        
        checkpoint_path = f"./outputs/example-sbert-model/checkpoint-{epoch}"
        logger.log_epoch(
            epoch=epoch,
            train_loss=1.0 - epoch * 0.15,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path
        )
    
    # Mock evaluation on test sets
    # Generate mock predictions and labels
    num_test_examples = 100
    true_labels = torch.randint(0, 3, (num_test_examples,))
    predictions = torch.randint(0, 3, (num_test_examples,))
    
    # Log evaluation results
    logger.log_evaluation("test_lay", predictions, true_labels, checkpoint_name="final")
    logger.log_evaluation("test_expert", predictions, true_labels, checkpoint_name="final")
    
    # Plot training curves
    logger.plot_training_curve()
    
    # Compare checkpoints
    logger.compare_checkpoints()
    
    # Generate summary report
    summary = logger.generate_summary_report()
    print(f"\nSummary Report:\n{summary}")


def integration_with_sbert_train():
    """
    Example showing how to integrate NLILogger with the SBERT training script.
    """
    # Simplified mock training function with NLILogger integration
    def train_with_logger(args):
        # Create logger
        logger = create_logger(
            model_name=args.model_name, 
            output_dir=args.output_dir
        )
        
        # Log hyperparameters
        logger.log_hyperparameters(vars(args))
        
        # Mock dataset loading
        logger.logger.info("Loading dataset...")
        
        # Training loop
        best_val_accuracy = 0
        best_checkpoint = None
        
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            
            # Training steps
            for step in range(10):  # Mock 10 steps per epoch
                # Forward pass, loss calculation, etc.
                loss = 1.0 - (epoch * 0.1 + step * 0.01)  # Mock decreasing loss
                
                # Log step
                logger.log_train_step(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    lr=args.learning_rate * (1 - epoch/args.num_epochs)
                )
                
                epoch_loss += loss / 10  # Average loss for the epoch
            
            # Validation
            val_metrics = {
                "accuracy": 0.7 + epoch * 0.05,
                "macro_f1": 0.65 + epoch * 0.06,
                "precision": 0.68 + epoch * 0.05,
                "recall": 0.72 + epoch * 0.04
            }
            
            # Save checkpoint
            checkpoint_path = f"{args.output_dir}/checkpoint-{epoch}"
            logger.log_epoch(
                epoch=epoch,
                train_loss=epoch_loss,
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_path
            )
            
            # Track best model
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_checkpoint = checkpoint_path
        
        # Final evaluation on test sets
        logger.logger.info(f"Evaluating best checkpoint: {best_checkpoint}")
        
        # Mock predictions
        num_test_examples = 100
        true_labels = torch.randint(0, 3, (num_test_examples,))
        predictions = torch.randint(0, 3, (num_test_examples,))
        
        # Log test results
        logger.log_evaluation("test_lay", predictions, true_labels, checkpoint_name="best")
        logger.log_evaluation("test_expert", predictions, true_labels, checkpoint_name="best")
        
        # Generate visualizations and report
        logger.plot_training_curve()
        logger.compare_checkpoints()
        logger.generate_summary_report()
        
        return logger, best_checkpoint

    # Mock arguments
    class Args:
        def __init__(self):
            self.model_name = "indo-sbert-classifier"
            self.output_dir = "./logs"
            self.num_epochs = 5
            self.learning_rate = 2e-5
    
    # Run the training
    train_with_logger(Args())


def integration_with_improved_sbert():
    """
    Example showing how to integrate NLILogger with the improved SBERT model.
    """
    # Add this to the top of train_improved_sbert.py:
    # from nli_logger import create_logger
    
    # Mock the training function with logger
    def train_improved_sbert(args):
        # Create logger
        logger = create_logger(
            model_name="improved-sbert",
            output_dir=args.output_dir
        )
        
        # Log hyperparameters and dataset stats
        logger.log_hyperparameters(vars(args))
        
        # Load dataset and model (mocked)
        logger.logger.info("Loading dataset and model...")
        
        # Training loop with integrated logging
        for epoch in range(args.num_epochs):
            # Training steps
            total_loss = 0
            for step in range(50):  # Mock 50 batches per epoch
                # Mock training step
                batch_loss = 1.0 / (1.0 + epoch + step/100)
                total_loss += batch_loss
                
                # Log every 10 steps
                if step % 10 == 0:
                    logger.log_train_step(
                        epoch=epoch,
                        step=step,
                        loss=batch_loss,
                        lr=args.learning_rate,
                        additional_metrics={
                            "batch_size": args.batch_size,
                            "grad_norm": 0.5 - epoch * 0.05
                        }
                    )
            
            # Validation after each epoch
            val_predictions = torch.randint(0, 3, (200,))  # Mock predictions
            val_labels = torch.randint(0, 3, (200,))       # Mock labels
            
            # Calculate metrics manually or use logger's evaluation
            val_metrics = logger.log_evaluation(
                "validation", 
                val_predictions, 
                val_labels,
                checkpoint_name=f"epoch-{epoch}"
            )
            
            # Log epoch with metrics
            avg_loss = total_loss / 50
            checkpoint_path = f"{args.output_dir}/improved-sbert-{epoch}"
            
            logger.log_epoch(
                epoch=epoch,
                train_loss=avg_loss,
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_path
            )
        
        # Final evaluation on test sets
        test_lay_preds = torch.randint(0, 3, (300,))
        test_lay_labels = torch.randint(0, 3, (300,))
        
        test_expert_preds = torch.randint(0, 3, (300,))
        test_expert_labels = torch.randint(0, 3, (300,))
        
        logger.log_evaluation("test_lay", test_lay_preds, test_lay_labels, checkpoint_name="final")
        logger.log_evaluation("test_expert", test_expert_preds, test_expert_labels, checkpoint_name="final")
        
        # Generate report and visualizations
        logger.plot_training_curve()
        logger.compare_checkpoints("test_lay")
        summary = logger.generate_summary_report()
        
        print(f"Training completed. Summary:\n{summary}")
    
    # Mock args
    class ImprovedSBERTArgs:
        def __init__(self):
            self.num_epochs = 5
            self.batch_size = 16
            self.learning_rate = 2e-5
            self.output_dir = "./outputs/improved-sbert"
            self.use_cross_attention = True
            self.pooling_mode = "mean_pooling"
            self.max_grad_norm = 1.0
    
    # Run the function
    train_improved_sbert(ImprovedSBERTArgs())


if __name__ == "__main__":
    print("Running NLILogger examples...")
    
    example_training_loop()
    print("\n" + "="*50 + "\n")
    
    integration_with_sbert_train()
    print("\n" + "="*50 + "\n")
    
    integration_with_improved_sbert()
