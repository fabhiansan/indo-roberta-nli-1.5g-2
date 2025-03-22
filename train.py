"""
Main training script for Indonesian RoBERTa NLI model.
"""

import argparse
import torch
import time

from model import RobertaNLIModel
from data_loader import get_indonli_data
from trainer import Trainer
from utils import set_seed, setup_logging, push_to_hub, count_parameters
from config import Config


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a RoBERTa model for Indonesian NLI")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default=Config.model_name,
                        help="Name of the pre-trained model")
    
    # Data parameters
    parser.add_argument("--max_seq_length", type=int, default=Config.max_seq_length,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=Config.eval_batch_size,
                        help="Evaluation batch size")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=Config.learning_rate,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay,
                        help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=Config.adam_epsilon,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm,
                        help="Maximum gradient norm")
    parser.add_argument("--epochs", type=int, default=Config.epochs,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=Config.gradient_accumulation_steps,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--warmup_steps", type=int, default=Config.warmup_steps,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=Config.early_stopping_patience,
                        help="Patience for early stopping")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=Config.output_dir,
                        help="Directory to save outputs")
    parser.add_argument("--log_dir", type=str, default=Config.log_dir,
                        help="Directory to save logs")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=Config.seed,
                        help="Random seed")
    
    # Hugging Face Hub
    parser.add_argument("--push_to_hub", action="store_true", default=Config.push_to_hub,
                        help="Whether to push the model to the Hugging Face Hub")
    parser.add_argument("--hub_model_name", type=str, default=Config.hub_model_name,
                        help="Name of the model on the Hub")
    parser.add_argument("--hub_organization", type=str, default=Config.hub_organization,
                        help="Organization on the Hub")
    
    # Evaluation only
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for evaluation")
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting Indonesian NLI training with RoBERTa (%s)", args.model_name)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load data
    logger.info("Loading and preprocessing data...")
    data = get_indonli_data(
        model_name=args.model_name,
        batch_size=args.batch_size if not args.eval_only else args.eval_batch_size,
        max_length=args.max_seq_length
    )
    data_loaders = data['data_loaders']
    tokenizer = data['tokenizer']
    
    # Initialize model
    if args.eval_only and args.checkpoint:
        logger.info("Loading model from checkpoint: %s", args.checkpoint)
        model = RobertaNLIModel.from_pretrained(args.checkpoint)
    else:
        logger.info("Initializing model: %s", args.model_name)
        model = RobertaNLIModel(args.model_name)
    
    # Log model size
    num_params = count_parameters(model)
    logger.info("Model size: %s trainable parameters", f"{num_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_loaders=data_loaders,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir
    )
    
    # Evaluation only
    if args.eval_only:
        logger.info("Running evaluation only...")
        print("\nEvaluating on validation split:")
        trainer.evaluate('validation')
        
        print("\nEvaluating on test_lay split:")
        trainer.evaluate('test_lay')
        
        print("\nEvaluating on test_expert split:")
        trainer.evaluate('test_expert')
        
        return
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info("Training completed in %.2f seconds", training_time)
    
    # Push to Hugging Face Hub
    if args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        repo_url = push_to_hub(
            model=model,
            tokenizer=tokenizer,
            model_name=args.hub_model_name,
            organization=args.hub_organization
        )
        logger.info("Model available at: %s", repo_url)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
