"""
Script to push all checkpoint models to Hugging Face Hub.
"""

import os
import argparse
import glob
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer
from utils import setup_logging
import re


def push_checkpoint_to_hub(checkpoint_path, hub_model_id, organization=None, base_model_name=None):
    """
    Push a checkpoint model and tokenizer to the Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        hub_model_id: Base model ID to use on the Hub
        organization: Optional organization to push to
        base_model_name: Name of the base model to load the tokenizer from
    
    Returns:
        Repository URL
    """
    # Extract checkpoint information from the path
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Create a unique model ID for this checkpoint
    if "epoch" in checkpoint_name:
        # Extract epoch number using regex
        epoch_match = re.search(r'epoch-(\d+)', checkpoint_name)
        epoch_num = epoch_match.group(1) if epoch_match else "unknown"
        unique_id = f"{hub_model_id}-epoch-{epoch_num}"
    else:
        # For the best model or other special checkpoints
        unique_id = f"{hub_model_id}-{checkpoint_name}"
    
    # If organization is provided, include it in the repo name
    if organization:
        repo_id = f"{organization}/{unique_id}"
    else:
        repo_id = unique_id
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    try:
        # Load model from checkpoint
        model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
        
        # Load tokenizer from base model since checkpoints don't have tokenizer files
        logger.info(f"Loading tokenizer from {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        logger.info(f"Pushing to Hub as {repo_id}")
        
        # Push to Hub
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        
        logger.info(f"Successfully pushed to https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"
    
    except Exception as e:
        logger.error(f"Failed to push {checkpoint_path}: {str(e)}")
        return None


def find_all_checkpoints(output_dir):
    """
    Find all checkpoint directories in the output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        List of checkpoint paths
    """
    # Look for directories that start with "checkpoint" or match "best_model"
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    # Also include best_model if it exists
    best_model_path = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_path):
        checkpoint_dirs.append(best_model_path)
    
    return checkpoint_dirs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Push all checkpoints to Hugging Face Hub")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory containing model checkpoints")
    parser.add_argument("--hub_model_id", type=str, required=True,
                        help="Base model ID for Hugging Face Hub (without organization)")
    parser.add_argument("--organization", type=str, default=None,
                        help="Optional Hugging Face organization")
    parser.add_argument("--base_model_name", type=str, default="cahya/roberta-base-indonesian-1.5G",
                        help="Name of the base model to load the tokenizer from")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Find all checkpoints
    checkpoints = find_all_checkpoints(args.output_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    if len(checkpoints) == 0:
        logger.warning(f"No checkpoints found in {args.output_dir}")
        return
    
    # Push each checkpoint to Hub
    for checkpoint in checkpoints:
        logger.info(f"Processing {checkpoint}")
        push_checkpoint_to_hub(
            checkpoint, 
            args.hub_model_id, 
            args.organization, 
            args.base_model_name
        )


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    main()
