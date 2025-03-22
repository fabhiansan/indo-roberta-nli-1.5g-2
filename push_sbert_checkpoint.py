"""
Script to push the fine-tuned SBERT model to Hugging Face Hub.
"""

import os
import argparse
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
import shutil

def push_sbert_to_hub(
    checkpoint_path,
    hub_model_id,
    organization=None,
    commit_message=None
):
    """
    Push a fine-tuned SBERT model to the Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        hub_model_id: Model ID on the HF Hub
        organization: Optional organization name
        commit_message: Commit message for the push
    """
    try:
        # Load the model
        logging.info(f"Loading model from {checkpoint_path}")
        model = SentenceTransformer(checkpoint_path)
        
        # Set repository ID
        if organization:
            repo_id = f"{organization}/{hub_model_id}"
        else:
            repo_id = hub_model_id
        
        # Set commit message
        if commit_message is None:
            commit_message = f"Upload model {Path(checkpoint_path).name}"
        
        # Push to hub
        logging.info(f"Pushing model to hub: {repo_id}")
        model.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            token=None  # Will use the token from the Hugging Face CLI
        )
        
        logging.info(f"Successfully pushed model to {repo_id}")
        return True
    
    except Exception as e:
        logging.error(f"Error pushing model to hub: {e}")
        return False


def push_all_checkpoints(
    checkpoints_dir,
    hub_model_id,
    organization=None,
    model_card_path=None
):
    """
    Push all checkpoints in a directory to the Hugging Face Hub.
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
        hub_model_id: Base model ID on the HF Hub
        organization: Optional organization name
        model_card_path: Path to the model card markdown file
    """
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # Get list of checkpoint directories
    checkpoints = [
        d for d in os.listdir(checkpoints_dir) 
        if os.path.isdir(os.path.join(checkpoints_dir, d)) and "checkpoint" in d
    ]
    
    if not checkpoints:
        logging.error(f"No checkpoints found in {checkpoints_dir}")
        return
    
    logging.info(f"Found {len(checkpoints)} checkpoints to push")
    
    # Sort checkpoints by step number
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
    )
    
    # Process the main model first (latest checkpoint)
    latest_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])
    
    # Push the main model
    logging.info(f"Pushing the main model (latest checkpoint): {latest_checkpoint}")
    success = push_sbert_to_hub(
        checkpoint_path=latest_checkpoint,
        hub_model_id=hub_model_id,
        organization=organization,
        commit_message=f"Upload main model from {Path(latest_checkpoint).name}"
    )
    
    if not success:
        logging.error("Failed to push the main model. Aborting.")
        return
    
    # Copy model card if provided
    if model_card_path and os.path.exists(model_card_path):
        try:
            from huggingface_hub import HfApi, Repository
            
            # Set repository ID
            if organization:
                repo_id = f"{organization}/{hub_model_id}"
            else:
                repo_id = hub_model_id
            
            # Initialize Hugging Face API
            api = HfApi()
            
            # Clone the repository
            repo_dir = f"tmp_repo_{hub_model_id.replace('/', '_')}"
            repo = Repository(
                local_dir=repo_dir,
                clone_from=repo_id,
                use_auth_token=True
            )
            
            # Copy the model card
            shutil.copy(model_card_path, os.path.join(repo_dir, "README.md"))
            
            # Push the changes
            repo.git_add("README.md")
            repo.git_commit("Add model card")
            repo.git_push()
            
            logging.info(f"Model card pushed to {repo_id}")
            
            # Clean up
            shutil.rmtree(repo_dir, ignore_errors=True)
            
        except Exception as e:
            logging.error(f"Error pushing model card: {e}")
    
    # Push other checkpoints as references
    for checkpoint in checkpoints[:-1]:  # Skip the latest one which was already pushed
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
        
        # Extract step number for the reference model ID
        step = checkpoint.split("-")[-1] if "-" in checkpoint else "unknown"
        reference_model_id = f"{hub_model_id}-step-{step}"
        
        logging.info(f"Pushing checkpoint {checkpoint} as {reference_model_id}")
        
        push_sbert_to_hub(
            checkpoint_path=checkpoint_path,
            hub_model_id=reference_model_id,
            organization=organization,
            commit_message=f"Upload checkpoint from {checkpoint}"
        )
    
    logging.info("Finished pushing all checkpoints")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Push SBERT checkpoints to Hugging Face Hub")
    
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=True,
        help="Base model ID on the Hugging Face Hub"
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Optional organization name"
    )
    parser.add_argument(
        "--model_card_path",
        type=str,
        default=None,
        help="Path to the model card markdown file"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    push_all_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        hub_model_id=args.hub_model_id,
        organization=args.organization,
        model_card_path=args.model_card_path
    )


if __name__ == "__main__":
    main()
