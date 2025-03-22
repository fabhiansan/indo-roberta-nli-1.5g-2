"""
Utility functions for the Indonesian NLI project.
"""

import os
import random
import numpy as np
import torch
import logging
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set to {seed}")


def setup_logging(log_dir="./logs"):
    """
    Set up logging for the training process.
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def push_to_hub(model, tokenizer, model_name, organization=None):
    """
    Push the model and tokenizer to the Hugging Face Hub.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer used with the model
        model_name: Name to use on the Hub
        organization: Optional organization to push to
        
    Returns:
        Repository URL
    """
    repo_name = model_name.split("/")[-1] if "/" in model_name else model_name
    
    if organization:
        repo_name = f"{organization}/{repo_name}"
    
    # Push the model
    model.push_to_hub(repo_name)
    
    # Push the tokenizer
    tokenizer.push_to_hub(repo_name)
    
    print(f"Model and tokenizer pushed to {repo_name}")
    return f"https://huggingface.co/{repo_name}"


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(elapsed):
    """
    Format elapsed time.
    
    Args:
        elapsed: Elapsed time in seconds
        
    Returns:
        Formatted time string
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def create_label_map():
    """
    Create mapping between label IDs and label names.
    
    Returns:
        Dictionary mapping label IDs to names
    """
    return {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
