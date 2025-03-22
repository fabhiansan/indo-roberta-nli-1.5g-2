"""
Configuration settings for the Indonesian NLI project.
"""

class Config:
    """Configuration class for training and evaluation parameters."""
    
    # Model parameters
    model_name = "cahya/roberta-base-indonesian-1.5G"
    num_labels = 3  # entailment, neutral, contradiction
    
    # Data parameters
    max_seq_length = 128
    batch_size = 16
    
    # Training parameters
    learning_rate = 2e-5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    
    # Learning rate scheduler
    warmup_steps = 0
    
    # Training loop
    epochs = 5
    gradient_accumulation_steps = 1
    early_stopping_patience = 3
    
    # Output directories
    output_dir = "./outputs"
    log_dir = "./logs"
    
    # Reproducibility
    seed = 42
    
    # Hugging Face Hub
    push_to_hub = False
    hub_model_name = "roberta-indonesian-nli"
    hub_organization = None  # Set to your organization if you have one
    
    # Evaluation
    eval_batch_size = 32
