"""
Model definition for the Indonesian NLI project.
"""

from transformers import RobertaForSequenceClassification
import torch.nn as nn


def get_roberta_model(model_name="cahya/roberta-base-indonesian-1.5G", num_labels=3):
    """
    Initialize a RoBERTa model for sequence classification.
    
    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of output labels (3 for NLI: entailment, neutral, contradiction)
        
    Returns:
        Initialized RobertaForSequenceClassification model
    """
    # Load model with classification head
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    return model


class RobertaNLIModel(nn.Module):
    """
    Wrapper class for RoBERTa model for NLI task.
    This allows for future customization if needed.
    """
    
    def __init__(self, model_name="cahya/roberta-base-indonesian-1.5G", num_labels=3):
        """
        Initialize the RoBERTa NLI model.
        
        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of output labels
        """
        super(RobertaNLIModel, self).__init__()
        self.model = get_roberta_model(model_name, num_labels)
        
    def forward(self, **inputs):
        """
        Forward pass through the model.
        
        Args:
            inputs: Input tensors including input_ids, attention_mask, etc.
            
        Returns:
            Model outputs
        """
        return self.model(**inputs)
    
    def save_pretrained(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        self.model.save_pretrained(path)
        
    @classmethod
    def from_pretrained(cls, path_or_name, num_labels=3):
        """
        Load a model from a saved checkpoint or Hugging Face Hub.
        
        Args:
            path_or_name: Path to saved model or name on Hugging Face Hub
            num_labels: Number of output labels
            
        Returns:
            Loaded RobertaNLIModel instance
        """
        instance = cls()
        instance.model = RobertaForSequenceClassification.from_pretrained(
            path_or_name,
            num_labels=num_labels
        )
        return instance
