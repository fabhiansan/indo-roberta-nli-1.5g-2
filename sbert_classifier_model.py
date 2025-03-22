"""
SBERT model with explicit classifier for NLI tasks.
"""

import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
import logging
import os


class SBERTWithClassifier(nn.Module):
    """
    Model combining SBERT with a classification head for NLI tasks.
    
    This model takes a pre-trained SBERT model and adds a classification layer
    on top for explicit classification of NLI labels (entailment, neutral, contradiction).
    """
    
    def __init__(
        self,
        sbert_model_name: str,
        num_classes: int = 3,
        freeze_sbert: bool = False,
        dropout_prob: float = 0.1,
        hidden_size: int = 512,
        combination_mode: str = "concat",
        device: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            sbert_model_name: Name or path of the pre-trained SBERT model
            num_classes: Number of output classes (default: 3 for NLI)
            freeze_sbert: Whether to freeze the SBERT parameters
            dropout_prob: Dropout probability for the classifier
            hidden_size: Size of the hidden layer in the classifier
            combination_mode: How to combine premise and hypothesis embeddings
                              ("concat", "diff", "mult", or "concat_diff_mult")
            device: Device to use for computation ("cpu", "cuda", or None for auto-detection)
        """
        super(SBERTWithClassifier, self).__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert = SentenceTransformer(sbert_model_name, device=self.device)
        self.embedding_dim = self.sbert.get_sentence_embedding_dimension()
        self.combination_mode = combination_mode
        self.num_classes = num_classes
        self.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        
        # Determine input size based on combination mode
        if combination_mode == "concat":
            input_size = self.embedding_dim * 2
        elif combination_mode in ["diff", "mult"]:
            input_size = self.embedding_dim
        elif combination_mode == "concat_diff_mult":
            input_size = self.embedding_dim * 4
        else:
            raise ValueError(f"Unsupported combination mode: {combination_mode}")
        
        # Freeze SBERT parameters if specified
        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False
        
        # Create classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Move classifier to the same device as SBERT
        self.classifier = self.classifier.to(self.device)
    
    def forward(
        self,
        premises: List[str],
        hypotheses: List[str]
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            premises: List of premise sentences
            hypotheses: List of hypothesis sentences
            
        Returns:
            Logits for each class
        """
        # Get embeddings
        premise_embeddings = self.sbert.encode(
            premises,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        hypothesis_embeddings = self.sbert.encode(
            hypotheses,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Combine embeddings based on combination mode
        if self.combination_mode == "concat":
            combined = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
        elif self.combination_mode == "diff":
            combined = premise_embeddings - hypothesis_embeddings
        elif self.combination_mode == "mult":
            combined = premise_embeddings * hypothesis_embeddings
        elif self.combination_mode == "concat_diff_mult":
            diff = premise_embeddings - hypothesis_embeddings
            mult = premise_embeddings * hypothesis_embeddings
            combined = torch.cat([premise_embeddings, hypothesis_embeddings, diff, mult], dim=1)
        
        # Get classification logits
        logits = self.classifier(combined)
        return logits
    
    def save(self, output_path: str):
        """
        Save the model to the specified path.
        
        Args:
            output_path: Directory to save the model
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save SBERT model
        self.sbert.save(os.path.join(output_path, "sbert"))
        
        # Save classifier
        torch.save(self.classifier.state_dict(), os.path.join(output_path, "classifier.pt"))
        
        # Save configuration
        config = {
            "combination_mode": self.combination_mode,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "id2label": self.id2label,
            "label2id": self.label2id
        }
        torch.save(config, os.path.join(output_path, "config.pt"))
        
        logging.info(f"Model saved to {output_path}")
    
    @classmethod
    def load(cls, input_path: str, device: Optional[str] = None):
        """
        Load the model from the specified path.
        
        Args:
            input_path: Directory containing the saved model
            device: Device to use for computation
            
        Returns:
            Loaded model
        """
        # Load SBERT model
        sbert_path = os.path.join(input_path, "sbert")
        sbert = SentenceTransformer(sbert_path, device=device)
        
        # Load configuration
        config_path = os.path.join(input_path, "config.pt")
        config = torch.load(config_path)
        
        # Create model
        model = cls(
            sbert_model_name=sbert_path,
            num_classes=config["num_classes"],
            combination_mode=config["combination_mode"],
            device=device
        )
        
        # Set loaded SBERT
        model.sbert = sbert
        
        # Load classifier weights
        classifier_path = os.path.join(input_path, "classifier.pt")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location=model.device))
        
        # Set label mappings
        model.id2label = config["id2label"]
        model.label2id = config["label2id"]
        
        return model
    
    def predict(
        self,
        premises: List[str],
        hypotheses: List[str],
        return_probabilities: bool = False
    ) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
        """
        Predict NLI labels for the given premise-hypothesis pairs.
        
        Args:
            premises: List of premise sentences
            hypotheses: List of hypothesis sentences
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of predicted labels or tuple of (labels, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self(premises, hypotheses)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Convert prediction indices to labels
            predicted_labels = [self.id2label[pred.item()] for pred in predictions]
            
            if return_probabilities:
                return predicted_labels, probabilities.cpu().numpy().tolist()
            return predicted_labels
