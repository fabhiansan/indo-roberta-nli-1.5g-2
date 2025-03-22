"""
Improved SBERT model architecture with simplified but effective components.
This improved version balances complexity and performance with a focus on what works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class SimplifiedPooling(nn.Module):
    """
    Simplified pooling mechanisms for sentence embeddings.
    """
    
    def __init__(self, hidden_size, pooling_mode="mean_pooling"):
        """
        Initialize simplified pooling.
        
        Args:
            hidden_size: Size of hidden state from BERT
            pooling_mode: Pooling strategy ('mean_pooling', 'max_pooling', 'cls')
        """
        super(SimplifiedPooling, self).__init__()
        self.hidden_size = hidden_size
        self.pooling_mode = pooling_mode
    
    def forward(self, token_embeddings, attention_mask):
        """
        Forward pass.
        
        Args:
            token_embeddings: Token embeddings from BERT [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Sentence embedding [batch_size, hidden_size]
        """
        if self.pooling_mode == "cls":
            # Use CLS token
            return token_embeddings[:, 0]
        
        elif self.pooling_mode == "max_pooling":
            # Max pooling over time
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings = token_embeddings.clone()
            
            # Set padding tokens to large negative value so they don't affect max
            token_embeddings[input_mask_expanded == 0] = -1e9
            
            # Take max over time dimension
            return torch.max(token_embeddings, 1)[0]
        
        else:  # Default: mean_pooling
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask


class LightweightCrossAttention(nn.Module):
    """
    Lightweight cross-attention between two sentence embeddings.
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        """
        Initialize lightweight cross-attention.
        
        Args:
            hidden_size: Size of embeddings
            dropout: Dropout probability
        """
        super(LightweightCrossAttention, self).__init__()
        
        # Simplified projection layers with fewer parameters
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection with dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, premise_embedding, hypothesis_embedding):
        """
        Forward pass.
        
        Args:
            premise_embedding: Premise embedding [batch_size, hidden_size]
            hypothesis_embedding: Hypothesis embedding [batch_size, hidden_size]
            
        Returns:
            Cross-attended embeddings
        """
        # Add batch dimension if not present
        if len(premise_embedding.shape) == 2:
            premise_embedding = premise_embedding.unsqueeze(1)  # [batch, 1, hidden]
        if len(hypothesis_embedding.shape) == 2:
            hypothesis_embedding = hypothesis_embedding.unsqueeze(1)  # [batch, 1, hidden]
        
        # Compute query, key, value
        query = self.query(premise_embedding)
        key = self.key(hypothesis_embedding)
        value = self.value(hypothesis_embedding)
        
        # Calculate attention scores
        attention_scores = torch.bmm(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (premise_embedding.size(-1) ** 0.5)  # Scale
        
        # Apply softmax normalization
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value
        context = torch.bmm(attention_probs, value)
        
        # Apply residual connection and normalization
        context = self.layer_norm(context + premise_embedding)
        
        # Remove sequence dimension if it was added
        if context.size(1) == 1:
            context = context.squeeze(1)
        
        return context


class EfficientClassifierHead(nn.Module):
    """
    Efficient classifier head for NLI task with optimal size.
    """
    
    def __init__(self, input_size, hidden_size=256, dropout=0.1):
        """
        Initialize efficient classifier.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            dropout: Dropout probability
        """
        super(EfficientClassifierHead, self).__init__()
        
        # Just one hidden layer with proper regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 3)  # 3 NLI classes
    
    def forward(self, features):
        """
        Forward pass.
        
        Args:
            features: Input features
            
        Returns:
            Logits for NLI classes
        """
        x = self.dropout1(features)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout2(x)
        return self.classifier(x)


class ImprovedSBERTModel(nn.Module):
    """
    Improved SBERT model for sentence embeddings with balanced architecture.
    """
    
    def __init__(self, 
                 model_name="firqaaa/indo-sentence-bert-base",
                 pooling_mode="mean_pooling",
                 classifier_hidden_size=256,
                 dropout=0.1,
                 use_cross_attention=True):
        """
        Initialize improved SBERT model.
        
        Args:
            model_name: Pretrained SBERT model name or path
            pooling_mode: Pooling strategy ('mean_pooling', 'max_pooling', 'cls')
            classifier_hidden_size: Size of classifier hidden layer
            dropout: Dropout probability
            use_cross_attention: Whether to use cross-attention between sentences
        """
        super(ImprovedSBERTModel, self).__init__()
        
        # Load model and config
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        # Simple but effective pooling
        self.pooling = SimplifiedPooling(
            hidden_size=hidden_size, 
            pooling_mode=pooling_mode
        )
        
        # Initialize cross-attention if enabled
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = LightweightCrossAttention(
                hidden_size=hidden_size,
                dropout=dropout
            )
        
        # Classifier input size depends on the feature combination approach
        if use_cross_attention:
            # With cross-attention, use more targeted feature combination
            classifier_input_size = hidden_size * 3  # [p, h, cross(p,h)]
        else:
            # Without cross-attention, use concatenation of embeddings and difference
            classifier_input_size = hidden_size * 3  # [p, h, |p-h|]
        
        # Efficient classifier head
        self.classifier = EfficientClassifierHead(
            input_size=classifier_input_size,
            hidden_size=classifier_hidden_size,
            dropout=dropout
        )
    
    def encode(self, input_ids, attention_mask):
        """
        Encode sentences to embeddings.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            
        Returns:
            Sentence embedding
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply pooling to get sentence embedding
        sentence_embedding = self.pooling(
            token_embeddings=outputs.last_hidden_state,
            attention_mask=attention_mask
        )
        
        return sentence_embedding
    
    def forward(self, premise_input_ids, premise_attention_mask, 
                hypothesis_input_ids, hypothesis_attention_mask):
        """
        Forward pass.
        
        Args:
            premise_input_ids: Input ids for premise
            premise_attention_mask: Attention mask for premise
            hypothesis_input_ids: Input ids for hypothesis
            hypothesis_attention_mask: Attention mask for hypothesis
            
        Returns:
            Logits for NLI classes
        """
        # Encode premise and hypothesis
        premise_embedding = self.encode(
            input_ids=premise_input_ids,
            attention_mask=premise_attention_mask
        )
        
        hypothesis_embedding = self.encode(
            input_ids=hypothesis_input_ids,
            attention_mask=hypothesis_attention_mask
        )
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            cross_embedding = self.cross_attention(
                premise_embedding=premise_embedding,
                hypothesis_embedding=hypothesis_embedding
            )
            
            # Combine features: [premise, hypothesis, cross]
            combined_features = torch.cat([
                premise_embedding,
                hypothesis_embedding,
                cross_embedding
            ], dim=-1)
        else:
            # Combine features: [premise, hypothesis, |premise-hypothesis|]
            combined_features = torch.cat([
                premise_embedding,
                hypothesis_embedding,
                torch.abs(premise_embedding - hypothesis_embedding)
            ], dim=-1)
        
        # Apply classifier
        logits = self.classifier(combined_features)
        
        return logits
    
    def save_pretrained(self, output_dir):
        """
        Save model and components to output directory.
        
        Args:
            output_dir: Directory to save model
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save BERT model
        self.bert.save_pretrained(output_dir)
        
        # Save model config
        config = {
            "pooling_mode": self.pooling.pooling_mode,
            "use_cross_attention": self.use_cross_attention,
            "classifier_hidden_size": self.classifier.dense.out_features,
            "dropout": self.classifier.dropout1.p
        }
        
        # Save config as JSON
        import json
        with open(os.path.join(output_dir, "improved_sbert_config.json"), "w") as f:
            json.dump(config, f)
        
        # Save full model state dict
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load model from path.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Loaded model
        """
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_path, "improved_sbert_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config if not found
            config = {
                "pooling_mode": "mean_pooling",
                "use_cross_attention": True,
                "classifier_hidden_size": 256,
                "dropout": 0.1
            }
        
        # Create model
        model = cls(
            model_name=model_path,
            pooling_mode=config["pooling_mode"],
            classifier_hidden_size=config["classifier_hidden_size"],
            dropout=config["dropout"],
            use_cross_attention=config["use_cross_attention"]
        )
        
        # Load state dict if available
        model_path_bin = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_path_bin):
            model.load_state_dict(torch.load(model_path_bin, map_location=torch.device("cpu")))
        
        return model
