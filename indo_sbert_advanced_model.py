"""
Advanced SBERT model architecture with sophisticated classifier head.
This file contains model classes and utility functions for embedding generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class AdvancedPooling(nn.Module):
    """
    Advanced pooling mechanisms for sentence embeddings.
    """
    
    def __init__(self, 
                 hidden_size, 
                 pooling_mode="mean_pooling",
                 attention_hidden_size=128):
        """
        Initialize advanced pooling.
        
        Args:
            hidden_size: Size of hidden state from BERT
            pooling_mode: Pooling strategy ('mean_pooling', 'max_pooling', 'cls', 'attention')
            attention_hidden_size: Hidden size for attention mechanism
        """
        super(AdvancedPooling, self).__init__()
        self.hidden_size = hidden_size
        self.pooling_mode = pooling_mode
        
        # If using attention pooling, create attention layer
        if pooling_mode == "attention":
            self.attention_query = nn.Linear(hidden_size, attention_hidden_size)
            self.attention_key = nn.Linear(hidden_size, attention_hidden_size)
            self.attention_value = nn.Linear(hidden_size, hidden_size)
    
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
        
        elif self.pooling_mode == "attention":
            # Attention-based pooling
            batch_size, seq_len, hidden_size = token_embeddings.size()
            
            # Create attention scores
            query = self.attention_query(token_embeddings)  # [batch_size, seq_len, attn_hidden]
            key = self.attention_key(token_embeddings)      # [batch_size, seq_len, attn_hidden]
            
            # Compute attention weights
            attn_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
            attn_scores = attn_scores / (self.hidden_size ** 0.5)  # Scale
            
            # Apply mask to ignore padding tokens
            mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
            # Normalize attention weights
            attn_weights = F.softmax(attn_scores, dim=2)  # [batch_size, seq_len, seq_len]
            
            # Apply attention weights
            value = self.attention_value(token_embeddings)  # [batch_size, seq_len, hidden]
            weighted = torch.bmm(attn_weights, value)       # [batch_size, seq_len, hidden]
            
            # Use weighted sum of CLS token attention
            return weighted[:, 0]
        
        else:  # Default: mean_pooling
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask


class CrossAttention(nn.Module):
    """
    Cross-attention between two sentence embeddings.
    """
    
    def __init__(self, hidden_size, num_attention_heads=8, dropout=0.1):
        """
        Initialize cross-attention.
        
        Args:
            hidden_size: Size of embeddings
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def transpose_for_scores(self, x):
        """Split heads and transpose."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
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
        
        # Prepare query, key, value
        mixed_query_layer = self.query(premise_embedding)
        mixed_key_layer = self.key(hypothesis_embedding)
        mixed_value_layer = self.value(hypothesis_embedding)
        
        # Split heads
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Apply softmax normalization
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape to original dimensions
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        output = self.dropout(output)
        output = self.layer_norm(output + premise_embedding)
        
        # Remove sequence dimension if it was added
        if output.size(1) == 1:
            output = output.squeeze(1)
        
        return output


class EnhancedClassifierHead(nn.Module):
    """
    Enhanced classifier head for NLI task with multiple layers.
    """
    
    def __init__(self, input_size, hidden_sizes=[512, 256], dropout=0.2):
        """
        Initialize enhanced classifier.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(EnhancedClassifierHead, self).__init__()
        
        # Create list of layers
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 3))  # 3 classes: entailment, neutral, contradiction
        
        # Create sequential model
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Logits for NLI classes
        """
        return self.classifier(x)


class SBERTAdvancedModel(nn.Module):
    """
    Advanced BERT model for sentence embeddings with sophisticated classifier.
    """
    
    def __init__(self, 
                 model_name,
                 pooling_mode="mean_pooling",
                 classifier_hidden_sizes=[512, 256],
                 dropout=0.2,
                 use_cross_attention=True):
        """
        Initialize advanced SBERT model.
        
        Args:
            model_name: Pretrained model name or path
            pooling_mode: Pooling strategy ('mean_pooling', 'max_pooling', 'cls', 'attention')
            classifier_hidden_sizes: List of hidden layer sizes for classifier
            dropout: Dropout probability
            use_cross_attention: Whether to use cross-attention between sentences
        """
        super(SBERTAdvancedModel, self).__init__()
        
        # Load model and config
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        # Initialize pooling
        self.pooling = AdvancedPooling(
            hidden_size=hidden_size, 
            pooling_mode=pooling_mode
        )
        
        # Initialize cross-attention if enabled
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = CrossAttention(hidden_size)
        
        # Classifier input size depends on the feature combination approach
        if use_cross_attention:
            # With cross-attention, we use the cross-attended features plus the original
            classifier_input_size = hidden_size * 3  # [p, h, cross]
        else:
            # Without cross-attention, we use concatenation of embeddings and difference
            classifier_input_size = hidden_size * 3  # [p, h, |p-h|]
        
        # Initialize classifier
        self.classifier = EnhancedClassifierHead(
            input_size=classifier_input_size,
            hidden_sizes=classifier_hidden_sizes,
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply pooling
        embeddings = self.pooling(outputs.last_hidden_state, attention_mask)
        
        return embeddings
    
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
        # Get embeddings
        premise_embedding = self.encode(premise_input_ids, premise_attention_mask)
        hypothesis_embedding = self.encode(hypothesis_input_ids, hypothesis_attention_mask)
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            cross_p_h = self.cross_attention(premise_embedding, hypothesis_embedding)
            
            # Concatenate embeddings and cross-attention output
            combined_features = torch.cat([
                premise_embedding, 
                hypothesis_embedding, 
                cross_p_h
            ], dim=1)
        else:
            # Get absolute difference
            abs_diff = torch.abs(premise_embedding - hypothesis_embedding)
            
            # Concatenate embeddings and abs diff
            combined_features = torch.cat([
                premise_embedding, 
                hypothesis_embedding, 
                abs_diff
            ], dim=1)
        
        # Get logits from classifier
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
        
        # Save model configuration
        model_config = {
            "pooling_mode": self.pooling.pooling_mode,
            "use_cross_attention": self.use_cross_attention,
            "classifier_hidden_sizes": [m.out_features for m in self.classifier.classifier 
                                       if isinstance(m, nn.Linear)][:-1]  # Exclude last layer
        }
        
        # Save configuration as JSON
        import json
        with open(os.path.join(output_dir, "sbert_advanced_config.json"), "w") as f:
            json.dump(model_config, f)
        
        # Save classifier weights
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
        
        # Save pooling weights if using attention pooling
        if self.pooling.pooling_mode == "attention":
            torch.save(self.pooling.state_dict(), os.path.join(output_dir, "pooling.pt"))
        
        # Save cross-attention weights if used
        if self.use_cross_attention:
            torch.save(self.cross_attention.state_dict(), os.path.join(output_dir, "cross_attention.pt"))
    
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
        
        # Load configuration
        config_path = os.path.join(model_path, "sbert_advanced_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
        else:
            # Default configuration if not found
            model_config = {
                "pooling_mode": "mean_pooling",
                "use_cross_attention": True,
                "classifier_hidden_sizes": [512, 256]
            }
        
        # Create model
        model = cls(
            model_name=model_path,
            pooling_mode=model_config.get("pooling_mode", "mean_pooling"),
            classifier_hidden_sizes=model_config.get("classifier_hidden_sizes", [512, 256]),
            use_cross_attention=model_config.get("use_cross_attention", True)
        )
        
        # Load classifier weights
        classifier_path = os.path.join(model_path, "classifier.pt")
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        
        # Load pooling weights if using attention pooling
        if model.pooling.pooling_mode == "attention":
            pooling_path = os.path.join(model_path, "pooling.pt")
            if os.path.exists(pooling_path):
                model.pooling.load_state_dict(torch.load(pooling_path, map_location="cpu"))
        
        # Load cross-attention weights if used
        if model.use_cross_attention:
            cross_attention_path = os.path.join(model_path, "cross_attention.pt")
            if os.path.exists(cross_attention_path):
                model.cross_attention.load_state_dict(torch.load(cross_attention_path, map_location="cpu"))
        
        return model
