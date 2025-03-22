"""
Dataset classes and utilities for the advanced SBERT model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import logging


class AdvancedNLIDataset(Dataset):
    """
    Dataset for NLI task with advanced preprocessing.
    """
    
    def __init__(self, examples, tokenizer, max_length=128, dynamic_padding=True):
        """
        Initialize advanced NLI dataset.
        
        Args:
            examples: List of examples with premise, hypothesis, and label
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            dynamic_padding: Whether to use dynamic padding for batches
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary of inputs
        """
        example = self.examples[idx]
        
        # Get premise, hypothesis, and label with robust handling
        try:
            if isinstance(example, dict):
                premise = example.get("premise", "")
                hypothesis = example.get("hypothesis", "")
                label = example.get("label", 1)  # Default to neutral if missing
            elif hasattr(example, "premise") and hasattr(example, "hypothesis"):
                premise = example.premise
                hypothesis = example.hypothesis
                label = getattr(example, "label", 1)
            else:
                # Try to handle other formats
                premise = str(example[0]) if len(example) > 0 else ""
                hypothesis = str(example[1]) if len(example) > 1 else ""
                label = example[2] if len(example) > 2 else 1
        except:
            # Fallback to empty strings
            premise = ""
            hypothesis = ""
            label = 1
        
        # Process the label
        if isinstance(label, str):
            label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
            label = label_map.get(label, 1)  # Default to neutral if unknown
        
        # Tokenize inputs
        if self.dynamic_padding:
            # Return text for dynamic batching
            return {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label
            }
        else:
            # Tokenize here for static batching
            encoded_premise = self.tokenizer(
                premise,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            encoded_hypothesis = self.tokenizer(
                hypothesis,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension (will be added by DataLoader)
            encoded_premise = {k: v.squeeze(0) for k, v in encoded_premise.items()}
            encoded_hypothesis = {k: v.squeeze(0) for k, v in encoded_hypothesis.items()}
            
            return {
                "premise_input_ids": encoded_premise["input_ids"],
                "premise_attention_mask": encoded_premise["attention_mask"],
                "hypothesis_input_ids": encoded_hypothesis["input_ids"],
                "hypothesis_attention_mask": encoded_hypothesis["attention_mask"],
                "label": torch.tensor(label)
            }
    
    @staticmethod
    def collate_fn(batch, tokenizer, max_length=128):
        """
        Collate function for dynamic batching.
        
        Args:
            batch: Batch of examples
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Batch with tokenized inputs
        """
        premises = [item["premise"] for item in batch]
        hypotheses = [item["hypothesis"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Tokenize batch
        encoded_premises = tokenizer(
            premises,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        encoded_hypotheses = tokenizer(
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create batch
        return {
            "premise_input_ids": encoded_premises["input_ids"],
            "premise_attention_mask": encoded_premises["attention_mask"],
            "hypothesis_input_ids": encoded_hypotheses["input_ids"],
            "hypothesis_attention_mask": encoded_hypotheses["attention_mask"],
            "labels": torch.tensor(labels)
        }


def load_indonli_data(split="train"):
    """
    Load the IndoNLI dataset for the specified split.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test_lay', 'test_expert')
        
    Returns:
        List of examples
    """
    try:
        dataset = load_dataset("afaji/indonli", split=split)
        logging.info(f"Loaded {len(dataset)} examples from {split} split")
        
        # Convert dataset to list of dictionaries for consistency
        examples = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                example = {
                    "premise": item["premise"] if "premise" in item else "",
                    "hypothesis": item["hypothesis"] if "hypothesis" in item else "",
                    "label": item["label"] if "label" in item else "neutral"
                }
                examples.append(example)
            except Exception as e:
                logging.warning(f"Error processing example {i}: {str(e)}")
                # Add default example
                examples.append({
                    "premise": "",
                    "hypothesis": "",
                    "label": "neutral"
                })
        
        return examples
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return []


def create_dataloaders(model_name, batch_size=16, max_length=128, splits=None, dynamic_padding=True):
    """
    Create dataloaders for training and evaluation.
    
    Args:
        model_name: Model name for tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        splits: List of splits to load (defaults to all splits)
        dynamic_padding: Whether to use dynamic padding
        
    Returns:
        Dictionary of dataloaders
    """
    if splits is None:
        splits = ["train", "validation", "test_lay", "test_expert"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataloaders
    dataloaders = {}
    for split in splits:
        # Load data
        examples = load_indonli_data(split)
        
        # Create dataset
        dataset = AdvancedNLIDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            dynamic_padding=dynamic_padding
        )
        
        # Create dataloader
        if dynamic_padding:
            dataloaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(split == "train"),
                collate_fn=lambda batch: AdvancedNLIDataset.collate_fn(batch, tokenizer, max_length)
            )
        else:
            dataloaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(split == "train")
            )
    
    return dataloaders
