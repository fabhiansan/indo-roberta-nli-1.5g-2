"""
Data loading and preprocessing utilities for the Indonesian NLI project.
"""

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer


class IndoNLIDataset(Dataset):
    """Dataset wrapper for the Indonesian NLI dataset."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_indonli_dataset():
    """Load the Indonesian NLI dataset from Hugging Face."""
    return load_dataset("afaji/indonli")


def preprocess_data(dataset, tokenizer, max_length=128):
    """
    Preprocess the dataset by tokenizing premises and hypotheses.
    
    Args:
        dataset: The dataset to preprocess
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Preprocessed dataset with tokenized inputs
    """
    train_encodings = tokenizer(
        dataset['train']['premise'],
        dataset['train']['hypothesis'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    valid_encodings = tokenizer(
        dataset['validation']['premise'],
        dataset['validation']['hypothesis'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    test_lay_encodings = tokenizer(
        dataset['test_lay']['premise'],
        dataset['test_lay']['hypothesis'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    test_expert_encodings = tokenizer(
        dataset['test_expert']['premise'],
        dataset['test_expert']['hypothesis'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    train_dataset = IndoNLIDataset(
        {k: v.numpy() for k, v in train_encodings.items()},
        dataset['train']['label']
    )
    
    valid_dataset = IndoNLIDataset(
        {k: v.numpy() for k, v in valid_encodings.items()},
        dataset['validation']['label']
    )
    
    test_lay_dataset = IndoNLIDataset(
        {k: v.numpy() for k, v in test_lay_encodings.items()},
        dataset['test_lay']['label']
    )
    
    test_expert_dataset = IndoNLIDataset(
        {k: v.numpy() for k, v in test_expert_encodings.items()},
        dataset['test_expert']['label']
    )
    
    return {
        'train': train_dataset,
        'validation': valid_dataset,
        'test_lay': test_lay_dataset,
        'test_expert': test_expert_dataset
    }


def create_data_loaders(datasets, batch_size=16):
    """
    Create PyTorch DataLoader objects for each dataset split.
    
    Args:
        datasets: Dictionary containing dataset splits
        batch_size: Batch size for training
        
    Returns:
        Dictionary containing DataLoader objects for each split
    """
    return {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True
        ),
        'validation': DataLoader(
            datasets['validation'],
            batch_size=batch_size,
            shuffle=False
        ),
        'test_lay': DataLoader(
            datasets['test_lay'],
            batch_size=batch_size,
            shuffle=False
        ),
        'test_expert': DataLoader(
            datasets['test_expert'],
            batch_size=batch_size,
            shuffle=False
        )
    }


def get_indonli_data(model_name="cahya/roberta-base-indonesian-1.5G", batch_size=16, max_length=128):
    """
    Main function to prepare the IndoNLI data for training.
    
    Args:
        model_name: Name of the model/tokenizer to use
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing DataLoader objects for each split, and the tokenizer
    """
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Load the dataset
    dataset = load_indonli_dataset()
    
    # Preprocess the data
    processed_datasets = preprocess_data(dataset, tokenizer, max_length)
    
    # Create data loaders
    data_loaders = create_data_loaders(processed_datasets, batch_size)
    
    return {
        'data_loaders': data_loaders,
        'tokenizer': tokenizer
    }
