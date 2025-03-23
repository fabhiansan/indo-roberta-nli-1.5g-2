"""
Fine-tuning script for Sentence-BERT using IndoNLI dataset.
"""

import os
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    LoggingHandler,
    losses,
    models,
    util,
    evaluation
)
from datasets import load_dataset
from utils import set_seed, setup_logging
import matplotlib.pyplot as plt
import pandas as pd
from nli_logger import create_logger, NLILogger

def load_indonli_data(split="train"):
    """
    Load the IndoNLI dataset for the specified split.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test_lay', 'test_expert')
        
    Returns:
        List of examples in the format expected by SentenceTransformer
    """
    dataset = load_dataset("afaji/indonli", split=split)
    logging.info(f"Loaded {len(dataset)} examples from {split} split")
    
    # Convert dataset to examples format expected by SentenceTransformer
    examples = []
    
    # For NLI-type training with SBERT, we follow the paper's approach:
    # - Entailment pairs are considered similar (score of 1)
    # - Contradiction pairs are considered dissimilar (score of 0)
    # - Neutral pairs can be assigned a similarity score of 0.5
    
    label_mapping = {
        "entailment": 1.0,
        "neutral": 0.5,
        "contradiction": 0.0
    }
    
    for example in dataset:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = example["label"]
        
        # Convert textual label to similarity score
        if isinstance(label, int):
            # If label is already an integer, map it to similarity score
            # Assuming 0=entailment, 1=neutral, 2=contradiction
            similarity_score = 1.0 if label == 0 else (0.5 if label == 1 else 0.0)
        else:
            # If label is a string, use the mapping
            similarity_score = label_mapping.get(label, 0.5)
        
        # Create InputExample
        examples.append(InputExample(texts=[premise, hypothesis], label=similarity_score))
    
    return examples


def load_data_for_sbert(batch_size=16):
    """
    Load all splits of the IndoNLI dataset and create DataLoaders.
    
    Args:
        batch_size: Batch size for the DataLoaders
        
    Returns:
        Dictionary containing DataLoader objects for each split
    """
    # Load all splits
    train_examples = load_indonli_data("train")
    validation_examples = load_indonli_data("validation")
    test_lay_examples = load_indonli_data("test_lay")
    test_expert_examples = load_indonli_data("test_expert")
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    return {
        "train": train_dataloader,
        "train_examples": train_examples,
        "validation": validation_examples,
        "test_lay": test_lay_examples,
        "test_expert": test_expert_examples
    }


def create_evaluator(model, examples, name):
    """
    Create an evaluator for SBERT model.
    
    Args:
        model: SentenceTransformer model
        examples: List of InputExample objects
        name: Name for the evaluator
        
    Returns:
        Evaluator object
    """
    # Extract premises and hypotheses as sentences
    sentences1 = [example.texts[0] for example in examples]
    sentences2 = [example.texts[1] for example in examples]
    
    # Extract labels
    scores = [example.label for example in examples]
    
    # Create binary classification evaluator
    return evaluation.EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        name=name
    )


def train_sbert(
    model_name="firqaaa/indo-sentence-bert-base",
    output_path="./outputs/indo-sbert-nli",
    batch_size=16,
    num_epochs=3,
    warmup_steps=0,
    learning_rate=2e-5,
    max_seq_length=128,
    seed=42
):
    """
    Train a Sentence-BERT model on the IndoNLI dataset.
    
    Args:
        model_name: Pre-trained model name or path
        output_path: Directory to save the model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        seed: Random seed
    """
    # Set random seed
    set_seed(seed)
    
    # Set up logging
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load the pre-trained SBERT model
    logging.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Optionally, adjust the max sequence length
    if max_seq_length != model.get_max_seq_length():
        logging.info(f"Setting max sequence length to {max_seq_length}")
        model.max_seq_length = max_seq_length
    
    # Load data
    logging.info("Loading data")
    data = load_data_for_sbert(batch_size=batch_size)
    
    # Create evaluator
    evaluator = create_evaluator(
        model, 
        data["validation"], 
        "indonli-validation"
    )
    
    # Set up the loss function for NLI
    logging.info("Setting up CosineSimilarityLoss for training")
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train the model
    logging.info(f"Starting training for {num_epochs} epochs")
    model.fit(
        train_objectives=[(data["train"], train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=output_path,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )
    
    # Evaluate on test sets
    logging.info("Evaluating on test_lay split")
    test_lay_evaluator = create_evaluator(
        model, 
        data["test_lay"], 
        "indonli-test-lay"
    )
    test_lay_score = test_lay_evaluator(model)
    logging.info(f"Test Lay Score: {test_lay_score}")
    
    logging.info("Evaluating on test_expert split")
    test_expert_evaluator = create_evaluator(
        model, 
        data["test_expert"], 
        "indonli-test-expert"
    )
    test_expert_score = test_expert_evaluator(model)
    logging.info(f"Test Expert Score: {test_expert_score}")
    
    # Save evaluation results
    with open(os.path.join(output_path, "evaluation_results.txt"), "w") as f:
        f.write(f"Validation Score: {evaluator(model)}\n")
        f.write(f"Test Lay Score: {test_lay_score}\n")
        f.write(f"Test Expert Score: {test_expert_score}\n")
    
    logging.info(f"Model saved to {output_path}")
    return model


def train_sbert_with_logger(
    model_name="firqaaa/indo-sentence-bert-base",
    output_path="./outputs/indo-sbert-nli",
    batch_size=16,
    num_epochs=3,
    warmup_steps=0,
    learning_rate=2e-5,
    max_seq_length=128,
    seed=42,
    logger=None
):
    """
    Train a Sentence-BERT model on the IndoNLI dataset using NLILogger.
    
    Args:
        model_name: Pre-trained model name or path
        output_path: Directory to save the model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        seed: Random seed
        logger: NLILogger instance
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Set random seed
    set_seed(seed)
    
    # Log start of training
    logger.logger.info(f"Starting similarity-based SBERT training with model: {model_name}")
    
    # Load dataset
    try:
        logger.logger.info("Loading IndoNLI dataset from Hugging Face")
        
        # Load dataset splits
        train_data = load_dataset("indonli", split="train")
        dev_data = load_dataset("indonli", split="validation")
        test_lay_data = load_dataset("indonli", split="test_lay")
        test_expert_data = load_dataset("indonli", split="test_expert")
        
        # Log dataset statistics
        logger.log_dataset_statistics({
            "train": train_data,
            "validation": dev_data,
            "test_lay": test_lay_data,
            "test_expert": test_expert_data
        })
        
    except Exception as e:
        logger.logger.error(f"Error loading dataset from Hugging Face: {e}")
        
        try:
            # Try loading from local files
            logger.logger.info("Attempting to load dataset from local files")
            
            train_data = load_dataset("json", data_files="data/indonli/train.json", split="train")
            dev_data = load_dataset("json", data_files="data/indonli/valid.json", split="train")
            test_lay_data = load_dataset("json", data_files="data/indonli/test_lay.json", split="train")
            test_expert_data = load_dataset("json", data_files="data/indonli/test_expert.json", split="train")
            
            # Log dataset statistics
            logger.log_dataset_statistics({
                "train": train_data,
                "validation": dev_data,
                "test_lay": test_lay_data,
                "test_expert": test_expert_data
            })
            
        except Exception as e2:
            logger.logger.error(f"Error loading dataset from local files: {e2}")
            raise RuntimeError(f"Failed to load dataset: {e}, {e2}")
    
    # Convert datasets to examples
    logger.logger.info("Converting dataset to SBERT examples")
    
    # For NLI-type training with SBERT, we follow the paper's approach:
    # - Entailment pairs are considered similar (score of 1)
    # - Contradiction pairs are considered dissimilar (score of 0)
    # - Neutral pairs can be assigned a similarity score of 0.5
    
    label_mapping = {
        "entailment": 1.0,
        "neutral": 0.5,
        "contradiction": 0.0
    }
    
    # Log the label mapping
    logger.logger.info(f"Using label mapping: {label_mapping}")
    
    # Convert datasets to SBERT format
    train_examples = []
    for item in train_data:
        if item["label"] in label_mapping:
            score = label_mapping[item["label"]]
            train_examples.append(InputExample(texts=[item["premise"], item["hypothesis"]], label=score))
    
    dev_examples = []
    for item in dev_data:
        if item["label"] in label_mapping:
            score = label_mapping[item["label"]]
            dev_examples.append(InputExample(texts=[item["premise"], item["hypothesis"]], label=score))
    
    test_lay_examples = []
    for item in test_lay_data:
        if item["label"] in label_mapping:
            score = label_mapping[item["label"]]
            test_lay_examples.append(InputExample(texts=[item["premise"], item["hypothesis"]], label=score))
    
    test_expert_examples = []
    for item in test_expert_data:
        if item["label"] in label_mapping:
            score = label_mapping[item["label"]]
            test_expert_examples.append(InputExample(texts=[item["premise"], item["hypothesis"]], label=score))
    
    logger.logger.info(f"Created {len(train_examples)} training examples")
    logger.logger.info(f"Created {len(dev_examples)} validation examples")
    logger.logger.info(f"Created {len(test_lay_examples)} test_lay examples")
    logger.logger.info(f"Created {len(test_expert_examples)} test_expert examples")
    
    # Create model
    logger.logger.info(f"Creating model from {model_name}")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Create evaluator for validation data
    logger.logger.info("Creating evaluator")
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        dev_examples,
        name="indonli-dev"
    )
    
    # Create train dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Calculate warmup steps if not provided
    if warmup_steps == 0:
        warmup_steps = int(len(train_dataloader) * 0.1)
        
    logger.logger.info(f"Using {warmup_steps} warmup steps")
    
    # Create loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Setup training arguments
    train_args = {
        "epochs": num_epochs,
        "evaluation_steps": int(len(train_dataloader) * 0.2),  # Evaluate every 20% of an epoch
        "evaluator": evaluator,
        "warmup_steps": warmup_steps,
        "output_path": output_path,
        "optimizer_params": {"lr": learning_rate},
    }
    
    # Log training arguments
    logger.logger.info(f"Training arguments: {train_args}")
    
    # Define custom callback to log with NLILogger
    class LoggerCallback:
        def __init__(self, logger):
            self.logger = logger
            self.step = 0
            self.epoch = 0
            self.best_score = -1
            self.best_checkpoint = None
        
        def on_epoch_end(self, score, epoch, steps):
            self.epoch = epoch
            
            # Log epoch
            val_metrics = {
                "cosine_similarity": score,
                # Approximate accuracy from cosine similarity
                "accuracy": (score + 1) / 2,  # Convert from [-1, 1] to [0, 1]
                "macro_f1": None  # Not available for cosine similarity
            }
            
            checkpoint_path = os.path.join(output_path, f"checkpoint-{epoch}")
            
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=None,  # Not available from callback
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_path
            )
            
            # Track best checkpoint
            if score > self.best_score:
                self.best_score = score
                self.best_checkpoint = checkpoint_path
                self.logger.logger.info(f"New best model: {checkpoint_path}")
        
        def on_step_end(self, score, epoch, steps, loss):
            self.step += 1
            
            # Log every 100 steps
            if self.step % 100 == 0:
                self.logger.log_train_step(
                    epoch=epoch,
                    step=self.step,
                    loss=loss,
                    additional_metrics={"cosine_similarity": score}
                )
    
    # Create logger callback
    logger_callback = LoggerCallback(logger)
    
    # Train the model
    logger.logger.info("Starting training")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        callback=logger_callback.on_step_end,
        epoch_callback=logger_callback.on_epoch_end,
        **train_args
    )
    
    # Evaluate on test sets
    logger.logger.info("Evaluating on test sets")
    
    # Function to get similarity scores
    def get_similarity_scores(model, examples):
        sentences1 = [example.texts[0] for example in examples]
        sentences2 = [example.texts[1] for example in examples]
        gold_scores = [example.label for example in examples]
        
        # Get embeddings
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        cosine_scores = cosine_scores.diag().cpu().numpy()
        
        # Convert similarity scores to predictions
        # Map scores to classes: [0.0-0.33] -> contradiction, [0.34-0.66] -> neutral, [0.67-1.0] -> entailment
        predictions = []
        for score in cosine_scores:
            if score < 0.33:
                predictions.append(2)  # Contradiction
            elif score < 0.66:
                predictions.append(1)  # Neutral
            else:
                predictions.append(0)  # Entailment
        
        # Convert gold scores to labels
        # Map scores to classes: 0.0 -> contradiction, 0.5 -> neutral, 1.0 -> entailment
        gold_labels = []
        for score in gold_scores:
            if score < 0.33:
                gold_labels.append(2)  # Contradiction
            elif score < 0.66:
                gold_labels.append(1)  # Neutral
            else:
                gold_labels.append(0)  # Entailment
        
        return predictions, gold_labels
    
    # Evaluate on test_lay
    logger.logger.info("Evaluating on test_lay dataset")
    test_lay_predictions, test_lay_labels = get_similarity_scores(model, test_lay_examples)
    logger.log_evaluation("test_lay", test_lay_predictions, test_lay_labels, checkpoint_name="final")
    
    # Evaluate on test_expert
    logger.logger.info("Evaluating on test_expert dataset")
    test_expert_predictions, test_expert_labels = get_similarity_scores(model, test_expert_examples)
    logger.log_evaluation("test_expert", test_expert_predictions, test_expert_labels, checkpoint_name="final")
    
    # Generate visualizations and report
    logger.plot_training_curve()
    logger.compare_checkpoints()
    logger.generate_summary_report()
    
    logger.logger.info("Training and evaluation complete")
    
    return model


def test_model(model_path, test_examples):
    """
    Test a trained Sentence-BERT model on example pairs.
    
    Args:
        model_path: Path to the trained model
        test_examples: List of examples to test
    """
    # Load the model
    model = SentenceTransformer(model_path)
    
    # Create an evaluator
    evaluator = create_evaluator(
        model, 
        test_examples, 
        "test"
    )
    
    # Evaluate
    score = evaluator(model)
    print(f"Test Score: {score}")
    
    # Show some examples
    for i, example in enumerate(test_examples[:5]):
        premise, hypothesis = example.texts
        expected_score = example.label
        
        # Encode sentences
        embedding1 = model.encode(premise, convert_to_tensor=True)
        embedding2 = model.encode(hypothesis, convert_to_tensor=True)
        
        # Compute similarity
        cosine_score = util.cos_sim(embedding1, embedding2).item()
        
        print(f"\nExample {i+1}:")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Expected similarity: {expected_score}")
        print(f"Model similarity: {cosine_score}")
        
        # Classify based on similarity threshold
        if cosine_score >= 0.7:
            nli_class = "entailment"
        elif cosine_score <= 0.3:
            nli_class = "contradiction"
        else:
            nli_class = "neutral"
            
        print(f"NLI Classification: {nli_class}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Sentence-BERT on IndoNLI dataset")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="firqaaa/indo-sentence-bert-base",
                        help="Pre-trained model name or path")
    
    # Training parameters
    parser.add_argument("--output_path", type=str, default="./outputs/indo-sbert-nli",
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # New logger parameter
    parser.add_argument("--use_new_logger", action="store_true",
                        help="Use the new NLILogger")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup either traditional logging or the new NLILogger
    if args.use_new_logger:
        logger = create_logger("similarity-sbert", args.output_path)
        logger.log_hyperparameters(vars(args))
        
        # Train with new logger
        train_sbert_with_logger(
            model_name=args.model_name,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            logger=logger
        )
    else:
        # Setup traditional logging
        logger = setup_logging()
        
        # Train with traditional methods
        train_sbert(
            model_name=args.model_name,
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
