"""
Script to push the fine-tuned custom SBERT models to Hugging Face Hub.
Supports both classification-based and similarity-based models.
"""

import os
import argparse
import logging
import json
import torch
import shutil
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi, Repository
from utils import setup_logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def create_model_card(
    model_id,
    model_type,
    output_path,
    model_card_template=None,
    config=None,
    evaluation_results=None
):
    """
    Create a model card for the fine-tuned model.
    
    Args:
        model_id: Model ID on the HF Hub
        model_type: Type of model ("classifier" or "similarity")
        output_path: Path to save the model card
        model_card_template: Path to the model card template
        config: Model configuration
        evaluation_results: Evaluation results
        
    Returns:
        Path to the created model card
    """
    model_card_path = os.path.join(output_path, "README.md")
    
    # Default model description
    if model_type == "classifier":
        description = "A BERT-based model fine-tuned for Natural Language Inference with a classification head."
        usage_example = """```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load model and tokenizer
bert = AutoModel.from_pretrained("fabhiansan/indo-sbert-nli-classifier")
tokenizer = AutoTokenizer.from_pretrained("fabhiansan/indo-sbert-nli-classifier")

# Load classifier weights
classifier_path = "classifier.pt"  # This file is included in the model repository
classifier = nn.Linear(bert.config.hidden_size * 3, 3)  # For entailment, neutral, contradiction
classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device("cpu")))

# Function for mean pooling
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Example NLI inputs
premise = "Keindahan alam yang terdapat di Gunung Batu Jonggol ini dapat Anda manfaatkan sebagai objek fotografi yang cantik."
hypothesis = "Keindahan alam tidak dapat difoto."

# Encode inputs
encoded_premise = tokenizer(premise, padding=True, truncation=True, return_tensors="pt")
encoded_hypothesis = tokenizer(hypothesis, padding=True, truncation=True, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs_premise = bert(**encoded_premise)
    outputs_hypothesis = bert(**encoded_hypothesis)
    
    # Mean pooling
    embedding_premise = mean_pooling(outputs_premise.last_hidden_state, encoded_premise["attention_mask"])
    embedding_hypothesis = mean_pooling(outputs_hypothesis.last_hidden_state, encoded_hypothesis["attention_mask"])
    
    # Concatenate embeddings with element-wise difference
    diff = torch.abs(embedding_premise - embedding_hypothesis)
    concatenated = torch.cat([embedding_premise, embedding_hypothesis, diff], dim=1)
    
    # Get logits and predictions
    logits = classifier(concatenated)
    predictions = F.softmax(logits, dim=1)

# Map predictions to labels
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
predicted_class_id = predictions.argmax().item()
predicted_label = id2label[predicted_class_id]

print(f"Premise: {premise}")
print(f"Hypothesis: {hypothesis}")
print(f"Prediction: {predicted_label}")
print(f"Probabilities: {predictions[0].tolist()}")
```"""
    else:  # similarity model
        description = "A BERT-based model fine-tuned for Natural Language Inference using a similarity approach."
        usage_example = """```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

# Load model and tokenizer
model = AutoModel.from_pretrained("fabhiansan/indo-sbert-nli-similarity")
tokenizer = AutoTokenizer.from_pretrained("fabhiansan/indo-sbert-nli-similarity")

# Function for mean pooling
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Example NLI inputs
premise = "Keindahan alam yang terdapat di Gunung Batu Jonggol ini dapat Anda manfaatkan sebagai objek fotografi yang cantik."
hypothesis = "Keindahan alam tidak dapat difoto."

# Encode inputs
encoded_premise = tokenizer(premise, padding=True, truncation=True, return_tensors="pt")
encoded_hypothesis = tokenizer(hypothesis, padding=True, truncation=True, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    # Get embeddings
    outputs_premise = model(**encoded_premise)
    outputs_hypothesis = model(**encoded_hypothesis)
    
    # Mean pooling
    embedding_premise = mean_pooling(outputs_premise.last_hidden_state, encoded_premise["attention_mask"])
    embedding_hypothesis = mean_pooling(outputs_hypothesis.last_hidden_state, encoded_hypothesis["attention_mask"])
    
    # Normalize embeddings
    embedding_premise = F.normalize(embedding_premise, p=2, dim=1)
    embedding_hypothesis = F.normalize(embedding_hypothesis, p=2, dim=1)
    
    # Compute similarity
    similarity = F.cosine_similarity(embedding_premise, embedding_hypothesis).item()

# Convert similarity to NLI label
if similarity >= 0.7:
    label = "entailment"
elif similarity <= 0.3:
    label = "contradiction"
else:
    label = "neutral"

print(f"Premise: {premise}")
print(f"Hypothesis: {hypothesis}")
print(f"Similarity: {similarity:.4f}")
print(f"NLI Label: {label}")
```"""
    
    # If a template is provided, use it
    if model_card_template and os.path.exists(model_card_template):
        with open(model_card_template, "r", encoding="utf-8") as f:
            template = f.read()
        
        # Replace placeholders
        model_card = template.replace("MODEL_ID", model_id)
        
        # Add evaluation results if provided
        if evaluation_results and os.path.exists(evaluation_results):
            with open(evaluation_results, "r", encoding="utf-8") as f:
                eval_text = f.read()
            # Find a suitable place to insert evaluation results
            if "## Evaluation Results" in model_card:
                model_card = model_card.replace("## Evaluation Results", f"## Evaluation Results\n\n{eval_text}")
    else:
        # Create a basic model card
        model_card = f"""---
language:
- id
license: apache-2.0
tags:
- indonesian
- nli
- bert
- sentence-embeddings
- natural-language-inference
- firqaaa/indo-sentence-bert-base
datasets:
- afaji/indonli
metrics:
- accuracy
---

# {model_id}

{description}

## Model Details

This model is a fine-tuned version of [firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base) for Natural Language Inference (NLI) tasks in Indonesian. It uses a {model_type}-based approach to determine the inferential relationship between a premise and hypothesis, classifying it as entailment, neutral, or contradiction.

## Training Data

The model was fine-tuned on the [afaji/indonli](https://huggingface.co/datasets/afaji/indonli) dataset, which contains Indonesian premise-hypothesis pairs labeled with entailment, neutral, or contradiction.

## Evaluation Results

"""
        # Add evaluation results if provided
        if evaluation_results and os.path.exists(evaluation_results):
            with open(evaluation_results, "r", encoding="utf-8") as f:
                model_card += f.read() + "\n\n"
        else:
            model_card += "Evaluation metrics will be added soon.\n\n"
        
        # Add usage example
        model_card += f"""## Usage

{usage_example}

## Limitations and Biases

- The model is specifically trained for Indonesian language and may not perform well on other languages or code-switched text.
- Performance may vary on domain-specific texts that differ significantly from the training data.
- Like all language models, this model may reflect biases present in the training data.

## Citation

If you use this model in your research, please cite:

```
@misc{{fabhiansan2025indonli,
  author = {{Fabhiansan}},
  title = {{Fine-tuned SBERT for Indonesian Natural Language Inference}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{model_id}}}}}
}}
```

And also cite the original SBERT and Indo-SBERT works:

```
@inproceedings{{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}}
```

```
@misc{{arasyi2022indo,
  author = {{Arasyi, Firqa}},
  title = {{indo-sentence-bert: Sentence Transformer for Bahasa Indonesia with Multiple Negative Ranking Loss}},
  year = {{2022}},
  month = {{9}},
  publisher = {{huggingface}},
  journal = {{huggingface repository}},
  howpublished = {{https://huggingface.co/firqaaa/indo-sentence-bert-base}}
}}
```
"""
    
    # Save model card
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    
    return model_card_path


def prepare_checkpoint(
    checkpoint_path,
    output_dir,
    model_type,
):
    """
    Prepare checkpoint for pushing to HF Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        output_dir: Directory to save prepared files
        model_type: Type of model ("classifier" or "similarity")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model files
    for file in os.listdir(checkpoint_path):
        # Skip large binary files that are not needed
        if file in ["optimizer.pt", "scheduler.pt", "training_args.bin"]:
            continue
        
        source = os.path.join(checkpoint_path, file)
        target = os.path.join(output_dir, file)
        
        if os.path.isdir(source):
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    
    # Add model_type to config
    try:
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            config["model_type_extension"] = model_type
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not update config.json: {e}")
    
    return output_dir


def push_to_hub(
    prepared_dir,
    hub_model_id,
    organization=None,
    commit_message=None,
    token=None,
):
    """
    Push model to the Hugging Face Hub.
    
    Args:
        prepared_dir: Directory with prepared model files
        hub_model_id: Model ID on the HF Hub
        organization: Optional organization name
        commit_message: Commit message for the push
        token: HF API token
        
    Returns:
        Boolean indicating success
    """
    try:
        # Set repository ID
        if organization:
            repo_id = f"{organization}/{hub_model_id}"
        else:
            repo_id = hub_model_id
        
        # Set commit message
        if commit_message is None:
            commit_message = f"Upload model {Path(prepared_dir).name}"
        
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        # Clone or create repo
        repo_url = api.create_repo(
            repo_id=repo_id,
            exist_ok=True,
            repo_type="model",
        )
        logging.info(f"Repository URL: {repo_url}")
        
        # Clone repository
        repo_dir = f"tmp_repo_{hub_model_id.replace('/', '_')}"
        repo = Repository(
            local_dir=repo_dir,
            clone_from=repo_id,
            use_auth_token=True,
        )
        
        # Copy files from prepared directory to repo
        for item in os.listdir(prepared_dir):
            source = os.path.join(prepared_dir, item)
            target = os.path.join(repo_dir, item)
            
            if os.path.isdir(source):
                if os.path.exists(target):
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        
        # Push to hub
        repo.git_add(".")
        repo.git_commit(commit_message)
        repo.git_push()
        
        logging.info(f"Successfully pushed model to {repo_id}")
        
        # Clean up
        shutil.rmtree(repo_dir, ignore_errors=True)
        
        return True
    
    except Exception as e:
        logging.error(f"Error pushing model to hub: {e}")
        return False


def push_classifier_model(
    checkpoint_path,
    hub_model_id,
    organization=None,
    model_card_template=None,
    token=None,
):
    """
    Push a classifier model to the Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        hub_model_id: Model ID on the HF Hub
        organization: Optional organization name
        model_card_template: Path to the model card template
        token: HF API token
        
    Returns:
        Boolean indicating success
    """
    # Create a temporary directory for preparing files
    temp_dir = f"temp_{hub_model_id.replace('/', '_')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Prepare checkpoint
        prepared_dir = prepare_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=temp_dir,
            model_type="classifier",
        )
        
        # Create model card
        evaluation_results = os.path.join(checkpoint_path, "..", "evaluation_results.txt")
        model_card_path = create_model_card(
            model_id=hub_model_id,
            model_type="classifier",
            output_path=prepared_dir,
            model_card_template=model_card_template,
            evaluation_results=evaluation_results if os.path.exists(evaluation_results) else None,
        )
        
        # Push to hub
        success = push_to_hub(
            prepared_dir=prepared_dir,
            hub_model_id=hub_model_id,
            organization=organization,
            token=token,
        )
        
        return success
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def push_similarity_model(
    checkpoint_path,
    hub_model_id,
    organization=None,
    model_card_template=None,
    token=None,
):
    """
    Push a similarity model to the Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        hub_model_id: Model ID on the HF Hub
        organization: Optional organization name
        model_card_template: Path to the model card template
        token: HF API token
        
    Returns:
        Boolean indicating success
    """
    # Create a temporary directory for preparing files
    temp_dir = f"temp_{hub_model_id.replace('/', '_')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Prepare checkpoint
        prepared_dir = prepare_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=temp_dir,
            model_type="similarity",
        )
        
        # Create model card
        evaluation_results = os.path.join(checkpoint_path, "..", "evaluation_results.txt")
        model_card_path = create_model_card(
            model_id=hub_model_id,
            model_type="similarity",
            output_path=prepared_dir,
            model_card_template=model_card_template,
            evaluation_results=evaluation_results if os.path.exists(evaluation_results) else None,
        )
        
        # Push to hub
        success = push_to_hub(
            prepared_dir=prepared_dir,
            hub_model_id=hub_model_id,
            organization=organization,
            token=token,
        )
        
        return success
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def push_all_checkpoints(
    classifier_dir,
    similarity_dir,
    hub_classifier_id,
    hub_similarity_id,
    organization=None,
    classifier_card=None,
    similarity_card=None,
    token=None,
):
    """
    Push all checkpoints to the Hugging Face Hub.
    
    Args:
        classifier_dir: Directory containing classifier checkpoints
        similarity_dir: Directory containing similarity checkpoints
        hub_classifier_id: Base model ID for classifier
        hub_similarity_id: Base model ID for similarity
        organization: Optional organization name
        classifier_card: Path to the classifier model card template
        similarity_card: Path to the similarity model card template
        token: HF API token
    """
    # Set up logging
    logger = setup_logging()
    
    # Push classifier checkpoints
    if os.path.exists(classifier_dir):
        logging.info(f"Processing classifier checkpoints from {classifier_dir}")
        
        # Get list of checkpoint directories
        checkpoints = [
            d for d in os.listdir(classifier_dir) 
            if os.path.isdir(os.path.join(classifier_dir, d)) and 
            (d.startswith("checkpoint-") or d == "best")
        ]
        
        if not checkpoints:
            logging.warning(f"No checkpoints found in {classifier_dir}")
        else:
            logging.info(f"Found {len(checkpoints)} classifier checkpoints to push")
            
            # Sort checkpoints by step number (put "best" at the end)
            def sort_key(checkpoint):
                if checkpoint == "best":
                    return float('inf')  # Always at the end
                return int(checkpoint.split("-")[-1]) if checkpoint.split("-")[-1].isdigit() else 0
                
            checkpoints = sorted(checkpoints, key=sort_key)
            
            # Process the best model first
            if "best" in checkpoints:
                checkpoints.remove("best")
                checkpoints.append("best")
            
            # Process each checkpoint
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(classifier_dir, checkpoint)
                
                # Set model ID
                if checkpoint == "best":
                    model_id = hub_classifier_id
                else:
                    # Extract step number
                    step = checkpoint.split("-")[-1] if "-" in checkpoint else "unknown"
                    model_id = f"{hub_classifier_id}-step-{step}"
                
                logging.info(f"Pushing classifier checkpoint {checkpoint} as {model_id}")
                
                success = push_classifier_model(
                    checkpoint_path=checkpoint_path,
                    hub_model_id=model_id,
                    organization=organization,
                    model_card_template=classifier_card,
                    token=token,
                )
                
                if not success and checkpoint == "best":
                    logging.error("Failed to push the main classifier model.")
    else:
        logging.warning(f"Classifier directory {classifier_dir} does not exist")
    
    # Push similarity checkpoints
    if os.path.exists(similarity_dir):
        logging.info(f"Processing similarity checkpoints from {similarity_dir}")
        
        # Get list of checkpoint directories
        checkpoints = [
            d for d in os.listdir(similarity_dir) 
            if os.path.isdir(os.path.join(similarity_dir, d)) and 
            (d.startswith("checkpoint-") or d == "best")
        ]
        
        if not checkpoints:
            logging.warning(f"No checkpoints found in {similarity_dir}")
        else:
            logging.info(f"Found {len(checkpoints)} similarity checkpoints to push")
            
            # Sort checkpoints by step number (put "best" at the end)
            def sort_key(checkpoint):
                if checkpoint == "best":
                    return float('inf')  # Always at the end
                return int(checkpoint.split("-")[-1]) if checkpoint.split("-")[-1].isdigit() else 0
                
            checkpoints = sorted(checkpoints, key=sort_key)
            
            # Process the best model first
            if "best" in checkpoints:
                checkpoints.remove("best")
                checkpoints.append("best")
            
            # Process each checkpoint
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(similarity_dir, checkpoint)
                
                # Set model ID
                if checkpoint == "best":
                    model_id = hub_similarity_id
                else:
                    # Extract step number
                    step = checkpoint.split("-")[-1] if "-" in checkpoint else "unknown"
                    model_id = f"{hub_similarity_id}-step-{step}"
                
                logging.info(f"Pushing similarity checkpoint {checkpoint} as {model_id}")
                
                success = push_similarity_model(
                    checkpoint_path=checkpoint_path,
                    hub_model_id=model_id,
                    organization=organization,
                    model_card_template=similarity_card,
                    token=token,
                )
                
                if not success and checkpoint == "best":
                    logging.error("Failed to push the main similarity model.")
    else:
        logging.warning(f"Similarity directory {similarity_dir} does not exist")
    
    logging.info("Finished pushing all checkpoints")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Push custom SBERT models to Hugging Face Hub")
    
    parser.add_argument(
        "--classifier_dir",
        type=str,
        help="Directory containing classifier model checkpoints",
    )
    parser.add_argument(
        "--similarity_dir",
        type=str,
        help="Directory containing similarity model checkpoints",
    )
    parser.add_argument(
        "--hub_classifier_id",
        type=str,
        default="indo-sbert-nli-classifier",
        help="Model ID for classifier on the HF Hub",
    )
    parser.add_argument(
        "--hub_similarity_id",
        type=str,
        default="indo-sbert-nli-similarity",
        help="Model ID for similarity model on the HF Hub",
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Optional organization name",
    )
    parser.add_argument(
        "--classifier_card",
        type=str,
        default=None,
        help="Path to the classifier model card template",
    )
    parser.add_argument(
        "--similarity_card",
        type=str,
        default=None,
        help="Path to the similarity model card template",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF API token (optional, will use the token from the HF CLI if not provided)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    push_all_checkpoints(
        classifier_dir=args.classifier_dir,
        similarity_dir=args.similarity_dir,
        hub_classifier_id=args.hub_classifier_id,
        hub_similarity_id=args.hub_similarity_id,
        organization=args.organization,
        classifier_card=args.classifier_card,
        similarity_card=args.similarity_card,
        token=args.token,
    )


if __name__ == "__main__":
    main()
