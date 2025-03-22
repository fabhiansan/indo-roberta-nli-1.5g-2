"""
Inference script for the Indonesian NLI model.
"""

import argparse
import torch
from transformers import RobertaTokenizer

from model import RobertaNLIModel
from utils import set_seed, create_label_map


def parse_args():
    """
    Parse command line arguments for inference.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Inference for Indonesian NLI model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model or model name on Hugging Face Hub")
    parser.add_argument("--premise", type=str, required=True,
                        help="Premise text")
    parser.add_argument("--hypothesis", type=str, required=True,
                        help="Hypothesis text")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def predict(model, tokenizer, premise, hypothesis, device):
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        premise: Premise text
        hypothesis: Hypothesis text
        device: Device to run inference on
        
    Returns:
        Prediction label and probabilities
    """
    # Tokenize inputs
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get prediction
        prediction = torch.argmax(logits, dim=1).item()
        
    return prediction, probabilities[0].cpu().numpy()


def main():
    """
    Main inference function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = RobertaNLIModel.from_pretrained(args.model_path)
    model.to(device)
    
    # Create or load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    
    # Make prediction
    label_id, probabilities = predict(
        model,
        tokenizer,
        args.premise,
        args.hypothesis,
        device
    )
    
    # Get label map
    label_map = create_label_map()
    
    # Print results
    print("\nPrediction Results:")
    print(f"Premise: {args.premise}")
    print(f"Hypothesis: {args.hypothesis}")
    print(f"Predicted relation: {label_map[label_id]} (ID: {label_id})")
    print("\nProbabilities:")
    for label_id, label_name in label_map.items():
        print(f"{label_name}: {probabilities[label_id]:.4f}")


if __name__ == "__main__":
    main()
