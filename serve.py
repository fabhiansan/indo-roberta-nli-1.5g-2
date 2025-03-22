"""
API server for the Indonesian NLI model using FastAPI.
"""

import os
import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer

from model import RobertaNLIModel
from utils import set_seed, create_label_map


# Define request and response models
class NLIRequest(BaseModel):
    premise: str
    hypothesis: str


class NLIResponse(BaseModel):
    label: str
    label_id: int
    probabilities: dict


# Initialize FastAPI app
app = FastAPI(
    title="Indonesian NLI API",
    description="API for Natural Language Inference using RoBERTa for Indonesian",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
label_map = None


def load_model(model_path):
    """
    Load the model and tokenizer.
    
    Args:
        model_path: Path to the model checkpoint
    """
    global model, tokenizer, device, label_map
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = RobertaNLIModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    # Create label map
    label_map = create_label_map()
    
    print(f"Model loaded from {model_path} on {device}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Indonesian NLI API is running. Use /predict for inference."}


@app.post("/predict", response_model=NLIResponse)
async def predict(request: NLIRequest):
    """
    Predict NLI label for premise and hypothesis.
    
    Args:
        request: NLIRequest containing premise and hypothesis
        
    Returns:
        NLIResponse with label, label_id, and probabilities
    """
    global model, tokenizer, device, label_map
    
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Tokenize inputs
    inputs = tokenizer(
        request.premise,
        request.hypothesis,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get prediction
        prediction = torch.argmax(logits, dim=1).item()
    
    # Prepare response
    prob_dict = {label_map[i]: float(probabilities[0][i]) for i in range(len(label_map))}
    
    return NLIResponse(
        label=label_map[prediction],
        label_id=prediction,
        probabilities=prob_dict
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Serve Indonesian NLI model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load model
    load_model(args.model_path)
    
    # Run server
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
