#!/bin/bash

# Script to train the improved SBERT model with error handling

# Set environment variables for better CUDA error reporting
export CUDA_LAUNCH_BLOCKING=1    # Get better error messages
export TORCH_USE_CUDA_DSA=1      # Use CUDA device-side assertions for debugging

# Run the training script with optimal parameters based on the classifier model
python train_improved_sbert.py \
  --base_model firqaaa/indo-sentence-bert-base \
  --pooling_mode mean_pooling \
  --classifier_hidden_size 256 \
  --dropout 0.2 \
  --use_cross_attention \
  --output_dir outputs/indo-sbert-improved \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --num_epochs 10 \
  --patience 3 \
  --seed 42
