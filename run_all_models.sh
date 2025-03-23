#!/bin/bash

# Run all NLI models with the same parameters and save metrics in a common location
# This script runs all five models:
# 1. Classifier-based SBERT
# 2. Similarity-based SBERT
# 3. Advanced SBERT
# 4. Improved SBERT
# 5. RoBERTa model

# Create timestamp for unified results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./model_comparison_${TIMESTAMP}"
echo "Creating results directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Set common parameters for all models
BASE_MODEL="firqaaa/indo-sentence-bert-base"
ROBERTA_MODEL="cahya/roberta-base-indonesian-1.5G"
NUM_EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
MAX_LENGTH=128
SEED=42

# Set environment variables for better CUDA handling
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Log the start of model comparison
echo "=== Starting Model Comparison (${TIMESTAMP}) ===" | tee "${RESULTS_DIR}/comparison.log"
echo "Parameters:" | tee -a "${RESULTS_DIR}/comparison.log"
echo "  Base Model: ${BASE_MODEL}" | tee -a "${RESULTS_DIR}/comparison.log"
echo "  RoBERTa Model: ${ROBERTA_MODEL}" | tee -a "${RESULTS_DIR}/comparison.log"
echo "  Epochs: ${NUM_EPOCHS}" | tee -a "${RESULTS_DIR}/comparison.log"
echo "  Batch Size: ${BATCH_SIZE}" | tee -a "${RESULTS_DIR}/comparison.log"
echo "  Learning Rate: ${LEARNING_RATE}" | tee -a "${RESULTS_DIR}/comparison.log"
echo "  Seed: ${SEED}" | tee -a "${RESULTS_DIR}/comparison.log"

# Function to run a model and log its output
run_model() {
    MODEL_NAME=$1
    COMMAND=$2
    
    echo "" | tee -a "${RESULTS_DIR}/comparison.log"
    echo "=== Running ${MODEL_NAME} ===" | tee -a "${RESULTS_DIR}/comparison.log"
    echo "Command: ${COMMAND}" | tee -a "${RESULTS_DIR}/comparison.log"
    echo "Start time: $(date)" | tee -a "${RESULTS_DIR}/comparison.log"
    
    # Execute the command and tee output to both stdout and a model-specific log file
    eval "${COMMAND}" 2>&1 | tee "${RESULTS_DIR}/${MODEL_NAME}.log"
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "${MODEL_NAME} completed successfully at $(date)" | tee -a "${RESULTS_DIR}/comparison.log"
    else
        echo "${MODEL_NAME} failed at $(date)" | tee -a "${RESULTS_DIR}/comparison.log"
    fi
}

# Create checkpoints directory for each model
mkdir -p "${RESULTS_DIR}/roberta/checkpoints"
mkdir -p "${RESULTS_DIR}/classifier/checkpoints"
mkdir -p "${RESULTS_DIR}/similarity/checkpoints"
mkdir -p "${RESULTS_DIR}/advanced/checkpoints"
mkdir -p "${RESULTS_DIR}/improved/checkpoints"

# 5. Run RoBERTa Model First (since this is the only one confirmed to work)
echo "Running RoBERTa model first..."
ROBERTA_CMD="python train.py \
    --model_name ${ROBERTA_MODEL} \
    --output_dir ${RESULTS_DIR}/roberta \
    --log_dir ${RESULTS_DIR}/roberta/logs \
    --batch_size ${BATCH_SIZE} \
    --epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --use_nli_logger"

run_model "roberta-nli" "${ROBERTA_CMD}"

# 1. Run Classifier-based SBERT
echo "Running Classifier-based SBERT..."
CLASSIFIER_CMD="python indo_sbert_train.py \
    --base_model ${BASE_MODEL} \
    --output_dir ${RESULTS_DIR}/classifier \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --use_new_logger"

run_model "classifier-sbert" "${CLASSIFIER_CMD}"

# 2. Run Similarity-based SBERT
echo "Running Similarity-based SBERT..."
SIMILARITY_CMD="python sbert_train.py \
    --model_name ${BASE_MODEL} \
    --output_path ${RESULTS_DIR}/similarity \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --use_new_logger"

run_model "similarity-sbert" "${SIMILARITY_CMD}"

# 3. Run Advanced SBERT
echo "Running Advanced SBERT..."
ADVANCED_CMD="python indo_sbert_advanced_train.py \
    --base_model ${BASE_MODEL} \
    --output_dir ${RESULTS_DIR}/advanced \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --use_new_logger"

run_model "advanced-sbert" "${ADVANCED_CMD}"

# 4. Run Improved SBERT
echo "Running Improved SBERT..."
IMPROVED_CMD="python train_improved_sbert.py \
    --base_model ${BASE_MODEL} \
    --pooling_mode mean_pooling \
    --classifier_hidden_size 256 \
    --dropout 0.2 \
    --use_cross_attention \
    --output_dir ${RESULTS_DIR}/improved \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --use_new_logger"

run_model "improved-sbert" "${IMPROVED_CMD}"

# Generate comparison results
echo "" | tee -a "${RESULTS_DIR}/comparison.log"
echo "=== Model Comparison Complete ===" | tee -a "${RESULTS_DIR}/comparison.log"
echo "End time: $(date)" | tee -a "${RESULTS_DIR}/comparison.log"

# Run comparison script
python compare_model_results.py --results_dir "${RESULTS_DIR}" | tee -a "${RESULTS_DIR}/comparison.log"

echo "All models have been trained and evaluated. Results are in ${RESULTS_DIR}"
echo "To view the comparison, check ${RESULTS_DIR}/comparison.log"
