# Indonesian RoBERTa NLI

A training pipeline for fine-tuning RoBERTa models for Natural Language Inference (NLI) on Indonesian texts.

## Overview

This repository contains scripts for training and evaluating a RoBERTa-based model for Natural Language Inference on Indonesian language data. The model is fine-tuned on the afaji/indonli dataset.

## Model

The training script fine-tunes the pre-trained model:
- `cahya/roberta-base-indonesian-1.5G`

## Dataset

We use the `afaji/indonli` dataset, which is split as follows:
- Train: 10,330 examples (lay annotators)
- Valid: 2,197 examples (lay annotators)
- Test Lay: 2,201 examples (lay annotators)
- Test Expert: 2,984 examples (expert annotators)

### Data Format Example
```json
{
  "premise": "Keindahan alam yang terdapat di Gunung Batu Jonggol ini dapat Anda manfaatkan sebagai objek fotografi yang cantik.", 
  "hypothesis": "Keindahan alam tidak dapat difoto.", 
  "label": 2
}
```

Labels:
- 0: entailment
- 1: neutral
- 2: contradiction

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python train.py --eval_only --checkpoint [path_to_checkpoint]
```

## Configuration

You can customize the training parameters in the `config.py` file.
