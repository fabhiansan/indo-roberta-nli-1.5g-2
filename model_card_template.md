---
language:
- id
license: apache-2.0
tags:
- indonesian
- nli
- roberta
- text-classification
- natural-language-inference
- cahya/roberta-base-indonesian-1.5G
datasets:
- afaji/indonli
metrics:
- accuracy
- f1
---

# Model Card for fabhiansan/indo-roberta-1.5g-nli

This model is a fine-tuned version of [cahya/roberta-base-indonesian-1.5G](https://huggingface.co/cahya/roberta-base-indonesian-1.5G) for Natural Language Inference (NLI) tasks in Indonesian. It's designed to determine the inferential relationship between a premise and hypothesis, classifying it as entailment, neutral, or contradiction.

## Model Details

### Model Description

**Model Architecture:** This model is based on the RoBERTa architecture, fine-tuned for sequence classification. The base model is a pre-trained Indonesian RoBERTa model with 125M parameters. The classification head consists of a linear layer that maps the [CLS] token representation to three output classes.

**Model Type:** Transformer-based sequence classification model
**Language:** Indonesian
**License:** Apache 2.0
**Base Model:** cahya/roberta-base-indonesian-1.5G (125M parameters)
**Fine-tuning Task:** Natural Language Inference (3-way classification)

## Training Data

The model was fine-tuned on the [afaji/indonli](https://huggingface.co/datasets/afaji/indonli) dataset, which contains:

- **Training set:** Approximately 10,500 premise-hypothesis pairs
- **Validation set:** Approximately 2,200 premise-hypothesis pairs
- **Test sets:** 
  - test_lay (lay person annotated): Approximately 2,300 pairs
  - test_expert (expert annotated): Approximately 2,600 pairs

Each example in the dataset is labeled as one of three classes:
- **Entailment:** The hypothesis logically follows from the premise
- **Neutral:** The hypothesis might be true given the premise, but doesn't necessarily follow
- **Contradiction:** The hypothesis contradicts the premise

### Preprocessing

The input data was tokenized using the RoBERTa tokenizer with the following settings:
- Maximum sequence length: 128 tokens
- Truncation: Yes
- Padding: To maximum length
- Special tokens: Added as per RoBERTa requirements

## Training Procedure

### Training Hyperparameters

The model was trained with the following hyperparameters:
- **Optimizer:** AdamW
- **Learning rate:** 2e-5
- **Batch size:** 16
- **Gradient accumulation steps:** 1
- **Weight decay:** 0.01
- **Epochs:** 5
- **Early stopping:** Yes, based on validation loss with patience of 3 epochs
- **Learning rate scheduler:** Linear with warmup (10% of training steps)

### Hardware

The model was trained on a CUDA-compatible GPU.

## Evaluation Results

The model was evaluated on the validation set and both test sets (lay and expert). Performance metrics include:

### Validation Set
- **Accuracy:** ~72%
- **Macro F1 Score:** ~72%

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| entailment | ~72% | ~74% | ~73% |
| neutral | ~69% | ~66% | ~67% |
| contradiction | ~74% | ~75% | ~74% |

### Test Sets
Performance metrics on the test sets demonstrate the model's generalization capability.

## Limitations and Biases

- The model is specifically trained for Indonesian language and may not perform well on other languages or code-switched text.
- Performance may vary on domain-specific texts that differ significantly from the training data.
- Like all language models, this model may reflect biases present in the training data.
- The model is not designed for use in high-stakes decision-making without human oversight.

## Intended Use and Ethical Considerations

This model is intended to be used for:
- Research on Indonesian natural language understanding
- Semantic text similarity tasks
- Information retrieval systems
- Educational tools for Indonesian language learning

**Not recommended for:**
- Critical decision-making without human review
- Applications where errors could lead to harmful consequences
- Contexts where perfect accuracy is required

## How to Use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("fabhiansan/indo-roberta-1.5g-nli")
tokenizer = AutoTokenizer.from_pretrained("fabhiansan/indo-roberta-1.5g-nli")

# Prepare input texts
premise = "Keindahan alam yang terdapat di Gunung Batu Jonggol ini dapat Anda manfaatkan sebagai objek fotografi yang cantik."
hypothesis = "Keindahan alam tidak dapat difoto."

# Tokenize input
inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)

# Get prediction
import torch
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Map predictions to labels
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
predicted_class_id = predictions.argmax().item()
predicted_label = id2label[predicted_class_id]

print(f"Premise: {premise}")
print(f"Hypothesis: {hypothesis}")
print(f"Prediction: {predicted_label}")
print(f"Probabilities: {predictions[0].tolist()}")
```

## Citation and Contact

### Citation

If you use this model in your research, please cite:

```
@misc{fabhiansan2025indonli,
  author = {Fabhiansan},
  title = {RoBERTa-based NLI Model for Indonesian},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/fabhiansan/indo-roberta-1.5g-nli}}
}
```

### Contact

For questions or feedback about this model, please contact me through the Hugging Face platform or raise an issue in the model repository.