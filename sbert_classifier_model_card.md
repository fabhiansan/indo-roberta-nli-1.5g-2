---
pipeline_tag: text-classification
tags:
- sentence-transformers
- text-classification
- natural-language-inference 
- nli
- transformers
- indonesian
- id
license: apache-2.0
language:
- id
library_name: sentence-transformers
datasets:
- afaji/indonli
---

# indo-sentence-bert-classifier-nli

This model combines the power of [sentence-transformers](https://www.SBERT.net) with an explicit classification head for Natural Language Inference (NLI) tasks in Indonesian. Unlike pure sentence embedding models, this hybrid approach directly outputs classification probabilities for entailment, neutral, and contradiction relationships between premise-hypothesis pairs.

## Model Description

The model consists of two main components:
1. **SBERT Encoder**: A pre-trained [firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base) model that generates 768-dimensional embeddings for Indonesian text
2. **Classification Head**: A fully-connected neural network that takes the combined sentence embeddings and predicts NLI relationships

**Key characteristics:**
- **Architecture**: SBERT + MLP Classifier
- **Base Model**: firqaaa/indo-sentence-bert-base
- **Embedding Dimension**: 768
- **Hidden Layer Size**: 512
- **Output Classes**: 3 (entailment, neutral, contradiction)
- **Encoding Combination**: Concatenation of premise and hypothesis embeddings
- **Max Sequence Length**: 128 tokens

## Training Data

The model was fine-tuned on the [afaji/indonli](https://huggingface.co/datasets/afaji/indonli) dataset, which contains:

- **Training set:** ~10,500 premise-hypothesis pairs
- **Validation set:** ~2,200 premise-hypothesis pairs  
- **Test sets:**
  - test_lay (lay person annotated): ~2,300 pairs
  - test_expert (expert annotated): ~2,600 pairs

Each example is labeled as:
- **Entailment**: The hypothesis logically follows from the premise
- **Neutral**: The hypothesis might be true given the premise, but doesn't necessarily follow
- **Contradiction**: The hypothesis contradicts the premise

## Usage

```python
from sbert_classifier_model import SBERTWithClassifier

# Load the model
model = SBERTWithClassifier.load("fabhiansan/indo-sentence-bert-classifier-nli")

# Define premise-hypothesis pairs for classification
premises = [
    "Keindahan alam yang terdapat di Gunung Batu Jonggol ini dapat Anda manfaatkan sebagai objek fotografi yang cantik.",
    "Jakarta adalah ibu kota Indonesia."
]
hypotheses = [
    "Keindahan alam tidak dapat difoto.",
    "Jakarta terletak di Pulau Jawa."
]

# Get predictions with probabilities
labels, probs = model.predict(premises, hypotheses, return_probabilities=True)

# Print results
for i in range(len(premises)):
    print(f"Premise: {premises[i]}")
    print(f"Hypothesis: {hypotheses[i]}")
    print(f"Prediction: {labels[i]}")
    print(f"Probabilities: {probs[i]}")
    print()
```

## How It's Different from Regular SBERT

Unlike a standard sentence-transformer model that primarily focuses on generating embeddings, this model:

1. **Directly outputs classification decisions** rather than requiring similarity threshold tuning
2. **Combines sentence representations in multiple ways** (concatenation, difference, multiplication)
3. **Includes a fully-connected neural network classifier** for improved performance
4. **Has been explicitly trained for the 3-way NLI classification task**

The standard SBERT approach treats NLI as a similarity task, while this model treats it as a direct classification task.

## Training Procedure

### Training Hyperparameters

The model was trained with the following hyperparameters:
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Weight Decay**: 0.01
- **Epochs**: 5 (with early stopping)
- **Loss Function**: Cross Entropy
- **Dropout Rate**: 0.1
- **Early Stopping**: Based on validation loss with patience of 3 epochs

### Architecture Details

```
SBERTWithClassifier(
  SBERT: SentenceTransformer(
    (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
    (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  )
  Classifier: Sequential(
    (0): Linear(in_features=1536, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=512, out_features=3, bias=True)
  )
)
```

## Evaluation Results

The model was evaluated on validation and test sets from the IndoNLI dataset:

### Validation Set
- **Accuracy**: ~74%
- **Macro F1 Score**: ~74%

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| entailment | ~75% | ~76% | ~75% |
| neutral | ~71% | ~68% | ~69% |
| contradiction | ~76% | ~77% | ~77% |

### Test Sets (Metrics may vary)
- **Test Lay Accuracy**: ~73%
- **Test Expert Accuracy**: ~72%

## Limitations and Biases

- The model is specifically trained for Indonesian language and may not perform well on other languages or code-switched text
- Performance may vary on domain-specific texts that differ significantly from the training data
- Like all language models, this model may reflect biases present in the training data
- The model is not designed for use in high-stakes decision-making without human oversight

## Citing & Authors

If you use this model in your research, please cite:

```
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
```

```
@misc{arasyi2022indo,
  author = {Arasyi, Firqa},
  title  = {indo-sentence-bert: Sentence Transformer for Bahasa Indonesia with Multiple Negative Ranking Loss},
  year = {2022},
  month = {9},
  publisher = {huggingface},
  journal = {huggingface repository},
  howpublished = {https://huggingface.co/firqaaa/indo-sentence-bert-base}
}
```

```
@misc{fabhiansan2025indobertnli,
  author = {Fabhiansan},
  title = {Fine-tuned SBERT with Classifier for Indonesian Natural Language Inference},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/fabhiansan/indo-sentence-bert-classifier-nli}}
}
```
