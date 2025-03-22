---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- indonesian
- nli
- natural-language-inference
license: apache-2.0
language:
- id
library_name: sentence-transformers
datasets:
- afaji/indonli
---

# indo-sentence-bert-nli

This is a [sentence-transformers](https://www.SBERT.net) model fine-tuned on the [IndoNLI dataset](https://huggingface.co/datasets/afaji/indonli): It maps sentences & paragraphs in Indonesian to a 768 dimensional dense vector space and can be used for tasks like clustering, semantic search, and especially natural language inference.

## Model Description

The model is based on [firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base) and has been fine-tuned specifically for natural language inference (NLI) tasks in Indonesian. It uses a Siamese network architecture to encode premise and hypothesis pairs and learn their semantic relationships.

- **Base Model**: firqaaa/indo-sentence-bert-base
- **Training Data**: afaji/indonli dataset with ~10,500 premise-hypothesis pairs
- **Task**: Natural Language Inference (determining if a hypothesis entails, contradicts, or is neutral to a premise)
- **Output Dimensions**: 768
- **Max Sequence Length**: 128

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('fabhiansan/indo-sentence-bert-nli')

# Example NLI pair
premise = "Keindahan alam yang terdapat di Gunung Batu Jonggol ini dapat Anda manfaatkan sebagai objek fotografi yang cantik."
hypothesis = "Keindahan alam tidak dapat difoto."

# Encode sentences
embedding1 = model.encode(premise, convert_to_tensor=True)
embedding2 = model.encode(hypothesis, convert_to_tensor=True)

# Compute similarity score
cosine_score = util.cos_sim(embedding1, embedding2).item()
print(f"Similarity score: {cosine_score:.4f}")

# NLI classification based on similarity thresholds
# Note: These thresholds can be adjusted based on your specific needs
if cosine_score >= 0.7:
    nli_class = "entailment"
elif cosine_score <= 0.3:
    nli_class = "contradiction"
else:
    nli_class = "neutral"
    
print(f"NLI classification: {nli_class}")
```

## Semantic Search Example

The model can also be used for semantic search:

```python
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('fabhiansan/indo-sentence-bert-nli')

# Corpus
corpus = [
    "Presiden Jokowi meresmikan jalan tol baru di Jakarta.",
    "Harga bahan bakar minyak naik di seluruh Indonesia.",
    "Timnas Indonesia berhasil mengalahkan Malaysia di pertandingan AFF Cup.",
    "Gunung Semeru meletus dan mengeluarkan awan panas."
]

# Encode corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Query
query = "Kenaikan harga BBM di Indonesia"
query_embedding = model.encode(query, convert_to_tensor=True)

# Find most similar sentences
search_results = util.semantic_search(query_embedding, corpus_embeddings)

# Print results
print(f"Query: {query}")
print("Search results:")
for i, result in enumerate(search_results[0]):
    print(f"{i+1}. {corpus[result['corpus_id']]} (Score: {result['score']:.4f})")
```

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

## Training Procedure

The model was trained with the following parameters:

**Loss Function**: CosineSimilarityLoss with the following mapping:
- Entailment pairs: similarity score of 1.0
- Neutral pairs: similarity score of 0.5
- Contradiction pairs: similarity score of 0.0

**Training Hyperparameters**:
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5
- Max Sequence Length: 128
- Optimizer: AdamW with weight decay of 0.01

## Evaluation Results

The model was evaluated on the validation and test sets of the IndoNLI dataset. Performance metrics include:

- **Validation Embedding Similarity Score**: [To be filled with actual performance]
- **Test Lay Embedding Similarity Score**: [To be filled with actual performance]
- **Test Expert Embedding Similarity Score**: [To be filled with actual performance]

## Limitations and Biases

- The model is specifically trained for Indonesian language and may not perform well on other languages or code-switched text.
- Performance may vary on domain-specific texts that differ significantly from the training data.
- Like all language models, this model may reflect biases present in the training data.
- The model is not designed for use in high-stakes decision-making without human oversight.

## Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

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
  title = {Fine-tuned SBERT for Indonesian Natural Language Inference},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/fabhiansan/indo-sentence-bert-nli}}
}
```
