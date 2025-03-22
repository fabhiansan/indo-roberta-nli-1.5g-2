---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:10330
- loss:CosineSimilarityLoss
base_model: firqaaa/indo-sentence-bert-base
widget:
- source_sentence: Daerah Bukittinggi juga terkesan asri dan hijau, sangat cocok disambangi
    bagi yang ingin menyegarkan pikiran.
  sentences:
  - Gravitasi tidak menarik apapun.
  - Raksasa spanyol Barcelona terdiri dari Rivaldo dan Patrick Kluivert waktu itu.
  - Semua orang menyambangi Bukittinggi untuk menyegarkan pikiran.
- source_sentence: Pada tanggal 24 Juni 1997, Anggun merilis album berbahasa Perancis
    pertamanya berjudul Au nom de la lune.
  sentences:
  - Jakarta berada di dataran yang lebih tinggi dari Bogor.
  - Au nom de la lune merupakan album kedua Anggun yang berbahasa Perancis.
  - Sebuah penemuan menunjukkan jika Antartika dahulu lebih mirip dengan Amerika Selatan
    saat ini.
- source_sentence: Bukan tanpa alasan Dany mengajak desainer tersebut untuk turut
    bekerjasama dengan Invasion 2017 ini.
  sentences:
  - Asia Selatan menjadi penghubung negara-negara Eropa dan negara-negara Asia Timur.
  - Desainer tersebut diajak bekerjasama dengan Invasion 2017.
  - ITB memiliki mahasiswa bernama Ahmad Faiz Sahupala.
- source_sentence: Partai Tani Indonesia adalah partai politik yang pernah ada di
    Indonesia.
  sentences:
  - Odell Beckham Jr dirawat karena membenci air putih.
  - Amir adalah politikus demokratis.
  - Salah satu partai politik yang pernah ada di Indonesia adalah Partai Tani Indonesia.
- source_sentence: Tidak hanya bagus untuk kesehatan tubuh, olahraga juga bisa mengatasi
    ngantuk. Tetapi, saat berpuasa dianjurkan untuk berolahraga setelah berbuka atau
    sahur.
  sentences:
  - JAKQ ditayangkan di televisi pada tahun 1970an.
  - Raisa masih berpacaran dengan Keena Pearce.
  - Olahraga tidak bisa mengatasi ngantuk.
datasets: []
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
model-index:
- name: SentenceTransformer based on firqaaa/indo-sentence-bert-base
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: indonli validation
      type: indonli-validation
    metrics:
    - type: pearson_cosine
      value: 0.6000309485042756
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.5781765797080199
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.594098479274715
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.5740490824874366
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.5949812269248916
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.5752843555977631
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.5982752074927783
      name: Pearson Dot
    - type: spearman_dot
      value: 0.5783714658102009
      name: Spearman Dot
    - type: pearson_max
      value: 0.6000309485042756
      name: Pearson Max
    - type: spearman_max
      value: 0.5783714658102009
      name: Spearman Max
---

# SentenceTransformer based on firqaaa/indo-sentence-bert-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base) <!-- at revision af8d649e60fbd85b6e1dee7649a749a83996304f -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Tidak hanya bagus untuk kesehatan tubuh, olahraga juga bisa mengatasi ngantuk. Tetapi, saat berpuasa dianjurkan untuk berolahraga setelah berbuka atau sahur.',
    'Olahraga tidak bisa mengatasi ngantuk.',
    'Raisa masih berpacaran dengan Keena Pearce.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity
* Dataset: `indonli-validation`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.6        |
| spearman_cosine    | 0.5782     |
| pearson_manhattan  | 0.5941     |
| spearman_manhattan | 0.574      |
| pearson_euclidean  | 0.595      |
| spearman_euclidean | 0.5753     |
| pearson_dot        | 0.5983     |
| spearman_dot       | 0.5784     |
| pearson_max        | 0.6        |
| **spearman_max**   | **0.5784** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 10,330 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                       | label                                                         |
  |:--------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                              | string                                                                           | float                                                         |
  | details | <ul><li>min: 11 tokens</li><li>mean: 30.32 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 12.0 tokens</li><li>max: 38 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                 | sentence_1                                                                               | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:-----------------|
  | <code>Ngantungan adalah desa yang berada di kecamatan Pasrepan, Kabupaten Pasuruan, Jawa Timur, Indonesia.</code>                                                                          | <code>Banyak desa di kecamatan Pasrepan.</code>                                          | <code>0.5</code> |
  | <code>Tahun lalu, Song Joong Ki merupakan peraih penghargaan yang sama dengan Park Bo Gum.</code>                                                                                          | <code>Park Bo Gum pernah memenangkan penghargaan yang sama seperti Song Joong Ki.</code> | <code>1.0</code> |
  | <code>Semua karena kandungan antioksidan, kafein, polifenol, dan vitaminnya yang beragam. Sejak dulu, teh memang dikenal sebagai minuman yang memiliki khasiat baik bagi kesehatan.</code> | <code>Salah satu minuman yang memiliki khasiat baik bagi kesehatan adalah teh.</code>    | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | indonli-validation_spearman_max |
|:-----:|:----:|:-------------------------------:|
| 1.0   | 323  | 0.5784                          |


### Framework Versions
- Python: 3.10.14
- Sentence Transformers: 3.0.1
- Transformers: 4.44.0
- PyTorch: 2.4.0+cu121
- Accelerate: 0.32.1
- Datasets: 2.20.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
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

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->