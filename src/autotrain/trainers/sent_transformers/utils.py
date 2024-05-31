import os

from autotrain import logger


MODEL_CARD = """
---
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- autotrain{base_model}
widget:
- source_sentence: 'search_query: i love autotrain'
  sentences:
  - 'search_query: huggingface auto train'
  - 'search_query: hugging face auto train'
  - 'search_query: i love autotrain'
pipeline_tag: sentence-similarity{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Sentence Transformers

## Validation Metrics
{validation_metrics}

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the Hugging Face Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'search_query: autotrain',
    'search_query: auto train',
    'search_query: i love autotrain',
]
embeddings = model.encode(sentences)
print(embeddings.shape)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
```
"""


def process_columns(data, config):
    # trainers: pair, pair_class, pair_score, triplet, qa
    # pair: anchor, positive
    # pair_class: premise, hypothesis, label
    # pair_score: sentence1, sentence2, score
    # triplet: anchor, positive, negative
    # qa: query, answer
    if config.trainer == "pair":
        if not (config.sentence1_column == "anchor" and config.sentence1_column in data.column_names):
            data = data.rename_column(config.sentence1_column, "anchor")
        if not (config.sentence2_column == "positive" and config.sentence2_column in data.column_names):
            data = data.rename_column(config.sentence2_column, "positive")
    elif config.trainer == "pair_class":
        if not (config.sentence1_column == "premise" and config.sentence1_column in data.column_names):
            data = data.rename_column(config.sentence1_column, "premise")
        if not (config.sentence2_column == "hypothesis" and config.sentence2_column in data.column_names):
            data = data.rename_column(config.sentence2_column, "hypothesis")
        if not (config.target_column == "label" and config.target_column in data.column_names):
            data = data.rename_column(config.target_column, "label")
    elif config.trainer == "pair_score":
        if not (config.sentence1_column == "sentence1" and config.sentence1_column in data.column_names):
            data = data.rename_column(config.sentence1_column, "sentence1")
        if not (config.sentence2_column == "sentence2" and config.sentence2_column in data.column_names):
            data = data.rename_column(config.sentence2_column, "sentence2")
        if not (config.target_column == "score" and config.target_column in data.column_names):
            data = data.rename_column(config.target_column, "score")
    elif config.trainer == "triplet":
        if not (config.sentence1_column == "anchor" and config.sentence1_column in data.column_names):
            data = data.rename_column(config.sentence1_column, "anchor")
        if not (config.sentence2_column == "positive" and config.sentence2_column in data.column_names):
            data = data.rename_column(config.sentence2_column, "positive")
        if not (config.sentence3_column == "negative" and config.sentence3_column in data.column_names):
            data = data.rename_column(config.sentence3_column, "negative")
    elif config.trainer == "qa":
        if not (config.sentence1_column == "query" and config.sentence1_column in data.column_names):
            data = data.rename_column(config.sentence1_column, "query")
        if not (config.sentence2_column == "answer" and config.sentence2_column in data.column_names):
            data = data.rename_column(config.sentence2_column, "answer")
    else:
        raise ValueError(f"Invalid trainer: {config.trainer}")
    return data


def create_model_card(config, trainer):
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        logger.info(eval_scores)
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items()]
        eval_scores = "\n\n".join(eval_scores)
    else:
        eval_scores = "No validation metrics available"

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card
