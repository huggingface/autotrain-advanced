import os

import evaluate
import nltk
import numpy as np


ROUGE_METRIC = evaluate.load("rouge")

MODEL_CARD = """
---
tags:
- autotrain
- text2text-generation{base_model}
widget:
- text: "I love AutoTrain"{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Seq2Seq

## Validation Metrics
{validation_metrics}
"""


def _seq2seq_metrics(pred, tokenizer):
    """
    Compute sequence-to-sequence metrics for predictions and labels.

    Args:
        pred (tuple): A tuple containing predictions and labels.
                      Predictions and labels are expected to be token IDs.
        tokenizer (PreTrainedTokenizer): The tokenizer used for decoding the predictions and labels.

    Returns:
        dict: A dictionary containing the computed ROUGE metrics and the average length of the generated sequences.
              The keys are the metric names and the values are the corresponding scores rounded to four decimal places.
    """
    predictions, labels = pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = ROUGE_METRIC.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def create_model_card(config, trainer):
    """
    Generates a model card string based on the provided configuration and trainer.

    Args:
        config (object): Configuration object containing the following attributes:
            - valid_split (optional): If not None, the function will include evaluation scores.
            - data_path (str): Path to the dataset.
            - project_name (str): Name of the project.
            - model (str): Path or identifier of the model.
        trainer (object): Trainer object with an `evaluate` method that returns evaluation metrics.

    Returns:
        str: A formatted model card string containing dataset information, validation metrics, and base model details.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
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
