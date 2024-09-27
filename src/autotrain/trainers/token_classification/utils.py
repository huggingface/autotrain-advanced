import os

import numpy as np
from seqeval import metrics


MODEL_CARD = """
---
tags:
- autotrain
- token-classification{base_model}
widget:
- text: "I love AutoTrain"{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Token Classification

## Validation Metrics
{validation_metrics}
"""


def token_classification_metrics(pred, label_list):
    """
    Compute token classification metrics including precision, recall, F1 score, and accuracy.

    Args:
        pred (tuple): A tuple containing predictions and labels.
                      Predictions should be a 3D array (batch_size, sequence_length, num_labels).
                      Labels should be a 2D array (batch_size, sequence_length).
        label_list (list): A list of label names corresponding to the indices used in predictions and labels.

    Returns:
        dict: A dictionary containing the following metrics:
              - "precision": Precision score of the token classification.
              - "recall": Recall score of the token classification.
              - "f1": F1 score of the token classification.
              - "accuracy": Accuracy score of the token classification.
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[predi] for (predi, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lbl] for (predi, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {
        "precision": metrics.precision_score(true_labels, true_predictions),
        "recall": metrics.recall_score(true_labels, true_predictions),
        "f1": metrics.f1_score(true_labels, true_predictions),
        "accuracy": metrics.accuracy_score(true_labels, true_predictions),
    }
    return results


def create_model_card(config, trainer):
    """
    Generates a model card string based on the provided configuration and trainer.

    Args:
        config (object): Configuration object containing model and dataset information.
        trainer (object): Trainer object used to evaluate the model.

    Returns:
        str: A formatted model card string with dataset tags, validation metrics, and base model information.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = ["eval_loss", "eval_precision", "eval_recall", "eval_f1", "eval_accuracy"]
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
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
