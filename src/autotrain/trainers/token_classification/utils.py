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
