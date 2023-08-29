import os

import numpy as np
import requests
from sklearn import metrics


BINARY_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1",
    "eval_auc",
    "eval_precision",
    "eval_recall",
)

MULTI_CLASS_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1_macro",
    "eval_f1_micro",
    "eval_f1_weighted",
    "eval_precision_macro",
    "eval_precision_micro",
    "eval_precision_weighted",
    "eval_recall_macro",
    "eval_recall_micro",
    "eval_recall_weighted",
)

MODEL_CARD = """
---
tags:
- autotrain
- text-classification
widget:
- text: "I love AutoTrain"
datasets:
- {dataset}
---

# Model Trained Using AutoTrain

- Problem type: Text Classification

## Validation Metrics
{validation_metrics}
"""


def _binary_classification_metrics(pred):
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    result = {
        "f1": metrics.f1_score(labels, predictions),
        "precision": metrics.precision_score(labels, predictions),
        "recall": metrics.recall_score(labels, predictions),
        "auc": metrics.roc_auc_score(labels, raw_predictions[:, 1]),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return result


def _multi_class_classification_metrics(pred):
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    results = {
        "f1_macro": metrics.f1_score(labels, predictions, average="macro"),
        "f1_micro": metrics.f1_score(labels, predictions, average="micro"),
        "f1_weighted": metrics.f1_score(labels, predictions, average="weighted"),
        "precision_macro": metrics.precision_score(labels, predictions, average="macro"),
        "precision_micro": metrics.precision_score(labels, predictions, average="micro"),
        "precision_weighted": metrics.precision_score(labels, predictions, average="weighted"),
        "recall_macro": metrics.recall_score(labels, predictions, average="macro"),
        "recall_micro": metrics.recall_score(labels, predictions, average="micro"),
        "recall_weighted": metrics.recall_score(labels, predictions, average="weighted"),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return results


def create_model_card(config, trainer, num_classes):
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = (
            BINARY_CLASSIFICATION_EVAL_METRICS if num_classes == 2 else MULTI_CLASS_CLASSIFICATION_EVAL_METRICS
        )
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    model_card = MODEL_CARD.format(
        dataset=config.data_path,
        validation_metrics=eval_scores,
    )
    return model_card


def pause_endpoint(params):
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {params.token}"}
    r = requests.post(api_url, headers=headers)
    return r.json()
