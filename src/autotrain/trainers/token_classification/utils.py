import numpy as np
from datasets import load_metric


_METRICS = load_metric("seqeval")

MODEL_CARD = """
---
tags:
- autotrain
- token-classification
widget:
- text: "I love AutoTrain"
datasets:
- {dataset}
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

    results = _METRICS.compute(predictions=true_predictions, references=true_labels)
    results = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
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

    model_card = MODEL_CARD.format(
        dataset=config.data_path,
        validation_metrics=eval_scores,
    )
    return model_card
