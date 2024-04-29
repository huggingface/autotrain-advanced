import numpy as np
from sklearn import metrics


SINGLE_COLUMN_REGRESSION_EVAL_METRICS = (
    "eval_loss",
    "eval_mse",
    "eval_mae",
    "eval_r2",
    "eval_rmse",
    "eval_explained_variance",
)


MODEL_CARD = """
---
tags:
- autotrain
- text-regression
widget:
- text: "I love AutoTrain"
datasets:
- {dataset}
---

# Model Trained Using AutoTrain

- Problem type: Text Regression

## Validation Metrics
{validation_metrics}
"""


def single_column_regression_metrics(pred):
    raw_predictions, labels = pred

    # try:
    #     raw_predictions = [r for preds in raw_predictions for r in preds]
    # except TypeError as err:
    #     if "numpy.float32" not in str(err):
    #         raise Exception(err)

    def safe_compute(metric_func, default=-999):
        try:
            return metric_func(labels, raw_predictions)
        except Exception:
            return default

    pred_dict = {
        "mse": safe_compute(lambda labels, predictions: metrics.mean_squared_error(labels, predictions)),
        "mae": safe_compute(lambda labels, predictions: metrics.mean_absolute_error(labels, predictions)),
        "r2": safe_compute(lambda labels, predictions: metrics.r2_score(labels, predictions)),
        "rmse": safe_compute(lambda labels, predictions: np.sqrt(metrics.mean_squared_error(labels, predictions))),
        "explained_variance": safe_compute(
            lambda labels, predictions: metrics.explained_variance_score(labels, predictions)
        ),
    }

    for key, value in pred_dict.items():
        pred_dict[key] = float(value)
    return pred_dict


def create_model_card(config, trainer):
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        eval_scores = [
            f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in SINGLE_COLUMN_REGRESSION_EVAL_METRICS
        ]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    model_card = MODEL_CARD.format(
        dataset=config.data_path,
        validation_metrics=eval_scores,
    )
    return model_card
