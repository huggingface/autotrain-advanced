import os

import albumentations as A
import numpy as np
from sklearn import metrics

from autotrain.trainers.image_regression.dataset import ImageRegressionDataset


VALID_METRICS = [
    "eval_loss",
    "eval_mse",
    "eval_mae",
    "eval_r2",
    "eval_rmse",
    "eval_explained_variance",
]

MODEL_CARD = """
---
tags:
- autotrain
- vision
- image-classification
- image-regression{base_model}
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Image Regression

## Validation Metrics

{validation_metrics}
"""


def image_regression_metrics(pred):
    """
    Calculate various regression metrics for image regression tasks.

    Args:
        pred (tuple): A tuple containing raw predictions and labels.
                      raw_predictions should be a list of lists or a list of numpy.float32 values.
                      labels should be a list of true values.

    Returns:
        dict: A dictionary containing the calculated metrics:
              - 'mse': Mean Squared Error
              - 'mae': Mean Absolute Error
              - 'r2': R^2 Score
              - 'rmse': Root Mean Squared Error
              - 'explained_variance': Explained Variance Score

              If an error occurs during the calculation of a metric, the value for that metric will be -999.
    """
    raw_predictions, labels = pred

    try:
        raw_predictions = [r for preds in raw_predictions for r in preds]
    except TypeError as err:
        if "numpy.float32" not in str(err):
            raise Exception(err)

    pred_dict = {}
    metrics_to_calculate = {
        "mse": metrics.mean_squared_error,
        "mae": metrics.mean_absolute_error,
        "r2": metrics.r2_score,
        "rmse": lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "explained_variance": metrics.explained_variance_score,
    }

    for key, func in metrics_to_calculate.items():
        try:
            pred_dict[key] = float(func(labels, raw_predictions))
        except Exception:
            pred_dict[key] = -999

    return pred_dict


def process_data(train_data, valid_data, image_processor, config):
    """
    Processes training and validation data by applying image transformations.

    Args:
        train_data (Dataset): The training dataset.
        valid_data (Dataset or None): The validation dataset. If None, only training data is processed.
        image_processor (ImageProcessor): An object containing image processing parameters such as size, mean, and std.
        config (dict): Configuration dictionary containing additional parameters for the dataset.

    Returns:
        tuple: A tuple containing the processed training dataset and the processed validation dataset (or None if valid_data is None).
    """
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    try:
        height, width = size
    except TypeError:
        height = size
        width = size

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    train_data = ImageRegressionDataset(train_data, train_transforms, config)
    if valid_data is not None:
        valid_data = ImageRegressionDataset(valid_data, val_transforms, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config, trainer):
    """
    Generates a model card string based on the provided configuration and trainer.

    Args:
        config (object): Configuration object containing various settings such as
                         valid_split, data_path, project_name, and model.
        trainer (object): Trainer object used to evaluate the model if validation
                          split is provided.

    Returns:
        str: A formatted model card string containing dataset information,
             validation metrics, and base model details.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in VALID_METRICS]
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
