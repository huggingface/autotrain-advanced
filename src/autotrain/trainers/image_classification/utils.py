import os

import albumentations as A
import numpy as np
from sklearn import metrics

from autotrain.trainers.image_classification.dataset import ImageClassificationDataset


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
- image-classification{base_model}
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Image Classification

## Validation Metrics
{validation_metrics}
"""


def _binary_classification_metrics(pred):
    """
    Computes various binary classification metrics given the predictions and labels.

    Args:
        pred (tuple): A tuple containing raw predictions and true labels.
                      raw_predictions (numpy.ndarray): The raw prediction scores from the model.
                      labels (numpy.ndarray): The true labels.

    Returns:
        dict: A dictionary containing the following metrics:
            - f1 (float): The F1 score.
            - precision (float): The precision score.
            - recall (float): The recall score.
            - auc (float): The Area Under the ROC Curve (AUC) score.
            - accuracy (float): The accuracy score.
    """
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
    """
    Compute various classification metrics for multi-class classification.

    Args:
        pred (tuple): A tuple containing raw predictions and true labels.
                      - raw_predictions (numpy.ndarray): The raw prediction scores for each class.
                      - labels (numpy.ndarray): The true labels.

    Returns:
        dict: A dictionary containing the following metrics:
              - "f1_macro": F1 score with macro averaging.
              - "f1_micro": F1 score with micro averaging.
              - "f1_weighted": F1 score with weighted averaging.
              - "precision_macro": Precision score with macro averaging.
              - "precision_micro": Precision score with micro averaging.
              - "precision_weighted": Precision score with weighted averaging.
              - "recall_macro": Recall score with macro averaging.
              - "recall_micro": Recall score with micro averaging.
              - "recall_weighted": Recall score with weighted averaging.
              - "accuracy": Accuracy score.
    """
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


def process_data(train_data, valid_data, image_processor, config):
    """
    Processes training and validation data for image classification.

    Args:
        train_data (Dataset): The training dataset.
        valid_data (Dataset or None): The validation dataset. Can be None if no validation data is provided.
        image_processor (ImageProcessor): An object containing image processing parameters such as size, mean, and std.
        config (dict): Configuration dictionary containing additional parameters for dataset processing.

    Returns:
        tuple: A tuple containing the processed training dataset and the processed validation dataset (or None if no validation data is provided).
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
    train_data = ImageClassificationDataset(train_data, train_transforms, config)
    if valid_data is not None:
        valid_data = ImageClassificationDataset(valid_data, val_transforms, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config, trainer, num_classes):
    """
    Generates a model card for the given configuration and trainer.

    Args:
        config (object): Configuration object containing various settings.
        trainer (object): Trainer object used for model training and evaluation.
        num_classes (int): Number of classes in the classification task.

    Returns:
        str: A formatted string representing the model card.

    The function evaluates the model if a validation split is provided in the config.
    It then formats the evaluation scores based on whether the task is binary or multi-class classification.
    If no validation split is provided, it notes that no validation metrics are available.

    The function also checks the data path and model path in the config to determine if they are directories.
    Based on these checks, it formats the dataset tag and base model information accordingly.

    Finally, it uses the formatted information to create and return the model card string.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = (
            BINARY_CLASSIFICATION_EVAL_METRICS if num_classes == 2 else MULTI_CLASS_CLASSIFICATION_EVAL_METRICS
        )
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
