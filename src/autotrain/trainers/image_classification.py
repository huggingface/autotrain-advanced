import os

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from sklearn import metrics
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from autotrain import utils
from autotrain.params import ImageBinaryClassificationParams, ImageMultiClassClassificationParams


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
- image-classification
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
datasets:
- {dataset}
co2_eq_emissions:
  emissions: {co2}
---

# Model Trained Using AutoTrain

- Problem type: Image Classification
- CO2 Emissions (in grams): {co2:.4f}

## Validation Metricsg
{validation_metrics}
"""


class Dataset:
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]["image"]
        target = int(self.data[item]["label"])

        image = self.transforms(image=np.array(image.convert("RGB")))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "pixel_values": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(target, dtype=torch.long),
        }


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


def process_data(train_data, valid_data, image_processor):
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
    train_data = Dataset(train_data, train_transforms)
    valid_data = Dataset(valid_data, val_transforms)
    return train_data, valid_data


@utils.job_watcher
def train(co2_tracker, payload, huggingface_token, model_path):
    # create model repo
    model_repo = utils.create_repo(
        project_name=payload["proj_name"],
        autotrain_user=payload["username"],
        huggingface_token=huggingface_token,
        model_path=model_path,
    )

    data_path = f"{payload['username']}/autotrain-data-{payload['proj_name']}"
    data = load_dataset(data_path, use_auth_token=huggingface_token)
    logger.info(f"Loaded data from {data_path}")
    job_config = payload["config"]["params"][0]
    job_config["model_name"] = payload["config"]["hub_model"]

    train_data = data["train"]
    valid_data = data["validation"]

    labels = train_data.features["label"].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    num_classes = len(labels)

    model_name = job_config["model_name"]
    device = job_config.get("device", "cuda")
    # remove model_name from job config
    del job_config["model_name"]
    if num_classes == 2:
        job_config["task"] = "image_binary_classification"
        job_config = ImageBinaryClassificationParams(**job_config)
    elif num_classes > 2:
        job_config["task"] = "image_multi_class_classification"
        job_config = ImageMultiClassClassificationParams(**job_config)
    else:
        raise ValueError("Invalid number of classes")

    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        use_auth_token=huggingface_token,
    )

    model_config._num_labels = len(label2id)
    model_config.label2id = label2id
    model_config.id2label = id2label

    logger.info(model_config)

    try:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=model_config,
            use_auth_token=huggingface_token,
            ignore_mismatched_sizes=True,
        )
    except OSError:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=model_config,
            use_auth_token=huggingface_token,
            from_tf=True,
            ignore_mismatched_sizes=True,
        )

    image_processor = AutoImageProcessor.from_pretrained(model_name, use_auth_token=huggingface_token)

    train_dataset, valid_dataset = process_data(train_data, valid_data, image_processor)

    # trainer specific

    logging_steps = int(0.2 * len(valid_dataset) / job_config.train_batch_size)
    if logging_steps == 0:
        logging_steps = 1
    fp16 = True
    if device == "cpu":
        fp16 = False

    training_args = dict(
        output_dir=model_path,
        per_device_train_batch_size=job_config.train_batch_size,
        per_device_eval_batch_size=job_config.train_batch_size,
        learning_rate=job_config.learning_rate,
        num_train_epochs=job_config.num_train_epochs,
        fp16=fp16,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        logging_steps=logging_steps,
        save_total_limit=1,
        save_strategy="epoch",
        disable_tqdm=not bool(os.environ.get("ENABLE_TQDM", 0)),
        gradient_accumulation_steps=job_config.gradient_accumulation_steps,
        report_to="none",
        auto_find_batch_size=True,
        lr_scheduler_type=job_config.scheduler,
        optim=job_config.optimizer,
        warmup_ratio=job_config.warmup_ratio,
        weight_decay=job_config.weight_decay,
        max_grad_norm=job_config.max_grad_norm,
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    callbacks_to_use = [early_stop]

    args = TrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=_binary_classification_metrics if num_classes == 2 else _multi_class_classification_metrics,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()

    logger.info("Finished training")
    logger.info(trainer.state.best_metric)
    eval_scores = trainer.evaluate()

    # create and save model card
    co2_consumed = co2_tracker.stop()
    co2_consumed = co2_consumed * 1000 if co2_consumed is not None else 0

    valid_metrics = BINARY_CLASSIFICATION_EVAL_METRICS if num_classes == 2 else MULTI_CLASS_CLASSIFICATION_EVAL_METRICS
    eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
    eval_scores = "\n\n".join(eval_scores)
    model_card = MODEL_CARD.format(
        language=payload["config"]["language"],
        dataset=data_path,
        co2=co2_consumed,
        validation_metrics=eval_scores,
    )
    logger.info(model_card)
    utils.save_model_card(model_card, model_path)

    # save model, image_processor and config
    model = utils.update_model_config(trainer.model, job_config)
    utils.save_tokenizer(image_processor, model_path)
    utils.save_model(model, model_path)
    utils.remove_checkpoints(model_path=model_path)

    # push model to hub
    logger.info("Pushing model to Hub")
    model_repo.git_pull()
    model_repo.git_add()
    model_repo.git_commit(commit_message="Commit From AutoTrain")
    model_repo.git_push()
