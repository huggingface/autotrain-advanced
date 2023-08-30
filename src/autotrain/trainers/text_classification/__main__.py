import argparse
import json
import os

import pandas as pd
from accelerate.state import PartialState
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from autotrain import logger
from autotrain.trainers.text_classification import utils
from autotrain.trainers.text_classification.dataset import TextClassificationDataset
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.utils import monitor


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = TextClassificationParams(**config)

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

    if PartialState().process_index == 0:
        logger.info("Starting training...")
        logger.info(f"Training config: {config}")

    train_data = None
    valid_data = None
    # check if config.train_split.csv exists in config.data_path
    if config.train_split is not None:
        train_path = f"{config.data_path}/{config.train_split}.csv"
        if os.path.exists(train_path):
            logger.info("loading dataset from csv")
            train_data = pd.read_csv(train_path)
            train_data = Dataset.from_pandas(train_data)
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                token=config.token,
            )

    if config.valid_split is not None:
        valid_path = f"{config.data_path}/{config.valid_split}.csv"
        if os.path.exists(valid_path):
            logger.info("loading dataset from csv")
            valid_data = pd.read_csv(valid_path)
            valid_data = Dataset.from_pandas(valid_data)
        else:
            valid_data = load_dataset(
                config.data_path,
                split=config.valid_split,
                token=config.token,
            )

    classes = train_data.features[config.target_column].names
    label2id = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError("Invalid number of classes. Must be greater than 1.")

    if config.valid_split is not None:
        num_classes_valid = len(valid_data.unique(config.target_column))
        if num_classes_valid != num_classes:
            raise ValueError(
                f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {num_classes_valid}"
            )

    model_config = AutoConfig.from_pretrained(config.model, num_labels=num_classes)
    model_config._num_labels = len(label2id)
    model_config.label2id = label2id
    model_config.id2label = {v: k for k, v in label2id.items()}

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model, config=model_config, trust_remote_code=True, token=config.token
        )
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model, config=model_config, from_tf=True, trust_remote_code=True, token=config.token
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model, token=config.token, trust_remote_code=True)
    train_data = TextClassificationDataset(data=train_data, tokenizer=tokenizer, config=config)
    if config.valid_split is not None:
        valid_data = TextClassificationDataset(data=valid_data, tokenizer=tokenizer, config=config)

    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1

    else:
        logging_steps = config.logging_steps

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        fp16=config.fp16,
        evaluation_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to="tensorboard",
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
    )

    if config.valid_split is not None:
        early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    args = TrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=utils._binary_classification_metrics
        if num_classes == 2
        else utils._multi_class_classification_metrics,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)
    tokenizer.save_pretrained(config.project_name)

    model_card = utils.create_model_card(config, trainer, num_classes)

    # remove token key from training_params.json located in output directory
    # first check if file exists
    if os.path.exists(f"{config.project_name}/training_params.json"):
        training_params = json.load(open(f"{config.project_name}/training_params.json"))
        training_params.pop("token")
        json.dump(training_params, open(f"{config.project_name}/training_params.json", "w"))

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(repo_id=config.repo_id, repo_type="model", private=True)
            api.upload_folder(folder_path=config.project_name, repo_id=config.repo_id, repo_type="model")

    if PartialState().process_index == 0:
        if "SPACE_ID" in os.environ:
            # shut down the space
            logger.info("Pausing space...")
            api = HfApi(token=config.token)
            api.pause_space(repo_id=os.environ["SPACE_ID"])

        if "ENDPOINT_ID" in os.environ:
            # shut down the endpoint
            logger.info("Pausing endpoint...")
            utils.pause_endpoint(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = TextClassificationParams(**training_config)
    train(config)
