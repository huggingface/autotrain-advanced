import argparse
import json
from functools import partial

from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import EarlyStoppingCallback
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.sent_transformers import utils
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = SentenceTransformersParams(**config)

    train_data = None
    valid_data = None
    # check if config.train_split.csv exists in config.data_path
    if config.train_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            logger.info("loading dataset from disk")
            train_data = load_from_disk(config.data_path)[config.train_split]
        else:
            if ":" in config.train_split:
                dataset_config_name, split = config.train_split.split(":")
                train_data = load_dataset(
                    config.data_path,
                    name=dataset_config_name,
                    split=split,
                    token=config.token,
                )
            else:
                train_data = load_dataset(
                    config.data_path,
                    split=config.train_split,
                    token=config.token,
                )

    if config.valid_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            logger.info("loading dataset from disk")
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            if ":" in config.valid_split:
                dataset_config_name, split = config.valid_split.split(":")
                valid_data = load_dataset(
                    config.data_path,
                    name=dataset_config_name,
                    split=split,
                    token=config.token,
                )
            else:
                valid_data = load_dataset(
                    config.data_path,
                    split=config.valid_split,
                    token=config.token,
                )

    num_classes = None
    if config.trainer == "pair_class":
        classes = train_data.features[config.target_column].names
        # label2id = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)

        if num_classes < 2:
            raise ValueError("Invalid number of classes. Must be greater than 1.")

        if config.valid_split is not None:
            num_classes_valid = len(valid_data.unique(config.target_column))
            if num_classes_valid != num_classes:
                raise ValueError(
                    f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {num_classes_valid}"
                )

    if config.logging_steps == -1:
        logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 25:
            logging_steps = 25
        config.logging_steps = logging_steps
    else:
        logging_steps = config.logging_steps

    logger.info(f"Logging steps: {logging_steps}")

    train_data = utils.process_columns(train_data, config)
    logger.info(f"Train data: {train_data}")
    if config.valid_split is not None:
        valid_data = utils.process_columns(valid_data, config)
        logger.info(f"Valid data: {valid_data}")

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        evaluation_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to=config.log,
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

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    if config.valid_split is not None:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    callbacks_to_use.extend([UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()])

    model = SentenceTransformer(
        config.model,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=config.token,
        model_kwargs={
            "ignore_mismatched_sizes": True,
        },
    )

    loss_mapping = {
        "pair": MultipleNegativesRankingLoss,
        "pair_class": partial(
            SoftmaxLoss,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_classes,
        ),
        "pair_score": CoSENTLoss,
        "triplet": MultipleNegativesRankingLoss,
        "qa": MultipleNegativesRankingLoss,
    }

    evaluator = None
    if config.valid_split is not None:
        if config.trainer == "pair_score":
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=valid_data[config.sentence1_column],
                sentences2=valid_data[config.sentence2_column],
                scores=valid_data[config.target_column],
                name=config.valid_split,
            )
        elif config.trainer == "triplet":
            evaluator = TripletEvaluator(
                anchors=valid_data[config.sentence1_column],
                positives=valid_data[config.sentence2_column],
                negatives=valid_data[config.sentence3_column],
            )

    logger.info("Setting up training arguments...")
    args = SentenceTransformerTrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
    )

    logger.info("Setting up trainer...")
    trainer = SentenceTransformerTrainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        loss=loss_mapping[config.trainer],
        evaluator=evaluator,
    )
    trainer.remove_callback(PrinterCallback)
    logger.info("Starting training...")
    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)

    model_card = utils.create_model_card(config, trainer)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            remove_autotrain_data(config)
            save_training_params(config)
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name,
                repo_id=f"{config.username}/{config.project_name}",
                repo_type="model",
            )

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = SentenceTransformersParams(**training_config)
    train(_config)