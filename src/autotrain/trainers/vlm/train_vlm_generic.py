from functools import partial

from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import ALLOW_REMOTE_CODE
from autotrain.trainers.vlm import utils


def collate_fn(examples, config, processor):
    prompts = ["answer " + example[config.prompt_text_column] for example in examples]
    labels = [example[config.text_column] for example in examples]
    images = [example[config.image_column].convert("RGB") for example in examples]
    tokens = processor(
        text=prompts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        tokenize_newline_separately=False,
    )
    return tokens


def train(config):
    valid_data = None
    if config.data_path == f"{config.project_name}/autotrain-data":
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

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    if config.trainer == "captioning":
        config.prompt_text_column = "caption"

    processor = AutoProcessor.from_pretrained(config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)

    args = TrainingArguments(**training_args)
    model = utils.get_model(config)

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(config)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
    )

    col_fn = partial(collate_fn, config=config, processor=processor)

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data if valid_data is not None else None,
        data_collator=col_fn,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    utils.post_training_steps(config, trainer)
