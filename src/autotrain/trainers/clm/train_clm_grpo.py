from datasets import load_dataset
from peft import LoraConfig
from transformers.trainer_callback import PrinterCallback
from trl import GRPOConfig, GRPOTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams


def reward_math_accuracy(completions, references, **kwargs):
    """
    Reward function for math accuracy. Rewards completions that match the reference answers.
    """
    rewards = []
    for completion, reference in zip(completions, references):
        # Extract numerical answer from completion and reference
        try:
            completion_num = float(''.join(filter(str.isdigit, completion.split('\n')[-1])))
            reference_num = float(''.join(filter(str.isdigit, reference.split('\n')[-1])))
            # Reward based on exact match
            reward = 1.0 if abs(completion_num - reference_num) < 1e-6 else -1.0
        except (ValueError, IndexError):
            reward = -1.0
        rewards.append(reward)
    return rewards


def train(config):
    logger.info("Starting GRPO training...")
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)
    train_data, valid_data = utils.process_input_data(config)
    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)

    # Configure GRPO specific parameters
    training_args["max_seq_length"] = config.block_size
    training_args["packing"] = True
    training_args["beta"] = 0.1  # GRPO hyperparameter
    training_args["gamma"] = 0.95  # GRPO hyperparameter
    training_args["kl_penalty"] = "kl"  # KL penalty type
    training_args["kl_threshold"] = 0.1  # KL threshold
    args = GRPOConfig(**training_args)

    model = utils.get_model(config, tokenizer)

    if config.peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=utils.get_target_modules(config),
        )

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(config)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
    )
    trainer = GRPOTrainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data if config.valid_split is not None else None,
        peft_config=peft_config if config.peft else None,
        tokenizer=tokenizer,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()
    utils.post_training_steps(config, trainer) 