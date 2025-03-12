import argparse
import logging
import os
import sys
import yaml
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset
from peft import get_peft_model, PeftModel
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    set_seed,
)

from autotrain.trainers.asr.params import WhisperTrainingParams
from autotrain.trainers.asr.utils import WhisperDataCollator, compute_metrics, load_audio_dataset, prepare_dataset
from autotrain.trainers.asr.whisper_peft import WhisperPeftModel

logger = logging.getLogger(__name__)

class WhisperPeftModel(PeftModel):
    """Custom PEFT model for Whisper that handles input formatting correctly."""
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that ensures only the expected inputs are passed to the model.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The model outputs
        """
        # Filter out unexpected inputs
        whisper_kwargs = {}
        if "input_features" in kwargs:
            whisper_kwargs["input_features"] = kwargs["input_features"]
        if "labels" in kwargs:
            whisper_kwargs["labels"] = kwargs["labels"]
        
        # Call the model with only the expected inputs
        return self.model.forward(**whisper_kwargs)

class WhisperTrainer(Seq2SeqTrainer):
    """Custom trainer for Whisper models that handles input formatting correctly."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss for Whisper models.
        
        This method ensures that the inputs are formatted correctly for the Whisper model.
        
        Args:
            model: The model to compute the loss for
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs along with the loss
            num_items_in_batch: Number of items in the batch (not used but needed for compatibility)
            
        Returns:
            The loss or a tuple of (loss, outputs) if return_outputs is True
        """
        # Extract only the inputs that Whisper expects
        whisper_inputs = {}
        if "input_features" in inputs:
            whisper_inputs["input_features"] = inputs["input_features"]
        if "labels" in inputs:
            whisper_inputs["labels"] = inputs["labels"]
        
        # Forward pass
        outputs = model(**whisper_inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Whisper ASR model")
    parser.add_argument(
        "--training_config",
        type=str,
        required=True,
        help="Path to the training config YAML file",
    )
    return parser.parse_args()

def train_whisper(
    params: WhisperTrainingParams,
    dataset_path: str,
    output_dir: str,
    audio_column: str = "audio",
    text_column: str = "text",
    dataset_config: Optional[str] = None,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_token: Optional[str] = None,
) -> None:
    """Main training function for Whisper ASR.
    
    Args:
        params (WhisperTrainingParams): Parameters for training.
        dataset_path (str): Path to the dataset.
        output_dir (str): Directory to save the model to.
        audio_column (str, optional): Name of the column containing audio data. Defaults to "audio".
        text_column (str, optional): Name of the column containing text transcriptions. Defaults to "text".
        dataset_config (Optional[str], optional): Configuration name for the dataset. Defaults to None.
        push_to_hub (bool, optional): Whether to push the model to the Hugging Face Hub. Defaults to False.
        hub_model_id (Optional[str], optional): Model ID on the Hugging Face Hub. Defaults to None.
        hub_token (Optional[str], optional): Hugging Face Hub token. Defaults to None.
    """
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is not available. Training will be slow on CPU.")
    else:
        logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    if params.seed is not None:
        set_seed(params.seed)
        logger.info(f"Random seed set to {params.seed}")
    
    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(params.model_name)
    model = model.to(device)
    processor = WhisperProcessor.from_pretrained(params.model_name)
    
    # Apply PEFT/LoRA if enabled
    if params.use_peft:
        logger.info("Applying PEFT/LoRA configuration")
        
        # Create a custom PEFT model with our WhisperPeftModel class
        model = get_peft_model(model, params.get_lora_config())
        # Replace the model with our custom WhisperPeftModel
        model.__class__ = WhisperPeftModel
        model.print_trainable_parameters()
    
    # Load and prepare dataset
    logger.info("Loading dataset")
    
    # Check available splits in the dataset
    try:
        # First try to load both train and validation splits
        train_dataset = load_audio_dataset(
            dataset_path=dataset_path,
            audio_column=audio_column,
            text_column=text_column,
            split="train",
            dataset_config=dataset_config,
        )
        
        try:
            eval_dataset = load_audio_dataset(
                dataset_path=dataset_path,
                audio_column=audio_column,
                text_column=text_column,
                split="validation",
                dataset_config=dataset_config,
            )
        except ValueError as e:
            # If validation split doesn't exist, create one from train
            logger.info("Validation split not found. Creating validation split from training data.")
            # Load the full dataset and split it
            full_dataset = load_dataset(dataset_path, dataset_config, split="train")
            splits = full_dataset.train_test_split(test_size=0.1)
            train_dataset = load_audio_dataset(
                dataset_path=dataset_path,
                audio_column=audio_column,
                text_column=text_column,
                split="train[:90%]",
                dataset_config=dataset_config,
            )
            # Create a custom validation dataset from the last 10% of training data
            eval_dataset = splits["test"]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
    
    logger.info("Preparing datasets")
    train_dataset = prepare_dataset(
        dataset=train_dataset,
        processor=processor,
        audio_column=audio_column,
        text_column=text_column,
        max_duration_secs=params.max_duration_secs,
        sampling_rate=params.sampling_rate,
    )
    
    eval_dataset = prepare_dataset(
        dataset=eval_dataset,
        processor=processor,
        audio_column=audio_column,
        text_column=text_column,
        max_duration_secs=params.max_duration_secs,
        sampling_rate=params.sampling_rate,
    )
    
    # Create data collator
    data_collator = WhisperDataCollator(processor=processor)
    
    # Calculate total training steps for warmup
    total_training_steps = (
        params.max_steps if params.max_steps is not None else 
        int(len(train_dataset) / (params.per_device_train_batch_size * params.gradient_accumulation_steps) * params.num_train_epochs)
    )
    
    # Calculate warmup steps based on ratio or absolute number
    warmup_steps = params.calculate_warmup_steps(total_training_steps)
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.learning_rate,
        num_train_epochs=params.num_train_epochs,
        max_steps=params.max_steps,
        warmup_steps=warmup_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=params.eval_steps,
        save_steps=params.save_steps,
        logging_steps=params.logging_steps,
        remove_unused_columns=True,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
        label_names=["labels"],
        # Add optimizer and scheduler parameters
        optim=params.optimizer_type.lower(),
        adam_beta1=params.optimizer_beta1,
        adam_beta2=params.optimizer_beta2,
        adam_epsilon=params.optimizer_epsilon,
        weight_decay=params.weight_decay,
        lr_scheduler_type=params.lr_scheduler_type.lower(),
        seed=params.seed,
        fp16=(params.mixed_precision == "fp16"),
        bf16=(params.mixed_precision == "bf16"),
        report_to=params.log if params.log != "none" else None,
    )
    
    # Initialize trainer with our custom WhisperTrainer
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    if params.use_peft:
        # Save PEFT/LoRA adapter separately
        model.save_pretrained(f"{output_dir}/adapter")

def main():
    """Main entry point for command-line execution."""
    args = parse_args()
    
    # Load training config from YAML file
    with open(args.training_config, "r") as f:
        training_config = yaml.safe_load(f)
    
    # Create WhisperTrainingParams from config
    params = WhisperTrainingParams(
        # Audio processing parameters
        sampling_rate=training_config.get("sampling_rate", 16000),
        max_duration_secs=training_config.get("max_duration_secs", 30.0),
        preprocessing_num_workers=training_config.get("preprocessing_num_workers", 4),
        
        # Model parameters
        model_name=training_config.get("model_name", "openai/whisper-small"),
        language=training_config.get("language", "en"),
        task=training_config.get("task", "transcribe"),
        
        # Training parameters
        learning_rate=training_config.get("learning_rate", 5e-5),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        eval_steps=training_config.get("eval_steps", 100),
        save_steps=training_config.get("save_steps", 500),
        logging_steps=training_config.get("logging_steps", 10),
        max_steps=training_config.get("max_steps"),
        warmup_steps=training_config.get("warmup_steps", 0),
        mixed_precision=training_config.get("mixed_precision", "fp16"),
        log=training_config.get("log", "tensorboard"),
        
        # PEFT/LoRA parameters
        use_peft=training_config.get("use_peft", True),
        lora_r=training_config.get("lora_r", 8),
        lora_alpha=training_config.get("lora_alpha", 32),
        lora_dropout=training_config.get("lora_dropout", 0.1),
        
        # Optimizer parameters
        optimizer_type=training_config.get("optimizer_type", "adamw"),
        optimizer_beta1=training_config.get("optimizer_beta1", 0.9),
        optimizer_beta2=training_config.get("optimizer_beta2", 0.999),
        optimizer_epsilon=training_config.get("optimizer_epsilon", 1e-8),
        weight_decay=training_config.get("weight_decay", 0.0),
        
        # Scheduler parameters
        lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
        lr_scheduler_warmup_ratio=training_config.get("lr_scheduler_warmup_ratio", 0.0),
        
        # Reproducibility
        seed=training_config.get("seed", 42),
    )
    
    # Start training
    train_whisper(
        params=params,
        dataset_path=training_config.get("dataset_path"),
        output_dir=training_config.get("output_dir"),
        audio_column=training_config.get("audio_column", "audio"),
        text_column=training_config.get("text_column", "text"),
        dataset_config=training_config.get("dataset_config"),
        push_to_hub=training_config.get("push_to_hub", False),
        hub_model_id=training_config.get("hub_model_id"),
        hub_token=training_config.get("hub_token"),
    )

if __name__ == "__main__":
    main() 