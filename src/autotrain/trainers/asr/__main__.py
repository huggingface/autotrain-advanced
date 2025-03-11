import argparse
import logging
import yaml
from typing import Optional, Dict, Any, Union, List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import get_peft_model, PeftModel, PeftConfig

from autotrain.trainers.asr.params import WhisperTrainingParams
from autotrain.trainers.asr.utils import (
    WhisperDataCollator,
    compute_metrics,
    load_audio_dataset,
    prepare_dataset,
)

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
) -> None:
    """Main training function for Whisper ASR."""
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is not available. Training will be slow on CPU.")
    else:
        logger.info(f"Using device: {device}")
    
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
        )
        
        try:
            eval_dataset = load_audio_dataset(
                dataset_path=dataset_path,
                audio_column=audio_column,
                text_column=text_column,
                split="validation",
            )
        except ValueError as e:
            # If validation split doesn't exist, create one from train
            logger.info("Validation split not found. Creating validation split from training data.")
            # Load the full dataset and split it
            full_dataset = load_dataset(dataset_path, split="train")
            splits = full_dataset.train_test_split(test_size=0.1)
            train_dataset = load_audio_dataset(
                dataset_path=dataset_path,
                audio_column=audio_column,
                text_column=text_column,
                split="train[:90%]",
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
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.learning_rate,
        num_train_epochs=params.num_train_epochs,
        max_steps=params.max_steps,
        warmup_steps=params.warmup_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=params.eval_steps,
        save_steps=params.save_steps,
        logging_steps=params.logging_steps,
        remove_unused_columns=True,
        push_to_hub=False,
        label_names=["labels"],
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

if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r") as f:
        training_config = yaml.safe_load(f)
    
    # Extract parameters for WhisperTrainingParams
    model_params = {}
    for key, value in training_config.items():
        if key not in ["dataset_path", "output_dir", "audio_column", "text_column"]:
            model_params[key] = value
    
    config = WhisperTrainingParams(**model_params)
    
    train_whisper(
        params=config,
        dataset_path=training_config.get("dataset_path"),
        output_dir=training_config.get("output_dir", "output"),
        audio_column=training_config.get("audio_column", "audio"),
        text_column=training_config.get("text_column", "text"),
    ) 