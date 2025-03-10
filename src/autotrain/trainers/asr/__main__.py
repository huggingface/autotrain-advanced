import logging
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from autotrain.trainers.asr.params import WhisperTrainingParams
from autotrain.trainers.asr.utils import (
    compute_metrics,
    load_audio_dataset,
    prepare_dataset,
)

logger = logging.getLogger(__name__)

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
        from peft import get_peft_model
        
        model = get_peft_model(model, params.get_lora_config())
        model.print_trainable_parameters()
    
    # Load and prepare dataset
    logger.info("Loading dataset")
    train_dataset = load_audio_dataset(
        dataset_path=dataset_path,
        audio_column=audio_column,
        text_column=text_column,
        split="train",
    )
    
    eval_dataset = load_audio_dataset(
        dataset_path=dataset_path,
        audio_column=audio_column,
        text_column=text_column,
        split="validation",
    )
    
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
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
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
    # Example usage
    params = WhisperTrainingParams()
    train_whisper(
        params=params,
        dataset_path="your_dataset_path",
        output_dir="output",
        audio_column="audio",
        text_column="text",
    ) 