from typing import Dict, Optional, Union, List, Any

import numpy as np
import torch
import evaluate
import librosa
from datasets import Dataset, load_dataset
from transformers import WhisperProcessor
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class WhisperDataCollator:
    """
    Data collator for Whisper ASR training.
    
    This collator handles batching of input features and labels for Whisper training,
    ensuring proper padding and formatting.
    
    Args:
        processor (WhisperProcessor): The Whisper processor used for tokenization.
        padding (bool, optional): Whether to pad sequences. Defaults to True.
    """
    processor: WhisperProcessor
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input features and labels
        input_features = [torch.tensor(feature["input_features"]) if isinstance(feature["input_features"], list) 
                         else feature["input_features"] for feature in features]
        
        # Ensure all input features are tensors
        input_features = [feat if isinstance(feat, torch.Tensor) else torch.tensor(feat) for feat in input_features]
        
        # Get labels
        labels = [feature["labels"] for feature in features]
        
        # Convert input features to batch
        batch = {"input_features": torch.stack(input_features)}
        
        # Pad labels
        if self.padding:
            max_label_length = max(len(label) for label in labels)
            padded_labels = []
            
            for label in labels:
                padding_length = max_label_length - len(label)
                padded_label = label + [self.processor.tokenizer.pad_token_id] * padding_length
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.tensor(padded_labels)
        else:
            batch["labels"] = torch.tensor(labels)
        
        # Replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch["labels"] == self.processor.tokenizer.pad_token_id, -100
        )
        
        return batch

def load_audio_dataset(
    dataset_path: str,
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "train",
    dataset_config: Optional[str] = None,
) -> Dataset:
    """Load an audio dataset from the Hugging Face Hub or local path and validate
    that it contains the required columns for ASR training.
    
    Args:
        dataset_path (str): Path or name of the dataset on the Hugging Face Hub.
        audio_column (str, optional): Name of the column containing audio data. Defaults to "audio".
        text_column (str, optional): Name of the column containing text transcriptions. Defaults to "text".
        split (str, optional): Dataset split to load ("train" or "validation"). Defaults to "train".
        dataset_config (Optional[str], optional): Configuration name for the dataset. Defaults to None.
        
    Returns:
        Dataset: The loaded dataset with validated columns.
        
    Raises:
        ValueError: If the dataset cannot be loaded or if required columns are missing.
    """
    # Special handling for Mozilla Common Voice dataset
    if dataset_path == "mozilla-foundation/common_voice_11_0" and dataset_config is None:
        # Default to 'en' if no config is provided
        logger.warning("No configuration provided for Mozilla Common Voice dataset. Using 'mr' (Marathi) as default.")
        dataset_config = "mr"
    
    try:
        # Try to load the dataset with the provided configuration
        dataset = load_dataset(dataset_path, dataset_config, split=split)
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
    
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Failed to load dataset from {dataset_path}")
    
    # For Mozilla Common Voice, the text column is 'sentence' by default
    if dataset_path == "mozilla-foundation/common_voice_11_0" and text_column == "text":
        text_column = "sentence"
        logger.info("Using 'sentence' column for text in Mozilla Common Voice dataset")
    
    required_columns = {audio_column, text_column}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
        
    return dataset

def prepare_dataset(
    dataset: Dataset,
    processor: WhisperProcessor,
    audio_column: str = "audio",
    text_column: str = "text",
    max_duration_secs: float = 30.0,
    sampling_rate: int = 16000,
) -> Dataset:
    """Prepare dataset by processing audio and text for Whisper training.
    
    This function processes each example in the dataset by:
    1. Loading and resampling audio to the target sampling rate
    2. Truncating or padding audio to a fixed duration
    3. Converting audio to input features using the Whisper processor
    4. Processing text transcriptions into token IDs
    
    Args:
        dataset (Dataset): Input dataset containing audio and text.
        processor (WhisperProcessor): Whisper processor for feature extraction.
        audio_column (str, optional): Name of the column containing audio data. Defaults to "audio".
        text_column (str, optional): Name of the column containing text transcriptions. Defaults to "text".
        max_duration_secs (float, optional): Maximum duration of audio in seconds. Defaults to 30.0.
        sampling_rate (int, optional): Target audio sampling rate in Hz. Defaults to 16000.
        
    Returns:
        Dataset: Processed dataset with input_features and labels.
        
    Raises:
        ValueError: If audio data format is unexpected.
    """
    
    def prepare_example(example):
        # Load and resample audio if needed
        audio = example[audio_column]
        if isinstance(audio, dict):
            audio_array = audio["array"]
            curr_sampling_rate = audio["sampling_rate"]
        else:
            raise ValueError(f"Unexpected audio format in column {audio_column}")
            
        # Resample if necessary
        if curr_sampling_rate != sampling_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=curr_sampling_rate,
                target_sr=sampling_rate,
            )
            
        # Truncate or pad audio
        max_samples = int(max_duration_secs * sampling_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        elif len(audio_array) < max_samples:
            audio_array = np.pad(audio_array, (0, max_samples - len(audio_array)))
            
        # Process audio
        input_features = processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features[0]
        
        # Process text
        labels = processor(text=example[text_column]).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels,
        }
    
    # Process the dataset
    return dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

def compute_metrics(pred, processor=None):
    """Compute Word Error Rate (WER) metric for ASR evaluation.
    
    This function calculates the Word Error Rate between predicted transcriptions
    and reference texts. It handles special token IDs and uses the evaluate library's
    WER implementation.
    
    Args:
        pred: Prediction object containing predictions and label_ids.
        processor (WhisperProcessor, optional): The processor to use for decoding.
            If None, assumes a global processor is available. Defaults to None.
        
    Returns:
        dict: Dictionary containing the "wer" (Word Error Rate) metric.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with processor.tokenizer.pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = evaluate.load("wer")
    return {"wer": wer.compute(predictions=pred_str, references=label_str)}

MODEL_CARD = """---
language:
- {language}
license: apache-2.0
tags:
- autotrain
- automatic-speech-recognition
- asr{base_model}
datasets:
{dataset_tag}
widget:
- audio: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/speech.wav"
---

# {model_title}

This model is a fine-tuned version of {model_name} on {dataset_description}. It achieves the following results on the evaluation set:

{metrics_summary}

## Model description

This model was trained using AutoTrain on a speech recognition task.

- Task: {task}
- Language: {language}

## Intended uses & limitations

This model is intended for Automatic Speech Recognition (ASR) in {language} language.

## Training and evaluation data

{dataset_info}

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: {learning_rate}
- train_batch_size: {train_batch_size}
- eval_batch_size: {eval_batch_size}
- seed: {seed}
- optimizer: {optimizer} with betas=({beta1},{beta2}) and epsilon={epsilon}
- lr_scheduler_type: {lr_scheduler}
- num_epochs: {num_epochs}
{warmup_info}
- mixed_precision_training: {mixed_precision}

### Training results

{detailed_metrics}

### Framework versions

- Transformers: {transformers_version}
- Pytorch: {pytorch_version}
- Datasets: {datasets_version}
- PEFT: {peft_version}
"""


def create_asr_model_card(config, trainer, processor=None):
    """
    Generates a model card for ASR models.
    
    Args:
        config (WhisperTrainingParams): Configuration object containing training parameters.
        trainer (WhisperTrainer): Trainer object used for evaluating the model.
        processor (WhisperProcessor, optional): The processor used for tokenization.
    
    Returns:
        str: A formatted model card string containing dataset information, validation metrics, and model details.
    """
    # Get evaluation metrics if available
    metrics_summary = ""
    detailed_metrics = ""
    
    if config.valid_split is not None and hasattr(trainer, "evaluate"):
        try:
            eval_scores = trainer.evaluate()
            
            # Format metrics summary
            metrics_list = []
            for k, v in eval_scores.items():
                key = k[len('eval_'):] if k.startswith('eval_') else k
                metrics_list.append(f"{key.capitalize()}: {v:.4f}")
            
            metrics_summary = "\n".join(metrics_list)
            
            # Format detailed metrics for the training results section
            detailed_metrics = "Training Loss | Epoch | Step | Validation Loss | WER\n"
            detailed_metrics += "--- | --- | --- | --- | ---\n"
            
            # Get training loss from trainer history if available
            train_loss = "N/A"
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
                for entry in trainer.state.log_history:
                    if "loss" in entry:
                        train_loss = f"{entry['loss']:.4f}"
                        break
            
            # Get validation metrics
            val_loss = eval_scores.get("eval_loss", "N/A")
            if isinstance(val_loss, (int, float)):
                val_loss = f"{val_loss:.4f}"
                
            wer = eval_scores.get("eval_wer", "N/A")
            if isinstance(wer, (int, float)):
                wer = f"{wer:.4f}"
            
            # Add row to detailed metrics
            detailed_metrics += f"{train_loss} | {config.num_train_epochs} | {trainer.state.global_step} | {val_loss} | {wer}"
            
        except Exception as e:
            logger.warning(f"Failed to compute evaluation metrics: {e}")
            metrics_summary = "No validation metrics available"
            detailed_metrics = "No detailed training results available"
    else:
        metrics_summary = "No validation metrics available"
        detailed_metrics = "No detailed training results available"
    
    # Dataset information
    if hasattr(config, "data_path"):
        data_path = config.data_path
    else:
        data_path = "unknown"
        
    if hasattr(config, "project_name"):
        project_name = config.project_name
    else:
        project_name = "whisper-finetuned"
        
    if data_path == f"{project_name}/autotrain-data" or os.path.isdir(data_path):
        dataset_tag = "- custom_dataset"
        dataset_description = "a custom dataset"
    else:
        dataset_tag = f"- {data_path}"
        dataset_description = f"the {data_path} dataset"
    
    # Base model information
    model_name = getattr(config, "model_name", "unknown")
    if os.path.isdir(model_name):
        base_model = ""
    else:
        base_model = f"\nbase_model: {model_name}"
    
    # Create dataset info section
    dataset_info = f"The model was trained on {dataset_description}"
    if hasattr(config, "train_split"):
        dataset_info += f" using the '{config.train_split}' split for training"
    if hasattr(config, "valid_split") and config.valid_split:
        dataset_info += f" and the '{config.valid_split}' split for validation"
    dataset_info += "."
    
    # Get model title
    model_parts = model_name.split("/")
    model_base_name = model_parts[-1] if len(model_parts) > 1 else model_parts[0]
    language = getattr(config, "language", "unknown")
    model_title = f"{model_base_name} {language.upper()}"
    
    # Get optimizer info
    optimizer_type = getattr(config, "optimizer_type", "adamw")
    optimizer_name = optimizer_type.capitalize()
    if optimizer_name.lower() == "adamw":
        optimizer_name = "AdamW"
    elif optimizer_name.lower() == "adam":
        optimizer_name = "Adam"
    
    # Get warmup info
    warmup_info = ""
    if hasattr(config, "warmup_steps") and config.warmup_steps > 0:
        warmup_info = f"- lr_scheduler_warmup_steps: {config.warmup_steps}"
    elif hasattr(config, "lr_scheduler_warmup_ratio") and config.lr_scheduler_warmup_ratio > 0:
        warmup_info = f"- lr_scheduler_warmup_ratio: {config.lr_scheduler_warmup_ratio}"
    
    # Get mixed precision info
    mixed_precision = "No"
    if hasattr(config, "mixed_precision"):
        if config.mixed_precision == "fp16":
            mixed_precision = "Native AMP (fp16)"
        elif config.mixed_precision == "bf16":
            mixed_precision = "Native AMP (bf16)"
    
    # Try to get library versions
    try:
        import transformers
        transformers_version = transformers.__version__
    except:
        transformers_version = "N/A"
        
    try:
        import torch
        pytorch_version = torch.__version__
    except:
        pytorch_version = "N/A"
        
    try:
        import datasets
        datasets_version = datasets.__version__
    except:
        datasets_version = "N/A"
        
    try:
        import peft
        peft_version = peft.__version__
    except:
        peft_version = "N/A"
    
    # Get training parameters with defaults
    num_epochs = getattr(config, "num_train_epochs", 3)
    learning_rate = getattr(config, "learning_rate", 5e-5)
    train_batch_size = getattr(config, "per_device_train_batch_size", 8)
    eval_batch_size = getattr(config, "per_device_eval_batch_size", 8)
    seed = getattr(config, "seed", 42)
    beta1 = getattr(config, "optimizer_beta1", 0.9)
    beta2 = getattr(config, "optimizer_beta2", 0.999)
    epsilon = getattr(config, "optimizer_epsilon", 1e-8)
    lr_scheduler = getattr(config, "lr_scheduler_type", "linear")
    task = getattr(config, "task", "transcribe")
    
    # Create model card
    model_card = MODEL_CARD.format(
        model_title=model_title,
        dataset_tag=dataset_tag,
        metrics_summary=metrics_summary,
        base_model=base_model,
        language=language,
        task=task,
        model_name=model_name,
        dataset_description=dataset_description,
        dataset_info=dataset_info,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        optimizer=optimizer_name,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        lr_scheduler=lr_scheduler,
        warmup_info=warmup_info,
        mixed_precision=mixed_precision,
        detailed_metrics=detailed_metrics,
        transformers_version=transformers_version,
        pytorch_version=pytorch_version,
        datasets_version=datasets_version,
        peft_version=peft_version
    )
    
    return model_card 