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
    dataset = load_dataset(dataset_path, dataset_config, split=split)
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Failed to load dataset from {dataset_path}")
    
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