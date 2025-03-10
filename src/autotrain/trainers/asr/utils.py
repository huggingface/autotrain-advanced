from typing import Dict, Optional, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import WhisperProcessor

def load_audio_dataset(
    dataset_path: str,
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "train",
) -> Dataset:
    """Load and prepare audio dataset for Whisper training.
    
    This function loads an audio dataset from the Hugging Face Hub or local path and validates
    that it contains the required columns for ASR training.
    
    Args:
        dataset_path (str): Path or name of the dataset on the Hugging Face Hub.
        audio_column (str, optional): Name of the column containing audio data. Defaults to "audio".
        text_column (str, optional): Name of the column containing text transcriptions. Defaults to "text".
        split (str, optional): Dataset split to load ("train" or "validation"). Defaults to "train".
        
    Returns:
        Dataset: The loaded dataset with validated columns.
        
    Raises:
        ValueError: If the dataset cannot be loaded or if required columns are missing.
    """
    dataset = load_dataset(dataset_path, split=split)
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
    
    return dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

def compute_metrics(pred):
    """Compute Word Error Rate (WER) metric for ASR evaluation.
    
    This function calculates the Word Error Rate between predicted transcriptions
    and reference texts. It handles special token IDs and uses the evaluate library's
    WER implementation.
    
    Args:
        pred: Prediction object containing predictions and label_ids.
        
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