import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, Optional
from datasets import Dataset

from autotrain import logger

print(">>> RUNNING NEW dataset.py FROM:", __file__)

def detect_model_type(model):
    """Detect model type from model class name."""
    model_class = type(model).__name__
    if 'Whisper' in model_class or 'Seq2Seq' in model_class:
        return "seq2seq"
    elif 'CTC' in model_class or 'Wav2Vec' in model_class or 'Hubert' in model_class:
        return "ctc"
    else:
        return "generic"

def safe_tokenize_text(processor, text, max_seq_length=128):
    """
    Simplified tokenization that works for all models.
    """
    
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'encode'):
        try:
            input_ids = processor.tokenizer.encode(
                text,
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            return input_ids.squeeze(0)
        except Exception as e:
            logger.warning(f"Whisper tokenizer failed: {e}")
    
    
    if hasattr(processor, "tokenizer"):
        try:
            return processor.tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.squeeze(0)
        except Exception as e:
            logger.warning(f"Processor tokenizer failed: {e}")
    
   
    try:
        return processor(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
    except Exception as e:
        logger.warning(f"Processor direct failed: {e}")
    
   
    logger.warning(f"All tokenization methods failed for text: {text[:50]}...")
    return torch.tensor([0] * min(max_seq_length, 10), dtype=torch.long)

class AutomaticSpeechRecognitionDataset:
    """
    Universal ASR Dataset that works with all models (Whisper, Wav2Vec2, Hubert, etc.).
    """
    def __init__(
        self,
        data: Dataset,
        processor: Any,
        model: Any,
        audio_column: str = "audio",
        text_column: str = "transcription",
        max_duration: float = 30.0,
        sampling_rate: int = 16000,
        model_type: str = None,
    ):
        self._data = data
        self.processor = processor
        self.model = model
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.max_seq_length = 128
        
        
        if model_type is not None:
            self.model_type = model_type
        else:
            self.model_type = detect_model_type(model)
        
        logger.info(f"Universal Dataset initialized with model_type: {self.model_type}")
        logger.info(f"Processor type: {type(processor).__name__}")
        logger.info(f"Model type: {type(model).__name__}")
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        """
        try:
            item = self._data[idx]
            
            
            audio_path = item[self.audio_column]
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")

            
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
           
            duration = len(audio) / sr
            if duration > self.max_duration:
                logger.warning(f"Audio duration {duration:.2f}s exceeds max_duration {self.max_duration}s, truncating")
                max_samples = int(self.max_duration * self.sampling_rate)
                audio = audio[:max_samples]
            
           
            if self.model_type == 'seq2seq':
                
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=False,  
                        truncation=True,
                    )
                    input_features = inputs.input_features[0]
                    
                    target_length = 3000
                    if input_features.shape[1] < target_length:
                        padding = torch.zeros(80, target_length - input_features.shape[1])
                        input_features = torch.cat([input_features, padding], dim=1)
                    elif input_features.shape[1] > target_length:
                        input_features = input_features[:, :target_length]
                except Exception as e:
                    logger.warning(f"Seq2Seq audio processing failed: {e}")
                    
                    input_features = torch.tensor(audio, dtype=torch.float32)
                    
            elif self.model_type == 'ctc':
                
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=False,  
                        truncation=True,
                    )
                    input_features = inputs.input_values[0]
                        
                except Exception as e:
                    logger.warning(f"CTC audio processing failed: {e}")
                    
                    input_features = torch.tensor(audio, dtype=torch.float32)
                    
            else:
                
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=False,  
                        truncation=True,
                    )
                    if hasattr(inputs, 'input_features'):
                        input_features = inputs.input_features[0]
                    elif hasattr(inputs, 'input_values'):
                        input_features = inputs.input_values[0]
                    else:
                        input_features = torch.tensor(audio, dtype=torch.float32)
                            
                except Exception as e:
                    logger.warning(f"Generic audio processing failed: {e}")
                    input_features = torch.tensor(audio, dtype=torch.float32)
            
            
            text = item[self.text_column]
            if not text or not isinstance(text, str):
                text = " "  
            
            
            labels = safe_tokenize_text(self.processor, text, self.max_seq_length)
            
           
            if self.model_type == 'seq2seq':
                return {
                    "input_features": input_features,
                    "labels": labels,
                }
            else:
                return {
                    "input_values": input_features,
                    "labels": labels,
                }

        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            
            if self.model_type == 'seq2seq':
                dummy_audio = torch.zeros(80, 100, dtype=torch.float32)  
            else:
                dummy_audio = torch.zeros(1000, dtype=torch.float32) 
            dummy_labels = torch.tensor([0], dtype=torch.long)
            
            if self.model_type == 'seq2seq':
                return {
                    "input_features": dummy_audio,
                    "labels": dummy_labels,
                }
            else:
                return {
                    "input_values": dummy_audio,
                    "labels": dummy_labels,
                }

def load_life_app_dataset(data_path):
    """Utility to load LiFE App dataset."""
    from datasets import load_from_disk
    return load_from_disk(data_path)