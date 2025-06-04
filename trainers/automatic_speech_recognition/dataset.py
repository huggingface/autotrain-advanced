import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, Optional
from datasets import Dataset
from transformers import ProcessorMixin

from autotrain import logger
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCTC


class AutomaticSpeechRecognitionDataset:
    """
    Dataset for automatic speech recognition.
    """
    def __init__(
        self,
        data: Dataset,
        processor: Any,
        config: Any,
        audio_column: str = "audio",
        text_column: str = "transcription",
        max_duration: float = 30.0,
        sampling_rate: int = 16000,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Dataset containing audio and text data
            processor: Audio processor for feature extraction
            config: Training configuration
            audio_column: Name of the column containing audio data
            text_column: Name of the column containing text data
            max_duration: Maximum duration of audio in seconds
            sampling_rate: Target sampling rate for audio
        """
        self._data = data
        self.processor = processor
        self.config = config
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        
        # Determine model type
        self.model_type = self._get_model_type()
        logger.info(f"Detected model type: {self.model_type}")
        
        # Verify audio files exist and can be loaded
        self._verify_audio_files()
        
    def _get_model_type(self) -> str:
        """
        Determine the type of ASR model being used.
        
        Returns:
            str: Model type ('seq2seq', 'ctc', or 'generic')
        """
        try:
            # Try loading as Seq2Seq model first
            AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model,
                token=self.config.token if self.config.token else None,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            return 'seq2seq'
        except Exception:
            try:
                # Try loading as CTC model
                AutoModelForCTC.from_pretrained(
                    self.config.model,
                    token=self.config.token if self.config.token else None,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
                return 'ctc'
            except Exception:
                # If both fail, return generic
                return 'generic'
        
    def _verify_audio_files(self):
        """
        Verify that all audio files exist and can be loaded.
        """
        invalid_files = []
        for idx, item in enumerate(self._data):
            try:
                audio_path = item[self.audio_column]
                if not os.path.exists(audio_path):
                    invalid_files.append((idx, audio_path, "File not found"))
                    continue
                    
                # Try to load the audio file
                try:
                    audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
                    duration = len(audio) / sr
                    if duration > self.max_duration:
                        invalid_files.append((idx, audio_path, f"Duration {duration:.2f}s exceeds max_duration {self.max_duration}s"))
                except Exception as e:
                    invalid_files.append((idx, audio_path, f"Error loading audio: {str(e)}"))
                    
            except Exception as e:
                invalid_files.append((idx, "Unknown", f"Error processing item: {str(e)}"))
                
        if invalid_files:
            error_msg = "Found invalid audio files:\n"
            for idx, path, reason in invalid_files[:5]:  # Show first 5 errors
                error_msg += f"Row {idx}: {path} - {reason}\n"
            if len(invalid_files) > 5:
                error_msg += f"... and {len(invalid_files) - 5} more files"
            raise ValueError(error_msg)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item to get

        Returns:
            dict: Processed item with input features and labels
        """
        try:
            item = self._data[idx]
            
            # Get audio path and verify it exists
            audio_path = item[self.audio_column]
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")

            # Load and process audio
            try:
                audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            except Exception as e:
                raise ValueError(f"Error loading audio file {audio_path}: {str(e)}")
            
            # Check duration
            duration = len(audio) / sr
            if duration > self.max_duration:
                raise ValueError(f"Audio duration {duration:.2f}s exceeds max_duration {self.max_duration}s")
            
            # Process audio with processor based on model type
            try:
                if self.model_type == 'seq2seq':
                    # For Seq2Seq models (Whisper, MMS, etc.)
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                        max_length=int(self.max_duration * self.sampling_rate),
                        truncation=True,
                    )
                    input_features = inputs.input_features[0]
                elif self.model_type == 'ctc':
                    # For CTC models (Wav2Vec2, Hubert, etc.)
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                        max_length=int(self.max_duration * self.sampling_rate),
                        truncation=True,
                    )
                    input_features = inputs.input_values[0]
                else:
                    # For generic models, try both approaches
                    try:
                        inputs = self.processor(
                            audio,
                            sampling_rate=self.sampling_rate,
                            return_tensors="pt",
                            padding=True,
                            max_length=int(self.max_duration * self.sampling_rate),
                            truncation=True,
                        )
                        if hasattr(inputs, 'input_features'):
                            input_features = inputs.input_features[0]
                        else:
                            input_features = inputs.input_values[0]
                    except Exception as e:
                        raise ValueError(f"Error processing audio with processor: {str(e)}")
            except Exception as e:
                raise ValueError(f"Error processing audio with processor: {str(e)}")
            
            # Get text and verify it exists
            text = item[self.text_column]
            if not text or not isinstance(text, str):
                raise ValueError(f"Invalid text in row {idx}: {text}")

            # Process text with processor
            try:
                with self.processor.as_target_processor():
                    labels = self.processor(text).input_ids
            except Exception as e:
                raise ValueError(f"Error processing text with processor: {str(e)}")
            
            # Return features based on model type
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
            raise 