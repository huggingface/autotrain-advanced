import os
import logging
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from typing import Dict, Optional, Any
from autotrain.preprocessor.base import AutoTrainPreprocessor
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AutomaticSpeechRecognitionPreprocessor(AutoTrainPreprocessor):
    """
    A preprocessor class for automatic speech recognition tasks.

    Attributes:
        train_data (pd.DataFrame): The training data.
        valid_data (Optional[pd.DataFrame]): The validation data.
        project_name (str): Name of the project.
        username (str): Hugging Face username.
        token (str): Hugging Face token.
        column_mapping (Dict[str, str]): Mapping of column names.
        test_size (float): Proportion of data to use for validation.
        seed (int): Random seed for splitting.
        local (bool): Whether to save data locally or push to hub.
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        project_name: str,
        username: str,
        token: str,
        column_mapping: Optional[Dict[str, str]] = None,
        valid_data: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        seed: int = 42,
        local: bool = False,
    ):
        # Set default column mapping to match user's CSV structure
        if column_mapping is None:
            column_mapping = {
                "audio": "audio",  # Path to audio files
                "transcription": "transcription",  # Text transcription
                "duration": "duration"  # Audio duration
            }
            
        super().__init__(
            train_data=train_data,
            token=token,
            project_name=project_name,
            username=username,
            column_mapping=column_mapping,
        )
        self.valid_data = valid_data
        self.test_size = test_size
        self.seed = seed
        self.local = local

        # Extract column names from mapping
        self.audio_column = self.column_mapping.get("audio", "audio")
        self.text_column = self.column_mapping.get("transcription", "transcription")
        self.duration_column = self.column_mapping.get("duration", "duration")

        # Validate columns exist
        if self.audio_column not in self.train_data.columns:
            raise ValueError(f"Audio column '{self.audio_column}' not found in training data")
        if self.text_column not in self.train_data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in training data")
        if self.duration_column not in self.train_data.columns:
            raise ValueError(f"Duration column '{self.duration_column}' not found in training data")
        
        if self.valid_data is not None:
            if self.audio_column not in self.valid_data.columns:
                raise ValueError(f"Audio column '{self.audio_column}' not found in validation data")
            if self.text_column not in self.valid_data.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in validation data")
            if self.duration_column not in self.valid_data.columns:
                raise ValueError(f"Duration column '{self.duration_column}' not found in validation data")

    def split(self):
        """Split data into train and validation sets."""
        if self.valid_data is not None:
            return self.train_data, self.valid_data
        
        train_df, valid_df = train_test_split(
            self.train_data,
            test_size=self.test_size,
            random_state=self.seed
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare_columns(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        """Prepare columns for training."""
        # Rename columns to standard names that the trainer expects
        train_df = train_df.rename(columns={
            self.audio_column: "autotrain_audio",
            self.text_column: "autotrain_transcription"
        })
        valid_df = valid_df.rename(columns={
            self.audio_column: "autotrain_audio",
            self.text_column: "autotrain_transcription"
        })
        return train_df, valid_df

    def prepare(self):
        """Prepare the dataset for training."""
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)

        # Save locally
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": valid_dataset
        })
        
        # Create output directory
        output_dir = os.path.join(self.project_name, "autotrain-data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset to disk
        dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved to {output_dir}")
        
        return output_dir

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the data for ASR training."""
        logger.info(f"Available columns in DataFrame: {df.columns.tolist()}")
        logger.info(f"Column mapping: {self.column_mapping}")
        
        # Process audio files
        logger.info(f"Processing audio files from column: {self.audio_column}")
        
        processed_audio = []
        for audio_path in df[self.audio_column]:
            processed_audio.append(self._process_audio(audio_path))
        
        # Clean text
        logger.info(f"Cleaning text from column: {self.text_column}")
        
        processed_text = df[self.text_column].apply(lambda x: x.strip().lower())
        
        # Create processed dataframe
        processed_df = pd.DataFrame({
            'audio': processed_audio,
            'transcription': processed_text
        })
        
        return processed_df

    def _process_audio(self, audio_path: str) -> np.ndarray:
        """Process an audio file."""
        logger.info(f"Processing audio file: {audio_path}")
        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=16000)
            logger.info(f"Successfully processed audio file: {audio_path}")
            return audio
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {str(e)}")
            raise

    def _create_datasets(self, processed_df: pd.DataFrame) -> DatasetDict:
        """Create train/validation/test datasets."""
        # Create train dataset
        train_dataset = Dataset.from_pandas(processed_df)
        
        # Create dataset dictionary
        datasets = DatasetDict({
            'train': train_dataset
        })
        
        return datasets

    def _save_datasets(self, datasets: DatasetDict):
        """Save datasets to disk."""
        save_path = f"{self.project_name}/autotrain-data"
        os.makedirs(save_path, exist_ok=True)
        
        logger.info("Saving training dataset...")
        datasets.save_to_disk(save_path)
        logger.info("Training dataset saved successfully") 