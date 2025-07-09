import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ALLOWED_AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "m4a")

@dataclass
class AutomaticSpeechRecognitionPreprocessor:
    """
    A class used to preprocess audio data for automatic speech recognition (ASR) tasks.

    Attributes
    ----------
    train_data : str
        Path to the training data directory (should contain audio/ folder and CSV).
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], optional
        Path to the validation data directory, by default None.
    test_size : Optional[float], optional
        Proportion of the dataset to include in the validation split, by default 0.2.
    seed : Optional[int], optional
        Random seed for reproducibility, by default 42.
    local : Optional[bool], optional
        Whether to save the dataset locally or push to the Hugging Face Hub, by default False.

    Methods
    -------
    __post_init__():
        Validates the structure and contents of the training and validation data directories.
    split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Splits the dataframe into training and validation sets.
    prepare() -> str:
        Prepares the dataset for training and either saves it locally or pushes it to the Hugging Face Hub.
    load_life_app_dataset_from_disk(data_path):
        Loads the LiFE App dataset from disk for ASR training.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # Validate train data directory
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        audio_dir = None
        csv_file = None
        for root, dirs, files in os.walk(self.train_data):
            for d in dirs:
                if d.lower() == 'audio':
                    audio_dir = os.path.join(root, d)
            for f in files:
                if f.endswith('.csv'):
                    csv_file = os.path.join(root, f)
        if not audio_dir or not csv_file:
            raise ValueError(f"{self.train_data} should contain an audio/ folder and a CSV file.")

        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(ALLOWED_AUDIO_EXTENSIONS)]
        if len(audio_files) < 5:
            raise ValueError(f"{audio_dir} should contain at least 5 audio files.")

        df = pd.read_csv(csv_file)
        if "autotrain_audio" in df.columns and "autotrain_transcription" in df.columns:
            df = df.rename(columns={
                "autotrain_audio": "audio",
                "autotrain_transcription": "transcription"
            })
        if 'audio' not in df.columns or 'transcription' not in df.columns:
            raise ValueError("CSV must have 'audio' and 'transcription' columns.")

        # Check that all audio files in CSV exist
        missing_files = []
        for _, row in df.iterrows():
            audio_path = os.path.join(audio_dir, str(row['audio']))
            if not os.path.exists(audio_path):
                missing_files.append(str(row['audio']))
        if missing_files:
            raise ValueError(f"The following audio files referenced in CSV are missing: {missing_files[:5]}...")

        self.audio_dir = audio_dir
        self.csv_file = csv_file
        self.df = df

        if self.valid_data:
            # Validate validation data directory
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            valid_audio_dir = None
            valid_csv_file = None
            for root, dirs, files in os.walk(self.valid_data):
                for d in dirs:
                    if d.lower() == 'audio':
                        valid_audio_dir = os.path.join(root, d)
                for f in files:
                    if f.endswith('.csv'):
                        valid_csv_file = os.path.join(root, f)
            if not valid_audio_dir or not valid_csv_file:
                raise ValueError(f"{self.valid_data} should contain an audio/ folder and a CSV file.")

            valid_audio_files = [f for f in os.listdir(valid_audio_dir) if f.endswith(ALLOWED_AUDIO_EXTENSIONS)]
            if len(valid_audio_files) < 5:
                raise ValueError(f"{valid_audio_dir} should contain at least 5 audio files.")

            valid_df = pd.read_csv(valid_csv_file)
            if "autotrain_audio" in valid_df.columns and "autotrain_transcription" in valid_df.columns:
                valid_df = valid_df.rename(columns={
                    "autotrain_audio": "audio",
                    "autotrain_transcription": "transcription"
                })
            if 'audio' not in valid_df.columns or 'transcription' not in valid_df.columns:
                raise ValueError("Validation CSV must have 'audio' and 'transcription' columns.")

            missing_valid_files = []
            for _, row in valid_df.iterrows():
                audio_path = os.path.join(valid_audio_dir, str(row['audio']))
                if not os.path.exists(audio_path):
                    missing_valid_files.append(str(row['audio']))
            if missing_valid_files:
                raise ValueError(f"The following validation audio files referenced in CSV are missing: {missing_valid_files[:5]}...")

            self.valid_audio_dir = valid_audio_dir
            self.valid_csv_file = valid_csv_file
            self.valid_df = valid_df

    def split(self, df):
        """
        Split a DataFrame into train and validation sets.
        """
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        """
        Prepare the dataset for training: copy, split, and format as needed.
        """
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            # Copy train and validation data as-is
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            dataset = load_dataset(
                "csv", 
                data_files={
                    "train": os.path.join(data_dir, "train", "data.csv"),
                    "validation": os.path.join(data_dir, "validation", "data.csv")
                }
            )
            dataset = dataset.rename_columns({"audio": "audio", "transcription": "transcription"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        else:
            # Split training data into train/validation
            train_df, valid_df = self.split(self.df)
            for split_name, split_df in zip(["train", "validation"], [train_df, valid_df]):
                split_audio_dir = os.path.join(data_dir, split_name, "audio")
                os.makedirs(split_audio_dir, exist_ok=True)
                split_csv = os.path.join(data_dir, split_name, "data.csv")
                new_rows = []
                for _, row in split_df.iterrows():
                    src_audio = os.path.join(self.audio_dir, str(row["audio"]))
                    dst_audio = os.path.join(split_audio_dir, os.path.basename(src_audio))
                    shutil.copy(src_audio, dst_audio)
                    new_rows.append({
                        "audio": os.path.join("audio", os.path.basename(src_audio)),
                        "transcription": row["transcription"]
                    })
                pd.DataFrame(new_rows).to_csv(split_csv, index=False)
            dataset = load_dataset(
                "csv", 
                data_files={
                    "train": os.path.join(data_dir, "train", "data.csv"),
                    "validation": os.path.join(data_dir, "validation", "data.csv")
                }
            )
            dataset = dataset.rename_columns({"audio": "audio", "transcription": "transcription"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"

    def load_life_app_dataset_from_disk(self, data_path):
        """
        Load a LiFE App dataset from disk for ASR training. (Extend as needed.)
        """