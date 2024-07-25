import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Features, Image, Value, load_dataset
from sklearn.model_selection import train_test_split


ALLOWED_EXTENSIONS = ("jpeg", "png", "jpg", "JPG", "JPEG", "PNG")


@dataclass
class VLMPreprocessor:
    train_data: str
    username: str
    project_name: str
    token: str
    column_mapping: dict
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def _process_metadata(self, data_path):
        metadata = pd.read_json(os.path.join(data_path, "metadata.jsonl"), lines=True)
        # make sure that the metadata.jsonl file contains the required columns: file_name, objects
        if "file_name" not in metadata.columns:
            raise ValueError(f"{data_path}/metadata.jsonl should contain 'file_name' column.")

        col_names = list(self.column_mapping.values())

        for col in col_names:
            if col not in metadata.columns:
                raise ValueError(f"{data_path}/metadata.jsonl should contain '{col}' column.")

        return metadata

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # check if self.train_data contains at least 5 image files in jpeg, png or jpg format only
        train_image_files = [f for f in os.listdir(self.train_data) if f.endswith(ALLOWED_EXTENSIONS)]
        if len(train_image_files) < 5:
            raise ValueError(f"{self.train_data} should contain at least 5 jpeg, png or jpg files.")

        # check if self.train_data contains a metadata.jsonl file
        if "metadata.jsonl" not in os.listdir(self.train_data):
            raise ValueError(f"{self.train_data} should contain a metadata.jsonl file.")

        # Check if valid data path exists
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # check if self.valid_data contains at least 5 image files in jpeg, png or jpg format only
            valid_image_files = [f for f in os.listdir(self.valid_data) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(valid_image_files) < 5:
                raise ValueError(f"{self.valid_data} should contain at least 5 jpeg, png or jpg files.")

            # check if self.valid_data contains a metadata.jsonl file
            if "metadata.jsonl" not in os.listdir(self.valid_data):
                raise ValueError(f"{self.valid_data} should contain a metadata.jsonl file.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            train_metadata = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            features = Features(
                {
                    "image": Image(),
                }
            )
            for _, col_map in self.column_mapping.items():
                features[col_map] = Value(dtype="string")

            dataset = load_dataset("imagefolder", data_dir=data_dir, features=features)

            rename_dict = {
                "image": "autotrain_image",
            }
            for col, col_map in self.column_mapping.items():
                if col == "text_column":
                    rename_dict[col_map] = "autotrain_text"
                elif col == "prompt_text_column":
                    rename_dict[col_map] = "autotrain_prompt"

            dataset = dataset.rename_columns(rename_dict)

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        else:
            metadata = pd.read_json(os.path.join(self.train_data, "metadata.jsonl"), lines=True)
            train_df, valid_df = self.split(metadata)

            # create train and validation folders
            os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)

            # move images to train and validation folders
            for row in train_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "train", row[1]["file_name"]),
                )

            for row in valid_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "validation", row[1]["file_name"]),
                )

            # save metadata.jsonl file to train and validation folders
            train_df.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_df.to_json(os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True)

            train_metadata = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            features = Features(
                {
                    "image": Image(),
                }
            )
            for _, col_map in self.column_mapping.items():
                features[col_map] = Value(dtype="string")

            dataset = load_dataset("imagefolder", data_dir=data_dir, features=features)

            rename_dict = {
                "image": "autotrain_image",
            }
            for col, col_map in self.column_mapping.items():
                if col == "text_column":
                    rename_dict[col_map] = "autotrain_text"
                elif col == "prompt_text_column":
                    rename_dict[col_map] = "autotrain_prompt"

            dataset = dataset.rename_columns(rename_dict)

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
