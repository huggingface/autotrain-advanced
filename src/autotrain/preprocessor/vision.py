import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from autotrain import logger


ALLOWED_EXTENSIONS = ("jpeg", "png", "jpg", "JPG", "JPEG", "PNG")


@dataclass
class ImageClassificationPreprocessor:
    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # Check if train data path contains at least 2 folders
        subfolders = [f.path for f in os.scandir(self.train_data) if f.is_dir()]
        # list subfolders
        logger.info(f"🚀 Subfolders: {subfolders}")
        if len(subfolders) < 2:
            raise ValueError(f"{self.train_data} should contain at least 2 subfolders.")

        # Check if each subfolder contains at least 5 image files in jpeg, png or jpg format only
        for subfolder in subfolders:
            image_files = [f for f in os.listdir(subfolder) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(image_files) < 5:
                raise ValueError(f"{subfolder} should contain at least 5 jpeg, png or jpg files.")
            # Check if there are no other files except image files in the subfolder
            if len(image_files) != len(os.listdir(subfolder)):
                raise ValueError(f"{subfolder} should not contain any other files except image files.")

            # Check if there are no subfolders inside subfolders
            subfolders_in_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir()]
            if len(subfolders_in_subfolder) > 0:
                raise ValueError(f"{subfolder} should not contain any subfolders.")

        if self.valid_data:
            # Check if valid data path exists
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # Check if valid data path contains at least 2 folders
            subfolders = [f.path for f in os.scandir(self.valid_data) if f.is_dir()]

            # make sure that the subfolders in train and valid data are the same
            train_subfolders = set(os.path.basename(f.path) for f in os.scandir(self.train_data) if f.is_dir())
            valid_subfolders = set(os.path.basename(f.path) for f in os.scandir(self.valid_data) if f.is_dir())
            if train_subfolders != valid_subfolders:
                raise ValueError(f"{self.valid_data} should have the same subfolders as {self.train_data}.")

            if len(subfolders) < 2:
                raise ValueError(f"{self.valid_data} should contain at least 2 subfolders.")

            # Check if each subfolder contains at least 5 image files in jpeg, png or jpg format only
            for subfolder in subfolders:
                image_files = [f for f in os.listdir(subfolder) if f.endswith(ALLOWED_EXTENSIONS)]
                if len(image_files) < 5:
                    raise ValueError(f"{subfolder} should contain at least 5 jpeg, png or jpg files.")

                # Check if there are no other files except image files in the subfolder
                if len(image_files) != len(os.listdir(subfolder)):
                    raise ValueError(f"{subfolder} should not contain any other files except image files.")

                # Check if there are no subfolders inside subfolders
                subfolders_in_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir()]
                if len(subfolders_in_subfolder) > 0:
                    raise ValueError(f"{subfolder} should not contain any subfolders.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df["subfolder"],
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

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns({"image": "autotrain_image", "label": "autotrain_label"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        else:
            subfolders = [f.path for f in os.scandir(self.train_data) if f.is_dir()]

            image_filenames = []
            subfolder_names = []

            for subfolder in subfolders:
                for filename in os.listdir(subfolder):
                    if filename.endswith(("jpeg", "png", "jpg")):
                        image_filenames.append(filename)
                        subfolder_names.append(os.path.basename(subfolder))

            df = pd.DataFrame({"image_filename": image_filenames, "subfolder": subfolder_names})
            train_df, valid_df = self.split(df)

            for row in train_df.itertuples():
                os.makedirs(os.path.join(data_dir, "train", row.subfolder), exist_ok=True)
                shutil.copy(
                    os.path.join(self.train_data, row.subfolder, row.image_filename),
                    os.path.join(data_dir, "train", row.subfolder, row.image_filename),
                )

            for row in valid_df.itertuples():
                os.makedirs(os.path.join(data_dir, "validation", row.subfolder), exist_ok=True)
                shutil.copy(
                    os.path.join(self.train_data, row.subfolder, row.image_filename),
                    os.path.join(data_dir, "validation", row.subfolder, row.image_filename),
                )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns({"image": "autotrain_image", "label": "autotrain_label"})
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
