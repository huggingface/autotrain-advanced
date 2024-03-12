import io
import os
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from autotrain import logger
from autotrain.preprocessor.dreambooth import DreamboothPreprocessor
from autotrain.preprocessor.tabular import (
    TabularBinaryClassificationPreprocessor,
    TabularMultiClassClassificationPreprocessor,
    TabularMultiColumnRegressionPreprocessor,
    TabularMultiLabelClassificationPreprocessor,
    TabularSingleColumnRegressionPreprocessor,
)
from autotrain.preprocessor.text import (
    LLMPreprocessor,
    Seq2SeqPreprocessor,
    TextBinaryClassificationPreprocessor,
    TextMultiClassClassificationPreprocessor,
    TextSingleColumnRegressionPreprocessor,
    TextTokenClassificationPreprocessor,
)
from autotrain.preprocessor.vision import ImageClassificationPreprocessor


def remove_non_image_files(folder):
    # Define allowed image file extensions
    allowed_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Get the file extension
            file_extension = os.path.splitext(file)[1]

            # If the file extension is not in the allowed list, remove the file
            if file_extension.lower() not in allowed_extensions:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

        # Recursively call the function on each subfolder
        for subfolder in dirs:
            remove_non_image_files(os.path.join(root, subfolder))


@dataclass
class AutoTrainDreamboothDataset:
    concept_images: List[Any]
    concept_name: str
    token: str
    project_name: str
    username: Optional[str] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        return info

    def __post_init__(self):
        self.task = "dreambooth"
        logger.info(self.__str__())

    @property
    def num_samples(self):
        return len(self.concept_images)

    def prepare(self):
        preprocessor = DreamboothPreprocessor(
            concept_images=self.concept_images,
            concept_name=self.concept_name,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainImageClassificationDataset:
    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "image_multi_class_classification"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0
        logger.info(self.__str__())

        self.num_files = self._count_files()

    @property
    def num_samples(self):
        return self.num_files

    def _count_files(self):
        num_files = 0
        zip_ref = zipfile.ZipFile(self.train_data, "r")
        for _ in zip_ref.namelist():
            num_files += 1
        if self.valid_data:
            zip_ref = zipfile.ZipFile(self.valid_data, "r")
            for _ in zip_ref.namelist():
                num_files += 1
        return num_files

    def prepare(self):
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

        random_uuid = uuid.uuid4()
        train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
        os.makedirs(train_dir, exist_ok=True)
        self.train_data.seek(0)
        content = self.train_data.read()
        bytes_io = io.BytesIO(content)

        zip_ref = zipfile.ZipFile(bytes_io, "r")
        zip_ref.extractall(train_dir)
        # remove the __MACOSX directory
        macosx_dir = os.path.join(train_dir, "__MACOSX")
        if os.path.exists(macosx_dir):
            os.system(f"rm -rf {macosx_dir}")
        remove_non_image_files(train_dir)

        valid_dir = None
        if self.valid_data:
            random_uuid = uuid.uuid4()
            valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(valid_dir, exist_ok=True)
            self.valid_data.seek(0)
            content = self.valid_data.read()
            bytes_io = io.BytesIO(content)
            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(valid_dir)
            # remove the __MACOSX directory
            macosx_dir = os.path.join(valid_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(valid_dir)

        preprocessor = ImageClassificationPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainDataset:
    train_data: List[str]
    task: str
    token: str
    project_name: str
    username: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    valid_data: Optional[List[str]] = None
    percent_valid: Optional[float] = None
    convert_to_class_label: Optional[bool] = False
    local: bool = False
    ext: Optional[str] = "csv"

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        info += f"Column mapping: {self.column_mapping}\n"
        return info

    def __post_init__(self):
        if self.valid_data is None:
            self.valid_data = []
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

        self.train_df, self.valid_df = self._preprocess_data()
        logger.info(self.__str__())

    def _preprocess_data(self):
        train_df = []
        for file in self.train_data:
            if isinstance(file, pd.DataFrame):
                train_df.append(file)
            else:
                if self.ext == "jsonl":
                    train_df.append(pd.read_json(file, lines=True))
                else:
                    train_df.append(pd.read_csv(file))
        if len(train_df) > 1:
            train_df = pd.concat(train_df)
        else:
            train_df = train_df[0]

        valid_df = None
        if len(self.valid_data) > 0:
            valid_df = []
            for file in self.valid_data:
                if isinstance(file, pd.DataFrame):
                    valid_df.append(file)
                else:
                    if self.ext == "jsonl":
                        valid_df.append(pd.read_json(file, lines=True))
                    else:
                        valid_df.append(pd.read_csv(file))
            if len(valid_df) > 1:
                valid_df = pd.concat(valid_df)
            else:
                valid_df = valid_df[0]
        return train_df, valid_df

    @property
    def num_samples(self):
        return len(self.train_df) + len(self.valid_df) if self.valid_df is not None else len(self.train_df)

    def prepare(self):
        if self.task == "text_binary_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextBinaryClassificationPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                convert_to_class_label=self.convert_to_class_label,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "text_multi_class_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextMultiClassClassificationPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                convert_to_class_label=self.convert_to_class_label,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "text_token_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextTokenClassificationPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "text_single_column_regression":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextSingleColumnRegressionPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "seq2seq":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = Seq2SeqPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "lm_training":
            text_column = self.column_mapping["text"]
            prompt_column = self.column_mapping.get("prompt")
            rejected_text_column = self.column_mapping.get("rejected_text")
            preprocessor = LLMPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                prompt_column=prompt_column,
                rejected_text_column=rejected_text_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "tabular_binary_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"][0]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularBinaryClassificationPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_multi_class_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"][0]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiClassClassificationPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_single_column_regression":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"][0]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularSingleColumnRegressionPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_multi_column_regression":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiColumnRegressionPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_multi_label_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiLabelClassificationPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        else:
            raise ValueError(f"Task {self.task} not supported")
