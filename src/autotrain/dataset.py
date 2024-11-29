import io
import os
import uuid
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from autotrain.preprocessor.tabular import (
    TabularBinaryClassificationPreprocessor,
    TabularMultiClassClassificationPreprocessor,
    TabularMultiColumnRegressionPreprocessor,
    TabularMultiLabelClassificationPreprocessor,
    TabularSingleColumnRegressionPreprocessor,
)
from autotrain.preprocessor.text import (
    LLMPreprocessor,
    SentenceTransformersPreprocessor,
    Seq2SeqPreprocessor,
    TextBinaryClassificationPreprocessor,
    TextExtractiveQuestionAnsweringPreprocessor,
    TextMultiClassClassificationPreprocessor,
    TextSingleColumnRegressionPreprocessor,
    TextTokenClassificationPreprocessor,
)
from autotrain.preprocessor.vision import (
    ImageClassificationPreprocessor,
    ImageRegressionPreprocessor,
    ObjectDetectionPreprocessor,
)
from autotrain.preprocessor.vlm import VLMPreprocessor


def remove_non_image_files(folder):
    """
    Remove non-image files from a specified folder and its subfolders.

    This function iterates through all files in the given folder and its subfolders,
    and removes any file that does not have an allowed image file extension. The allowed
    extensions are: .jpg, .jpeg, .png, .JPG, .JPEG, .PNG, and .jsonl.

    Args:
        folder (str): The path to the folder from which non-image files should be removed.

    Returns:
        None
    """
    # Define allowed image file extensions
    allowed_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".jsonl"}

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
class AutoTrainImageClassificationDataset:
    """
    A class to handle image classification datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

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

    def prepare(self):
        valid_dir = None
        if not isinstance(self.train_data, str):
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
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

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
class AutoTrainObjectDetectionDataset:
    """
    A dataset class for AutoTrain object detection tasks.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to be used for validation. Default is None.
        local (bool): Flag indicating if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

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
        self.task = "image_object_detection"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        if not isinstance(self.train_data, str):
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
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ObjectDetectionPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainVLMDataset:
    """
    A class to handle dataset for AutoTrain Vision-Language Model (VLM) task.

    Attributes:
    -----------
    train_data : str
        Path to the training data or a file-like object containing the training data.
    token : str
        Authentication token for accessing the dataset.
    project_name : str
        Name of the project.
    username : str
        Username of the project owner.
    column_mapping : Dict[str, str]
        Mapping of columns in the dataset.
    valid_data : Optional[str], default=None
        Path to the validation data or a file-like object containing the validation data.
    percent_valid : Optional[float], default=None
        Percentage of the training data to be used for validation if `valid_data` is not provided.
    local : bool, default=False
        Flag indicating whether the dataset is stored locally.

    Methods:
    --------
    __str__() -> str:
        Returns a string representation of the dataset.

    __post_init__():
        Initializes the dataset and sets default values for validation data percentage.

    prepare():
        Prepares the dataset for training by extracting and processing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    column_mapping: Dict[str, str]
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "vlm"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        if not isinstance(self.train_data, str):
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
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = VLMPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
            column_mapping=self.column_mapping,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainImageRegressionDataset:
    """
    AutoTrainImageRegressionDataset is a class designed for handling image regression datasets in the AutoTrain framework.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to be used for validation if valid_data is not provided. Default is None.
        local (bool): Flag indicating if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset information.

        __post_init__():
            Initializes the task attribute and sets the percent_valid attribute based on the presence of valid_data.

        prepare():
            Prepares the dataset for training by extracting and organizing the data, and returns a preprocessor object.
    """

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
        self.task = "image_single_column_regression"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        if not isinstance(self.train_data, str):
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
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ImageRegressionPreprocessor(
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
    """
    AutoTrainDataset class for handling various types of datasets and preprocessing tasks.

    Attributes:
        train_data (List[str]): List of file paths or DataFrames for training data.
        task (str): The type of task to perform (e.g., "text_binary_classification").
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (Optional[str]): Username of the project owner. Defaults to None.
        column_mapping (Optional[Dict[str, str]]): Mapping of column names. Defaults to None.
        valid_data (Optional[List[str]]): List of file paths or DataFrames for validation data. Defaults to None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Defaults to None.
        convert_to_class_label (Optional[bool]): Whether to convert labels to class labels. Defaults to False.
        local (bool): Whether the data is local. Defaults to False.
        ext (Optional[str]): File extension of the data files. Defaults to "csv".

    Methods:
        __str__(): Returns a string representation of the dataset.
        __post_init__(): Initializes validation data and preprocesses the data.
        _preprocess_data(): Preprocesses the training and validation data.
        num_samples(): Returns the total number of samples in the dataset.
        prepare(): Prepares the dataset for the specified task using the appropriate preprocessor.
    """

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
                convert_to_class_label=self.convert_to_class_label,
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

        elif self.task == "sentence_transformers":
            sentence1_column = self.column_mapping["sentence1"]
            sentence2_column = self.column_mapping["sentence2"]
            sentence3_column = self.column_mapping.get("sentence3")
            target_column = self.column_mapping.get("target")

            preprocessor = SentenceTransformersPreprocessor(
                train_data=self.train_df,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
                sentence1_column=sentence1_column,
                sentence2_column=sentence2_column,
                sentence3_column=sentence3_column,
                target_column=target_column,
                convert_to_class_label=self.convert_to_class_label,
            )
            return preprocessor.prepare()

        elif self.task == "text_extractive_question_answering":
            text_column = self.column_mapping["text"]
            question_column = self.column_mapping["question"]
            answer_column = self.column_mapping["answer"]
            preprocessor = TextExtractiveQuestionAnsweringPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                question_column=question_column,
                answer_column=answer_column,
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
