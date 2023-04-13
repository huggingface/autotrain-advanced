from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from loguru import logger

from autotrain.preprocessor.dreambooth import DreamboothPreprocessor
from autotrain.preprocessor.tabular import (
    TabularBinaryClassificationPreprocessor,
    TabularMultiClassClassificationPreprocessor,
    TabularSingleColumnRegressionPreprocessor,
)
from autotrain.preprocessor.text import (
    TextBinaryClassificationPreprocessor,
    TextMultiClassClassificationPreprocessor,
    TextSingleColumnRegressionPreprocessor,
)


@dataclass
class AutoTrainDreamboothDataset:
    num_concepts: int
    concept_images: List[List[str]]
    concept_names: List[str]
    token: str
    project_name: str
    username: str

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        return info

    def __post_init__(self):
        self.task = "dreambooth"
        logger.info(self.__str__())

    @property
    def num_samples(self):
        return sum([len(concept) for concept in self.concept_images])

    def prepare(self):
        preprocessor = DreamboothPreprocessor(
            num_concepts=self.num_concepts,
            concept_images=self.concept_images,
            concept_names=self.concept_names,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
        )
        preprocessor.prepare()


@dataclass
class AutoTrainImageClassificationDataset:
    train_data: str
    train_csv: str
    token: str
    project_name: str
    username: str
    column_mapping: Optional[str] = None
    valid_data: Optional[str] = None
    valid_csv: Optional[str] = None
    percent_valid: Optional[float] = None

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        info += f"Column mapping: {self.column_mapping}\n"
        return info

    def __post_init__(self):
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0
        logger.info(self.__str__())

    def _unzip_files(self):
        pass


@dataclass
class AutoTrainDataset:
    train_data: str
    task: str
    token: str
    project_name: str
    username: str
    column_mapping: Optional[str] = None
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        info += f"Column mapping: {self.column_mapping}\n"
        return info

    def __post_init__(self):
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0
        logger.info(self.__str__())

    @property
    def num_samples(self):
        train_df = []
        for file in self.train_data:
            if isinstance(file, pd.DataFrame):
                train_df.append(file)
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
                    valid_df.append(pd.read_csv(file))
            if len(valid_df) > 1:
                valid_df = pd.concat(valid_df)
            else:
                valid_df = valid_df[0]

        logger.info(train_df.head())
        if valid_df is not None:
            logger.info(valid_df.head())

        return len(train_df) + len(valid_df) if valid_df is not None else len(train_df)

    def prepare(self):
        logger.info(self.train_data)
        train_df = []
        for file in self.train_data:
            if isinstance(file, pd.DataFrame):
                train_df.append(file)
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
                    valid_df.append(pd.read_csv(file))
            if len(valid_df) > 1:
                valid_df = pd.concat(valid_df)
            else:
                valid_df = valid_df[0]

        logger.info(train_df.head())
        if valid_df is not None:
            logger.info(valid_df.head())

        if self.task == "text_binary_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextBinaryClassificationPreprocessor(
                train_data=train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
            )
            preprocessor.prepare()

        elif self.task == "text_multi_class_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextMultiClassClassificationPreprocessor(
                train_data=train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
            )
            preprocessor.prepare()

        elif self.task == "text_single_column_regression":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextSingleColumnRegressionPreprocessor(
                train_data=train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
            )
            preprocessor.prepare()
        elif self.task == "tabular_binary_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularBinaryClassificationPreprocessor(
                train_data=train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
            )
            preprocessor.prepare()
        elif self.task == "tabular_multi_class_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiClassClassificationPreprocessor(
                train_data=train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
            )
            preprocessor.prepare()
        elif self.task == "tabular_single_column_regression":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularSingleColumnRegressionPreprocessor(
                train_data=train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
            )
            preprocessor.prepare()
        else:
            raise ValueError(f"Task {self.task} not supported")
