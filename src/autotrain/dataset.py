from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    train_data: str
    task: str
    token: str
    project_name: str
    column_mapping: Optional[str] = None
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None

    def __post_init__(self):
        if self.valid_data is None and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data is not None and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data is not None:
            self.percent_valid = 0.0

        self.stratified_split_tasks = [
            "binary_classification",
            "multi_class_classification",
            "natural_language_inference",
            "image_binary_classification",
            "image_multi_class_classification",
            "tabular_binary_classification",
            "tabular_multi_class_classification",
            "tabular_multi_label_classification",
        ]

        self.random_split_tasks = [
            "extractive_question_answering",
            "summarization",
            "entity_extraction",
            "single_column_regression",
            "speech_recognition",
            "image_single_column_regression",
            "tabular_single_column_regression",
        ]

        self.no_split_tasks = [
            "dreambooth",
        ]

    def prepare(self):
        logger.info(self.train_data)
        train_df = []
        for file in self.train_data:
            train_df.append(pd.read_csv(file))
        train_df = pd.concat(train_df)

        valid_df = None
        if self.valid_data is not None:
            valid_df = []
            for file in self.valid_data:
                valid_df.append(pd.read_csv(file))
            valid_df = pd.concat(valid_df)

        # apply column mapping

        # if valid_df is None, then we need to split the train_df

        if self.task in self.stratified_split_tasks:
            train_df, valid_df = self.stratified_split(train_df, valid_df)
