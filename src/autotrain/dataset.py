from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class Dataset:
    train_data: str
    task: str
    token: str
    project_name: str
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
        train_df = pd.read_csv(self.train_data[0])
        logger.info(train_df)
