from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


RESERVED_COLUMNS = ["autotrain_id", "autotrain_label"]


@dataclass
class TabularBinaryClassificationPreprocessor:
    train_data: pd.DataFrame
    label_column: str
    username: str
    project_name: str
    id_column: Optional[str] = None
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42

    def __post_init__(self):
        # check if id_column and label_column are in train_data
        if self.id_column is not None:
            if self.id_column not in self.train_data.columns:
                raise ValueError(f"{self.id_column} not in train data")

        if self.label_column not in self.train_data.columns:
            raise ValueError(f"{self.label_column} not in train data")

        # check if id_column and label_column are in valid_data
        if self.valid_data is not None:
            if self.id_column is not None:
                if self.id_column not in self.valid_data.columns:
                    raise ValueError(f"{self.id_column} not in valid data")
            if self.label_column not in self.valid_data.columns:
                raise ValueError(f"{self.label_column} not in valid data")

        # make sure no reserved columns are in train_data or valid_data
        for column in RESERVED_COLUMNS:
            if column in self.train_data.columns:
                raise ValueError(f"{column} is a reserved column name")
            if self.valid_data is not None:
                if column in self.valid_data.columns:
                    raise ValueError(f"{column} is a reserved column name")

    def split(self):
        if self.valid_data is not None:
            return self.train_data, self.valid_data
        else:
            train_df, valid_df = train_test_split(
                self.train_data,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=self.train_data[self.label_column],
            )
            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            return train_df, valid_df

    def prepare_columns(self, train_df, valid_df):
        train_df.loc[:, "autotrain_id"] = train_df[self.id_column]
        train_df.loc[:, "autotrain_label"] = train_df[self.label_column]
        valid_df.loc[:, "autotrain_id"] = valid_df[self.id_column]
        valid_df.loc[:, "autotrain_label"] = valid_df[self.label_column]

        # drop id_column and label_column
        train_df = train_df.drop(columns=[self.id_column, self.label_column])
        valid_df = valid_df.drop(columns=[self.id_column, self.label_column])
        return train_df, valid_df

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)
        train_df = Dataset.from_pandas(train_df)
        valid_df = Dataset.from_pandas(valid_df)
        train_df.push_to_hub(f"{self.username}/autotrain-data-{self.project_name}", split="train", private=True)
        valid_df.push_to_hub(f"{self.username}/autotrain-data-{self.project_name}", split="validation", private=True)
        return train_df, valid_df


class TabularMultiClassClassificationPreprocessor(TabularBinaryClassificationPreprocessor):
    pass


class TabularSingleColumnRegressionPreprocessor(TabularBinaryClassificationPreprocessor):
    def split(self):
        if self.valid_data is not None:
            return self.train_data, self.valid_data
        else:
            train_df, valid_df = train_test_split(
                self.train_data,
                test_size=self.test_size,
                random_state=self.seed,
            )
            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            return train_df, valid_df
