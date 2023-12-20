from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


RESERVED_COLUMNS = ["autotrain_id", "autotrain_label"]


@dataclass
class TabularBinaryClassificationPreprocessor:
    train_data: pd.DataFrame
    label_column: str
    username: str
    project_name: str
    token: str
    id_column: Optional[str] = None
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

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
        train_df.loc[:, "autotrain_id"] = train_df[self.id_column] if self.id_column else list(range(len(train_df)))
        train_df.loc[:, "autotrain_label"] = train_df[self.label_column]
        valid_df.loc[:, "autotrain_id"] = valid_df[self.id_column] if self.id_column else list(range(len(valid_df)))
        valid_df.loc[:, "autotrain_label"] = valid_df[self.label_column]

        # drop id_column and label_column
        drop_cols = [self.id_column, self.label_column] if self.id_column else [self.label_column]
        train_df = train_df.drop(columns=drop_cols)
        valid_df = valid_df.drop(columns=drop_cols)
        return train_df, valid_df

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)
        train_df = Dataset.from_pandas(train_df)
        valid_df = Dataset.from_pandas(valid_df)
        if self.local:
            dataset = DatasetDict(
                {
                    "train": train_df,
                    "validation": valid_df,
                }
            )
            dataset.save_to_disk(f"{self.project_name}/autotrain-data")
        else:
            train_df.push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                split="train",
                private=True,
                token=self.token,
            )
            valid_df.push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                split="validation",
                private=True,
                token=self.token,
            )
        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


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


@dataclass
class TabularMultiLabelClassificationPreprocessor:
    train_data: pd.DataFrame
    label_column: List[str]
    username: str
    project_name: str
    id_column: Optional[str] = None
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    token: Optional[str] = None
    local: Optional[bool] = False

    def __post_init__(self):
        # check if id_column and label_column are in train_data
        if self.id_column is not None:
            if self.id_column not in self.train_data.columns:
                raise ValueError(f"{self.id_column} not in train data")

        for label in self.label_column:
            if label not in self.train_data.columns:
                raise ValueError(f"{label} not in train data")

        # check if id_column and label_column are in valid_data
        if self.valid_data is not None:
            if self.id_column is not None:
                if self.id_column not in self.valid_data.columns:
                    raise ValueError(f"{self.id_column} not in valid data")
            for label in self.label_column:
                if label not in self.valid_data.columns:
                    raise ValueError(f"{label} not in valid data")

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
        train_df.loc[:, "autotrain_id"] = train_df[self.id_column] if self.id_column else list(range(len(train_df)))

        for label in range(len(self.label_column)):
            train_df.loc[:, f"autotrain_label_{label}"] = train_df[self.label_column[label]]

        valid_df.loc[:, "autotrain_id"] = valid_df[self.id_column] if self.id_column else list(range(len(valid_df)))

        for label in range(len(self.label_column)):
            valid_df.loc[:, f"autotrain_label_{label}"] = valid_df[self.label_column[label]]

        # drop id_column and label_column
        drop_cols = [self.id_column] + self.label_column if self.id_column else self.label_column
        train_df = train_df.drop(columns=drop_cols)
        valid_df = valid_df.drop(columns=drop_cols)
        return train_df, valid_df

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)
        train_df = Dataset.from_pandas(train_df)
        valid_df = Dataset.from_pandas(valid_df)
        if self.local:
            dataset = DatasetDict(
                {
                    "train": train_df,
                    "validation": valid_df,
                }
            )
            dataset.save_to_disk(f"{self.project_name}/autotrain-data")
        else:
            train_df.push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                split="train",
                private=True,
                token=self.token,
            )
            valid_df.push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                split="validation",
                private=True,
                token=self.token,
            )
        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


class TabularMultiColumnRegressionPreprocessor(TabularMultiLabelClassificationPreprocessor):
    pass
