from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


RESERVED_COLUMNS = ["autotrain_id", "autotrain_label"]


@dataclass
class TabularBinaryClassificationPreprocessor:
    """
    A preprocessor class for tabular binary classification tasks.

    Attributes:
        train_data (pd.DataFrame): The training data.
        label_column (str): The name of the label column in the training data.
        username (str): The username for the Hugging Face Hub.
        project_name (str): The name of the project.
        token (str): The authentication token for the Hugging Face Hub.
        id_column (Optional[str]): The name of the ID column in the training data. Default is None.
        valid_data (Optional[pd.DataFrame]): The validation data. Default is None.
        test_size (Optional[float]): The proportion of the dataset to include in the validation split. Default is 0.2.
        seed (Optional[int]): The random seed for splitting the data. Default is 42.
        local (Optional[bool]): Whether to save the dataset locally or push to the Hugging Face Hub. Default is False.

    Methods:
        __post_init__(): Validates the presence of required columns in the training and validation data.
        split(): Splits the training data into training and validation sets if validation data is not provided.
        prepare_columns(train_df, valid_df): Prepares the columns by adding 'autotrain_id' and 'autotrain_label', and drops the original ID and label columns.
        prepare(): Prepares the dataset by splitting, processing columns, and saving or pushing the dataset to the Hugging Face Hub.
    """

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
    """
    TabularMultiLabelClassificationPreprocessor is a class for preprocessing tabular data for multi-label classification tasks.

    Attributes:
        train_data (pd.DataFrame): The training data.
        label_column (List[str]): List of columns to be used as labels.
        username (str): The username for the Hugging Face Hub.
        project_name (str): The project name for the Hugging Face Hub.
        id_column (Optional[str]): The column to be used as an identifier. Defaults to None.
        valid_data (Optional[pd.DataFrame]): The validation data. Defaults to None.
        test_size (Optional[float]): The proportion of the dataset to include in the validation split. Defaults to 0.2.
        seed (Optional[int]): The random seed for splitting the data. Defaults to 42.
        token (Optional[str]): The token for authentication with the Hugging Face Hub. Defaults to None.
        local (Optional[bool]): Whether to save the dataset locally or push to the Hugging Face Hub. Defaults to False.

    Methods:
        __post_init__(): Validates the presence of id_column and label_column in train_data and valid_data, and checks for reserved column names.
        split(): Splits the train_data into training and validation sets if valid_data is not provided.
        prepare_columns(train_df, valid_df): Prepares the columns by adding autotrain_id and autotrain_label columns, and drops the original id_column and label_column.
        prepare(): Prepares the dataset by splitting the data, preparing the columns, and converting to Hugging Face Dataset format. Saves the dataset locally or pushes to the Hugging Face Hub.
    """

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
