import ast
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Sequence
from sklearn.model_selection import train_test_split

from autotrain import logger


RESERVED_COLUMNS = ["autotrain_text", "autotrain_label", "autotrain_question", "autotrain_answer"]
LLM_RESERVED_COLUMNS = [
    "autotrain_prompt",
    "autotrain_context",
    "autotrain_rejected_text",
    "autotrain_prompt_start",
]


@dataclass
class TextBinaryClassificationPreprocessor:
    """
    A preprocessor class for binary text classification tasks.

    Attributes:
        train_data (pd.DataFrame): The training data.
        text_column (str): The name of the column containing text data.
        label_column (str): The name of the column containing label data.
        username (str): The username for the Hugging Face Hub.
        project_name (str): The project name for saving datasets.
        token (str): The authentication token for the Hugging Face Hub.
        valid_data (Optional[pd.DataFrame]): The validation data. Defaults to None.
        test_size (Optional[float]): The proportion of the dataset to include in the validation split. Defaults to 0.2.
        seed (Optional[int]): The random seed for splitting the data. Defaults to 42.
        convert_to_class_label (Optional[bool]): Whether to convert labels to class labels. Defaults to False.
        local (Optional[bool]): Whether to save the dataset locally. Defaults to False.

    Methods:
        __post_init__(): Validates the presence of required columns in the dataframes and checks for reserved column names.
        split(): Splits the training data into training and validation sets if validation data is not provided.
        prepare_columns(train_df, valid_df): Prepares the columns for training and validation dataframes.
        prepare(): Prepares the datasets for training and validation, converts labels if required, and saves or uploads the datasets.
    """

    train_data: pd.DataFrame
    text_column: str
    label_column: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    convert_to_class_label: Optional[bool] = False
    local: Optional[bool] = False

    def __post_init__(self):
        # check if text_column and label_column are in train_data
        if self.text_column not in self.train_data.columns:
            raise ValueError(f"{self.text_column} not in train data")
        if self.label_column not in self.train_data.columns:
            raise ValueError(f"{self.label_column} not in train data")
        # check if text_column and label_column are in valid_data
        if self.valid_data is not None:
            if self.text_column not in self.valid_data.columns:
                raise ValueError(f"{self.text_column} not in valid data")
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
        train_df.loc[:, "autotrain_text"] = train_df[self.text_column]
        train_df.loc[:, "autotrain_label"] = train_df[self.label_column]
        valid_df.loc[:, "autotrain_text"] = valid_df[self.text_column]
        valid_df.loc[:, "autotrain_label"] = valid_df[self.label_column]

        # drop text_column and label_column
        train_df = train_df.drop(columns=[self.text_column, self.label_column])
        valid_df = valid_df.drop(columns=[self.text_column, self.label_column])
        return train_df, valid_df

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        train_df.loc[:, "autotrain_label"] = train_df["autotrain_label"].astype(str)
        valid_df.loc[:, "autotrain_label"] = valid_df["autotrain_label"].astype(str)

        label_names = sorted(set(train_df["autotrain_label"].unique().tolist()))

        train_df = Dataset.from_pandas(train_df)
        valid_df = Dataset.from_pandas(valid_df)

        if self.convert_to_class_label:
            train_df = train_df.cast_column("autotrain_label", ClassLabel(names=label_names))
            valid_df = valid_df.cast_column("autotrain_label", ClassLabel(names=label_names))

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


class TextMultiClassClassificationPreprocessor(TextBinaryClassificationPreprocessor):
    """
    TextMultiClassClassificationPreprocessor is a class for preprocessing text data for multi-class classification tasks.

    This class inherits from TextBinaryClassificationPreprocessor and is designed to handle scenarios where the text data
    needs to be classified into more than two categories.

    Methods:
        Inherits all methods from TextBinaryClassificationPreprocessor.

    Attributes:
        Inherits all attributes from TextBinaryClassificationPreprocessor.
    """

    pass


class TextSingleColumnRegressionPreprocessor(TextBinaryClassificationPreprocessor):
    """
    A preprocessor class for single-column regression tasks, inheriting from TextBinaryClassificationPreprocessor.

    Methods
    -------
    split():
        Splits the training data into training and validation sets. If validation data is already provided, it returns
        the training and validation data as is. Otherwise, it performs a train-test split on the training data.

    prepare():
        Prepares the training and validation datasets by splitting the data, preparing the columns, and converting
        them to Hugging Face Datasets. The datasets are then either saved locally or pushed to the Hugging Face Hub,
        depending on the `local` attribute.
    """

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


class TextTokenClassificationPreprocessor(TextBinaryClassificationPreprocessor):
    """
    A preprocessor class for text token classification tasks, inheriting from TextBinaryClassificationPreprocessor.

    Methods
    -------
    split():
        Splits the training data into training and validation sets. If validation data is already provided, it returns
        the training and validation data as is. Otherwise, it splits the training data based on the test size and seed.

    prepare():
        Prepares the training and validation data for token classification. This includes splitting the data, preparing
        columns, evaluating text and label columns, and converting them to datasets. The datasets are then either saved
        locally or pushed to the Hugging Face Hub based on the configuration.
    """

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

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)
        try:
            train_df.loc[:, "autotrain_text"] = train_df["autotrain_text"].apply(lambda x: ast.literal_eval(x))
            valid_df.loc[:, "autotrain_text"] = valid_df["autotrain_text"].apply(lambda x: ast.literal_eval(x))
        except ValueError:
            logger.warning("Unable to do ast.literal_eval on train_df['autotrain_text']")
            logger.warning("assuming autotrain_text is already a list")
        try:
            train_df.loc[:, "autotrain_label"] = train_df["autotrain_label"].apply(lambda x: ast.literal_eval(x))
            valid_df.loc[:, "autotrain_label"] = valid_df["autotrain_label"].apply(lambda x: ast.literal_eval(x))
        except ValueError:
            logger.warning("Unable to do ast.literal_eval on train_df['autotrain_label']")
            logger.warning("assuming autotrain_label is already a list")

        label_names_train = sorted(set(train_df["autotrain_label"].explode().unique().tolist()))
        label_names_valid = sorted(set(valid_df["autotrain_label"].explode().unique().tolist()))
        label_names = sorted(set(label_names_train + label_names_valid))

        train_df = Dataset.from_pandas(train_df)
        valid_df = Dataset.from_pandas(valid_df)

        if self.convert_to_class_label:
            train_df = train_df.cast_column("autotrain_label", Sequence(ClassLabel(names=label_names)))
            valid_df = valid_df.cast_column("autotrain_label", Sequence(ClassLabel(names=label_names)))

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


@dataclass
class LLMPreprocessor:
    """
    A class used to preprocess data for large language model (LLM) training.

    Attributes
    ----------
    train_data : pd.DataFrame
        The training data.
    username : str
        The username for the Hugging Face Hub.
    project_name : str
        The name of the project.
    token : str
        The token for authentication.
    valid_data : Optional[pd.DataFrame], optional
        The validation data, by default None.
    test_size : Optional[float], optional
        The size of the test split, by default 0.2.
    seed : Optional[int], optional
        The random seed, by default 42.
    text_column : Optional[str], optional
        The name of the text column, by default None.
    prompt_column : Optional[str], optional
        The name of the prompt column, by default None.
    rejected_text_column : Optional[str], optional
        The name of the rejected text column, by default None.
    local : Optional[bool], optional
        Whether to save the dataset locally, by default False.

    Methods
    -------
    __post_init__()
        Validates the provided columns and checks for reserved column names.
    split()
        Splits the data into training and validation sets.
    prepare_columns(train_df, valid_df)
        Prepares the columns for training and validation datasets.
    prepare()
        Prepares the datasets and pushes them to the Hugging Face Hub or saves them locally.
    """

    train_data: pd.DataFrame
    username: str
    project_name: str
    token: str
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    text_column: Optional[str] = None
    prompt_column: Optional[str] = None
    rejected_text_column: Optional[str] = None
    local: Optional[bool] = False

    def __post_init__(self):
        if self.text_column is None:
            raise ValueError("text_column must be provided")

        # check if text_column and rejected_text_column are in train_data
        if self.prompt_column is not None and self.prompt_column not in self.train_data.columns:
            self.prompt_column = None
        if self.rejected_text_column is not None and self.rejected_text_column not in self.train_data.columns:
            self.rejected_text_column = None

        # make sure no reserved columns are in train_data or valid_data
        for column in RESERVED_COLUMNS + LLM_RESERVED_COLUMNS:
            if column in self.train_data.columns:
                raise ValueError(f"{column} is a reserved column name")
            if self.valid_data is not None:
                if column in self.valid_data.columns:
                    raise ValueError(f"{column} is a reserved column name")

    def split(self):
        if self.valid_data is not None:
            return self.train_data, self.valid_data
        # no validation is done in llm training if validation data is not provided
        return self.train_data, self.train_data
        # else:
        #     train_df, valid_df = train_test_split(
        #         self.train_data,
        #         test_size=self.test_size,
        #         random_state=self.seed,
        #     )
        #     train_df = train_df.reset_index(drop=True)
        #     valid_df = valid_df.reset_index(drop=True)
        #     return train_df, valid_df

    def prepare_columns(self, train_df, valid_df):
        drop_cols = [self.text_column]
        train_df.loc[:, "autotrain_text"] = train_df[self.text_column]
        valid_df.loc[:, "autotrain_text"] = valid_df[self.text_column]
        if self.prompt_column is not None:
            drop_cols.append(self.prompt_column)
            train_df.loc[:, "autotrain_prompt"] = train_df[self.prompt_column]
            valid_df.loc[:, "autotrain_prompt"] = valid_df[self.prompt_column]
        if self.rejected_text_column is not None:
            drop_cols.append(self.rejected_text_column)
            train_df.loc[:, "autotrain_rejected_text"] = train_df[self.rejected_text_column]
            valid_df.loc[:, "autotrain_rejected_text"] = valid_df[self.rejected_text_column]

        # drop drop_cols
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


@dataclass
class Seq2SeqPreprocessor:
    """
    Seq2SeqPreprocessor is a class for preprocessing sequence-to-sequence training data.

    Attributes:
        train_data (pd.DataFrame): The training data.
        text_column (str): The name of the column containing the input text.
        label_column (str): The name of the column containing the labels.
        username (str): The username for pushing data to the hub.
        project_name (str): The name of the project.
        token (str): The token for authentication.
        valid_data (Optional[pd.DataFrame]): The validation data. Default is None.
        test_size (Optional[float]): The proportion of the dataset to include in the validation split. Default is 0.2.
        seed (Optional[int]): The random seed for splitting the data. Default is 42.
        local (Optional[bool]): Whether to save the dataset locally or push to the hub. Default is False.

    Methods:
        __post_init__(): Validates the presence of required columns in the training and validation data.
        split(): Splits the training data into training and validation sets if validation data is not provided.
        prepare_columns(train_df, valid_df): Prepares the columns for training and validation data.
        prepare(): Prepares the dataset for training by splitting, preparing columns, and converting to Dataset objects.
    """

    train_data: pd.DataFrame
    text_column: str
    label_column: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # check if text_column and label_column are in train_data
        if self.text_column not in self.train_data.columns:
            raise ValueError(f"{self.text_column} not in train data")
        if self.label_column not in self.train_data.columns:
            raise ValueError(f"{self.label_column} not in train data")
        # check if text_column and label_column are in valid_data
        if self.valid_data is not None:
            if self.text_column not in self.valid_data.columns:
                raise ValueError(f"{self.text_column} not in valid data")
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
            )
            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            return train_df, valid_df

    def prepare_columns(self, train_df, valid_df):
        train_df.loc[:, "autotrain_text"] = train_df[self.text_column]
        train_df.loc[:, "autotrain_label"] = train_df[self.label_column]
        valid_df.loc[:, "autotrain_text"] = valid_df[self.text_column]
        valid_df.loc[:, "autotrain_label"] = valid_df[self.label_column]

        # drop text_column and label_column
        train_df = train_df.drop(columns=[self.text_column, self.label_column])
        valid_df = valid_df.drop(columns=[self.text_column, self.label_column])
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


@dataclass
class SentenceTransformersPreprocessor:
    """
    A preprocessor class for preparing datasets for sentence transformers.

    Attributes:
        train_data (pd.DataFrame): The training data.
        username (str): The username for the Hugging Face Hub.
        project_name (str): The project name for the Hugging Face Hub.
        token (str): The token for authentication with the Hugging Face Hub.
        valid_data (Optional[pd.DataFrame]): The validation data. Default is None.
        test_size (Optional[float]): The proportion of the dataset to include in the validation split. Default is 0.2.
        seed (Optional[int]): The random seed for splitting the data. Default is 42.
        local (Optional[bool]): Whether to save the dataset locally or push to the Hugging Face Hub. Default is False.
        sentence1_column (Optional[str]): The name of the first sentence column. Default is "sentence1".
        sentence2_column (Optional[str]): The name of the second sentence column. Default is "sentence2".
        sentence3_column (Optional[str]): The name of the third sentence column. Default is "sentence3".
        target_column (Optional[str]): The name of the target column. Default is "target".
        convert_to_class_label (Optional[bool]): Whether to convert the target column to class labels. Default is False.

    Methods:
        __post_init__(): Ensures no reserved columns are in train_data or valid_data.
        split(): Splits the train_data into training and validation sets if valid_data is not provided.
        prepare_columns(train_df, valid_df): Prepares the columns for training and validation datasets.
        prepare(): Prepares the datasets and either saves them locally or pushes them to the Hugging Face Hub.
    """

    train_data: pd.DataFrame
    username: str
    project_name: str
    token: str
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False
    sentence1_column: Optional[str] = "sentence1"
    sentence2_column: Optional[str] = "sentence2"
    sentence3_column: Optional[str] = "sentence3"
    target_column: Optional[str] = "target"
    convert_to_class_label: Optional[bool] = False

    def __post_init__(self):
        # make sure no reserved columns are in train_data or valid_data
        for column in RESERVED_COLUMNS + LLM_RESERVED_COLUMNS:
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
            )
            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            return train_df, valid_df

    def prepare_columns(self, train_df, valid_df):
        train_df.loc[:, "autotrain_sentence1"] = train_df[self.sentence1_column]
        train_df.loc[:, "autotrain_sentence2"] = train_df[self.sentence2_column]
        valid_df.loc[:, "autotrain_sentence1"] = valid_df[self.sentence1_column]
        valid_df.loc[:, "autotrain_sentence2"] = valid_df[self.sentence2_column]
        keep_cols = ["autotrain_sentence1", "autotrain_sentence2"]

        if self.sentence3_column is not None:
            train_df.loc[:, "autotrain_sentence3"] = train_df[self.sentence3_column]
            valid_df.loc[:, "autotrain_sentence3"] = valid_df[self.sentence3_column]
            keep_cols.append("autotrain_sentence3")

        if self.target_column is not None:
            train_df.loc[:, "autotrain_target"] = train_df[self.target_column]
            valid_df.loc[:, "autotrain_target"] = valid_df[self.target_column]
            keep_cols.append("autotrain_target")

        train_df = train_df[keep_cols]
        valid_df = valid_df[keep_cols]

        return train_df, valid_df

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        if self.convert_to_class_label:
            label_names = sorted(set(train_df["autotrain_target"].unique().tolist()))

        train_df = Dataset.from_pandas(train_df)
        valid_df = Dataset.from_pandas(valid_df)

        if self.convert_to_class_label:
            train_df = train_df.cast_column("autotrain_target", ClassLabel(names=label_names))
            valid_df = valid_df.cast_column("autotrain_target", ClassLabel(names=label_names))

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


@dataclass
class TextExtractiveQuestionAnsweringPreprocessor:
    """
    Preprocessor for text extractive question answering tasks.

    Attributes:
        train_data (pd.DataFrame): The training data.
        text_column (str): The name of the text column in the data.
        question_column (str): The name of the question column in the data.
        answer_column (str): The name of the answer column in the data.
        username (str): The username for the Hugging Face Hub.
        project_name (str): The project name for the Hugging Face Hub.
        token (str): The token for authentication with the Hugging Face Hub.
        valid_data (Optional[pd.DataFrame]): The validation data. Default is None.
        test_size (Optional[float]): The proportion of the dataset to include in the validation split. Default is 0.2.
        seed (Optional[int]): The random seed for splitting the data. Default is 42.
        local (Optional[bool]): Whether to save the dataset locally or push to the Hugging Face Hub. Default is False.

    Methods:
        __post_init__(): Validates the columns in the training and validation data and converts the answer column to a dictionary.
        split(): Splits the training data into training and validation sets if validation data is not provided.
        prepare_columns(train_df, valid_df): Prepares the columns for training and validation data.
        prepare(): Prepares the dataset for training by splitting, preparing columns, and converting to Hugging Face Dataset format.
    """

    train_data: pd.DataFrame
    text_column: str
    question_column: str
    answer_column: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # check if text_column, question_column, and answer_column are in train_data
        if self.text_column not in self.train_data.columns:
            raise ValueError(f"{self.text_column} not in train data")
        if self.question_column not in self.train_data.columns:
            raise ValueError(f"{self.question_column} not in train data")
        if self.answer_column not in self.train_data.columns:
            raise ValueError(f"{self.answer_column} not in train data")
        # check if text_column, question_column, and answer_column are in valid_data
        if self.valid_data is not None:
            if self.text_column not in self.valid_data.columns:
                raise ValueError(f"{self.text_column} not in valid data")
            if self.question_column not in self.valid_data.columns:
                raise ValueError(f"{self.question_column} not in valid data")
            if self.answer_column not in self.valid_data.columns:
                raise ValueError(f"{self.answer_column} not in valid data")

        # make sure no reserved columns are in train_data or valid_data
        for column in RESERVED_COLUMNS:
            if column in self.train_data.columns:
                raise ValueError(f"{column} is a reserved column name")
            if self.valid_data is not None:
                if column in self.valid_data.columns:
                    raise ValueError(f"{column} is a reserved column name")

        # convert answer_column to dict
        try:
            self.train_data.loc[:, self.answer_column] = self.train_data[self.answer_column].apply(
                lambda x: ast.literal_eval(x)
            )
        except ValueError:
            logger.warning("Unable to do ast.literal_eval on train_data[answer_column]")
            logger.warning("assuming answer_column is already a dict")

        if self.valid_data is not None:
            try:
                self.valid_data.loc[:, self.answer_column] = self.valid_data[self.answer_column].apply(
                    lambda x: ast.literal_eval(x)
                )
            except ValueError:
                logger.warning("Unable to do ast.literal_eval on valid_data[answer_column]")
                logger.warning("assuming answer_column is already a dict")

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

    def prepare_columns(self, train_df, valid_df):
        train_df.loc[:, "autotrain_text"] = train_df[self.text_column]
        train_df.loc[:, "autotrain_question"] = train_df[self.question_column]
        train_df.loc[:, "autotrain_answer"] = train_df[self.answer_column]
        valid_df.loc[:, "autotrain_text"] = valid_df[self.text_column]
        valid_df.loc[:, "autotrain_question"] = valid_df[self.question_column]
        valid_df.loc[:, "autotrain_answer"] = valid_df[self.answer_column]

        # drop all other columns
        train_df = train_df.drop(
            columns=[
                x for x in train_df.columns if x not in ["autotrain_text", "autotrain_question", "autotrain_answer"]
            ]
        )
        valid_df = valid_df.drop(
            columns=[
                x for x in valid_df.columns if x not in ["autotrain_text", "autotrain_question", "autotrain_answer"]
            ]
        )
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
