from typing import List, Optional, Union

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class TabularParams(AutoTrainParams):
    """
    TabularParams is a configuration class for tabular data training parameters.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the model to use. Default is "xgboost".
        username (Optional[str]): Hugging Face Username.
        seed (int): Random seed for reproducibility. Default is 42.
        train_split (str): Name of the training data split. Default is "train".
        valid_split (Optional[str]): Name of the validation data split.
        project_name (str): Name of the output directory. Default is "project-name".
        token (Optional[str]): Hub Token for authentication.
        push_to_hub (bool): Whether to push the model to the hub. Default is False.
        id_column (str): Name of the ID column. Default is "id".
        target_columns (Union[List[str], str]): Target column(s) in the dataset. Default is ["target"].
        categorical_columns (Optional[List[str]]): List of categorical columns.
        numerical_columns (Optional[List[str]]): List of numerical columns.
        task (str): Type of task (e.g., "classification"). Default is "classification".
        num_trials (int): Number of trials for hyperparameter optimization. Default is 10.
        time_limit (int): Time limit for training in seconds. Default is 600.
        categorical_imputer (Optional[str]): Imputer strategy for categorical columns.
        numerical_imputer (Optional[str]): Imputer strategy for numerical columns.
        numeric_scaler (Optional[str]): Scaler strategy for numerical columns.
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("xgboost", title="Model name")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    seed: int = Field(42, title="Seed")
    train_split: str = Field("train", title="Train split")
    valid_split: Optional[str] = Field(None, title="Validation split")
    project_name: str = Field("project-name", title="Output directory")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    id_column: str = Field("id", title="ID column")
    target_columns: Union[List[str], str] = Field(["target"], title="Target column(s)")
    categorical_columns: Optional[List[str]] = Field(None, title="Categorical columns")
    numerical_columns: Optional[List[str]] = Field(None, title="Numerical columns")
    task: str = Field("classification", title="Task")
    num_trials: int = Field(10, title="Number of trials")
    time_limit: int = Field(600, title="Time limit")
    categorical_imputer: Optional[str] = Field(None, title="Categorical imputer")
    numerical_imputer: Optional[str] = Field(None, title="Numerical imputer")
    numeric_scaler: Optional[str] = Field(None, title="Numeric scaler")
