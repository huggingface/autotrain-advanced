from typing import List, Optional, Union

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class TabularParams(AutoTrainParams):
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
    repo_id: Optional[str] = Field(None, title="Repo ID")
    categorical_columns: Optional[List[str]] = Field(None, title="Categorical columns")
    numerical_columns: Optional[List[str]] = Field(None, title="Numerical columns")
    task: str = Field("classification", title="Task")
    num_trials: int = Field(10, title="Number of trials")
    time_limit: int = Field(600, title="Time limit")
    categorical_imputer: Optional[str] = Field(None, title="Categorical imputer")
    numerical_imputer: Optional[str] = Field(None, title="Numerical imputer")
    numeric_scaler: Optional[str] = Field(None, title="Numeric scaler")
