import os
import subprocess
import sys
from argparse import ArgumentParser

import torch

from autotrain import logger
from autotrain.spacerunner import EndpointsRunner, SpaceRunner
from autotrain.trainers.tabular.__main__ import train as train_tabular
from autotrain.trainers.tabular.params import TabularParams

from . import BaseAutoTrainCommand


# class TabularClassificationParams(AutoTrainParams):
#     data_path: str = Field(None, title="Data path")
#     model: str = Field("xgboost", title="Model name")
#     username: str = Field(None, title="Hugging Face Username")
#     seed: int = Field(42, title="Seed")
#     train_split: str = Field("train", title="Train split")
#     valid_split: str = Field(None, title="Validation split")
#     project_name: str = Field("Project Name", title="Output directory")
#     token: str = Field(None, title="Hub Token")
#     push_to_hub: bool = Field(False, title="Push to hub")
#     id_column: str = Field("id", title="ID column")
#     target_columns: Union[List[str], str] = Field(["target"], title="Target column(s)")
#     repo_id: str = Field(None, title="Repo ID")
#     categorical_columns: List[str] = Field(None, title="Categorical columns")
#     numerical_columns: List[str] = Field(None, title="Numerical columns")
#     task: str = Field("classification", title="Task")
#     num_trials: int = Field(10, title="Number of trials")
#     time_limit: int = Field(600, title="Time limit")
#     categorical_imputer: str = Field(None, title="Categorical imputer")
#     numerical_imputer: str = Field(None, title="Numerical imputer")
#     numeric_scaler: str = Field(None, title="Numeric scaler")


def run_tabular_command_factory(args):
    return RunAutoTrainTabularCommand(args)


class RunAutoTrainTextClassificationCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--train",
                "help": "Train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Deploy the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Run inference",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--data-path",
                "help": "Train dataset to use",
                "required": False,
                "type": str,
            },
            {
                "arg": "--model",
                "help": "Model name",
                "required": True,
                "type": str,
            },
        ]
