from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import common_args, tabular_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.tabular.params import TabularParams

from . import BaseAutoTrainCommand


def run_tabular_command_factory(args):
    return RunAutoTrainTabularCommand(args)


class RunAutoTrainTabularCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--target-columns",
                "help": "Specify the names of the target or label columns separated by commas if multiple. These columns are what the model will predict. Required for defining the output of the model.",
                "required": True,
                "type": str,
            },
            {
                "arg": "--categorical-columns",
                "help": "List the names of columns that contain categorical data, useful for models that need explicit handling of such data. Categorical data is typically processed differently from numerical data, such as through encoding. If not specified, the model will infer the data type.",
                "required": False,
                "type": str,
            },
            {
                "arg": "--numerical-columns",
                "help": "Identify columns that contain numerical data. Proper specification helps in applying appropriate scaling and normalization techniques, which can significantly impact model performance. If not specified, the model will infer the data type.",
                "required": False,
                "type": str,
            },
            {
                "arg": "--id-column",
                "help": "Specify the column name that uniquely identifies each row in the dataset. This is critical for tracking samples through the model pipeline and is often excluded from model training. Required field.",
                "required": True,
                "type": str,
            },
            {
                "arg": "--task",
                "help": "Define the type of machine learning task, such as 'classification', 'regression'. This parameter determines the model's architecture and the loss function to use. Required to properly configure the model.",
                "required": True,
                "type": str,
                "choices": ["classification", "regression"],
            },
            {
                "arg": "--num-trials",
                "help": "Set the number of trials for hyperparameter tuning or model experimentation. More trials can lead to better model configurations but require more computational resources. Default is 100 trials.",
                "required": False,
                "type": int,
                "default": 100,
            },
            {
                "arg": "--time-limit",
                "help": "mpose a time limit (in seconds) for training or searching for the best model configuration. This helps manage resource allocation and ensures the process does not exceed available computational budgets. The default is 3600 seconds (1 hour).",
                "required": False,
                "type": int,
                "default": 3600,
            },
            {
                "arg": "--categorical-imputer",
                "help": "Select the method or strategy to impute missing values in categorical columns. Options might include 'most_frequent', 'None'. Correct imputation can prevent biases and improve model accuracy.",
                "required": False,
                "type": str,
                "choices": ["most_frequent", None],
                "default": None,
            },
            {
                "arg": "--numerical-imputer",
                "help": "Choose the imputation strategy for missing values in numerical columns. Common strategies include 'mean', & 'median'. Accurate imputation is vital for maintaining the integrity of numerical data.",
                "required": False,
                "type": str,
                "choices": ["mean", "median", None],
                "default": None,
            },
            {
                "arg": "--numeric-scaler",
                "help": "Determine the type of scaling to apply to numerical data. Examples include 'standard' (zero mean and unit variance), 'min-max' (scaled between given range), etc. Scaling is essential for many algorithms to perform optimally",
                "required": False,
                "type": str,
                "choices": ["standard", "minmax", "normal", "robust"],
            },
        ]
        arg_list = common_args() + arg_list
        remove_args = ["--disable_gradient_checkpointing", "--gradient_accumulation", "--epochs", "--log", "--lr"]
        arg_list = [arg for arg in arg_list if arg["arg"] not in remove_args]
        run_tabular_parser = parser.add_parser("tabular", description="✨ Run AutoTrain Tabular Data Training")
        for arg in arg_list:
            if "action" in arg:
                run_tabular_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_tabular_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_tabular_parser.set_defaults(func=run_tabular_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "push_to_hub",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

        if self.args.train:
            if self.args.project_name is None:
                raise ValueError("Project name must be specified")
            if self.args.data_path is None:
                raise ValueError("Data path must be specified")
            if self.args.model is None:
                raise ValueError("Model must be specified")
            if self.args.push_to_hub:
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

        self.args.target_columns = [k.strip() for k in self.args.target_columns.split(",")]

    def run(self):
        logger.info("Running Tabular Training")
        if self.args.train:
            params = TabularParams(**vars(self.args))
            params = tabular_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
