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
                "help": "Target column(s), separated by commas",
                "required": True,
                "type": str,
            },
            {
                "arg": "--categorical-columns",
                "help": "Categorical columns",
                "required": False,
                "type": str,
            },
            {
                "arg": "--numerical-columns",
                "help": "Numerical columns",
                "required": False,
                "type": str,
            },
            {
                "arg": "--id-column",
                "help": "ID column",
                "required": True,
                "type": str,
            },
            {
                "arg": "--task",
                "help": "Task",
                "required": True,
                "type": str,
            },
            {
                "arg": "--num-trials",
                "help": "Number of trials",
                "required": False,
                "type": int,
                "default": 100,
            },
            {
                "arg": "--time-limit",
                "help": "Time limit",
                "required": False,
                "type": int,
                "default": 3600,
            },
            {
                "arg": "--categorical-imputer",
                "help": "Categorical imputer",
                "required": False,
                "type": str,
            },
            {
                "arg": "--numerical-imputer",
                "help": "Numerical imputer",
                "required": False,
                "type": str,
            },
            {
                "arg": "--numeric-scaler",
                "help": "Numeric scaler",
                "required": False,
                "type": str,
            },
        ]
        arg_list.extend(common_args())
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
        logger.info("Running Tabular Training...")
        if self.args.train:
            params = TabularParams(
                data_path=self.args.data_path,
                model=self.args.model,
                username=self.args.username,
                seed=self.args.seed,
                train_split=self.args.train_split,
                valid_split=self.args.valid_split,
                project_name=self.args.project_name,
                token=self.args.token,
                push_to_hub=self.args.push_to_hub,
                id_column=self.args.id_column,
                target_columns=self.args.target_columns,
                repo_id=self.args.repo_id,
                categorical_columns=self.args.categorical_columns,
                numerical_columns=self.args.numerical_columns,
                task=self.args.task,
                num_trials=self.args.num_trials,
                time_limit=self.args.time_limit,
                categorical_imputer=self.args.categorical_imputer,
                numerical_imputer=self.args.numerical_imputer,
                numeric_scaler=self.args.numeric_scaler,
            )

            params = tabular_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            _ = project.create()
