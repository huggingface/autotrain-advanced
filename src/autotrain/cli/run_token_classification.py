from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import common_args, token_clf_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.token_classification.params import TokenClassificationParams

from . import BaseAutoTrainCommand


def run_token_classification_command_factory(args):
    return RunAutoTrainTokenClassificationCommand(args)


class RunAutoTrainTokenClassificationCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--tokens-column",
                "help": "Tokens column to use. Must be a stringified list of tokens",
                "required": False,
                "type": str,
                "default": "tokens",
            },
            {
                "arg": "--tags-column",
                "help": "Tags column to use. Must be a stringified list of tags",
                "required": False,
                "type": str,
                "default": "tags",
            },
            {
                "arg": "--max-seq-length",
                "help": "Maximum number of tokens in a sequence to use",
                "required": False,
                "type": int,
                "default": 128,
            },
            {
                "arg": "--warmup-ratio",
                "help": "Warmup proportion to use",
                "required": False,
                "type": float,
                "default": 0.1,
            },
            {
                "arg": "--optimizer",
                "help": "Optimizer to use",
                "required": False,
                "type": str,
                "default": "adamw_torch",
            },
            {
                "arg": "--scheduler",
                "help": "Scheduler to use",
                "required": False,
                "type": str,
                "default": "linear",
            },
            {
                "arg": "--weight-decay",
                "help": "Weight decay to use",
                "required": False,
                "type": float,
                "default": 0.0,
            },
            {
                "arg": "--max-grad-norm",
                "help": "Max gradient norm to use",
                "required": False,
                "type": float,
                "default": 1.0,
            },
            {
                "arg": "--logging-steps",
                "help": "Logging steps to use",
                "required": False,
                "type": int,
                "default": -1,
            },
            {
                "arg": "--evaluation-strategy",
                "help": "Evaluation strategy to use",
                "required": False,
                "type": str,
                "default": "epoch",
            },
            {
                "arg": "--save-total-limit",
                "help": "Save total limit to use",
                "required": False,
                "type": int,
                "default": 1,
            },
            {
                "arg": "--save-strategy",
                "help": "Save strategy to use",
                "required": False,
                "type": str,
                "default": "epoch",
            },
            {
                "arg": "--auto-find-batch-size",
                "help": "Auto find batch size True/False",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--mixed-precision",
                "help": "fp16, bf16, or None",
                "required": False,
                "type": str,
                "default": None,
                "choices": ["fp16", "bf16", None],
            },
        ]
        arg_list.extend(common_args())
        run_token_classification_parser = parser.add_parser(
            "token-classification", description="✨ Run AutoTrain Token Classification"
        )
        for arg in arg_list:
            if "action" in arg:
                run_token_classification_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_token_classification_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                )
        run_token_classification_parser.set_defaults(func=run_token_classification_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "auto_find_batch_size",
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
                if self.args.repo_id is None:
                    raise ValueError("Repo id must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

    def run(self):
        logger.info("Running Token Classification")
        if self.args.train:
            params = TokenClassificationParams(**vars(self.args))
            params = token_clf_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            _ = project.create()
