from argparse import ArgumentParser

import torch

from autotrain import logger
from autotrain.cli.utils import img_clf_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.image_classification.params import ImageClassificationParams

from . import BaseAutoTrainCommand


def run_image_classification_command_factory(args):
    return RunAutoTrainImageClassificationCommand(args)


class RunAutoTrainImageClassificationCommand(BaseAutoTrainCommand):
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
                "arg": "--train-split",
                "help": "Test dataset split to use",
                "required": False,
                "type": str,
                "default": "train",
            },
            {
                "arg": "--valid-split",
                "help": "Validation dataset split to use",
                "required": False,
                "type": str,
                "default": None,
            },
            {
                "arg": "--image-column",
                "help": "Image column to use",
                "required": False,
                "type": str,
                "default": "image",
            },
            {
                "arg": "--target-column",
                "help": "Target column to use",
                "required": False,
                "type": str,
                "default": "target",
            },
            {
                "arg": "--model",
                "help": "Model to use",
                "required": False,
                "type": str,
            },
            {
                "arg": "--lr",
                "help": "Learning rate to use",
                "required": False,
                "type": float,
                "default": 3e-5,
            },
            {
                "arg": "--epochs",
                "help": "Number of training epochs to use",
                "required": False,
                "type": int,
                "default": 1,
            },
            {
                "arg": "--batch-size",
                "help": "Training batch size to use",
                "required": False,
                "type": int,
                "default": 2,
            },
            {
                "arg": "--warmup-ratio",
                "help": "Warmup proportion to use",
                "required": False,
                "type": float,
                "default": 0.1,
            },
            {
                "arg": "--gradient-accumulation",
                "help": "Gradient accumulation steps to use",
                "required": False,
                "type": int,
                "default": 1,
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
                "arg": "--seed",
                "help": "Seed to use",
                "required": False,
                "type": int,
                "default": 42,
            },
            {
                "arg": "--logging-steps",
                "help": "Logging steps to use",
                "required": False,
                "type": int,
                "default": -1,
            },
            {
                "arg": "--project-name",
                "help": "Output directory",
                "required": False,
                "type": str,
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
            {
                "arg": "--push-to-hub",
                "help": "Push to hub True/False. In case you want to push the trained model to huggingface hub",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--repo-id",
                "help": "Repo id for hugging face hub",
                "required": False,
                "type": str,
            },
            {
                "arg": "--log",
                "help": "Use experiment tracking",
                "required": False,
                "type": str,
                "default": "none",
            },
            {
                "arg": "--backend",
                "help": "Backend to use: default or spaces. Spaces backend requires push_to_hub and repo_id",
                "required": False,
                "type": str,
                "default": "local-cli",
            },
        ]
        run_text_classification_parser = parser.add_parser(
            "image-classification", description="✨ Run AutoTrain Image Classification"
        )
        for arg in arg_list:
            if "action" in arg:
                run_text_classification_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_text_classification_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                )
        run_text_classification_parser.set_defaults(func=run_image_classification_command_factory)

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

        if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
            if not self.args.push_to_hub:
                raise ValueError("Push to hub must be specified for spaces backend")
            if self.args.username is None and self.args.repo_id is None:
                raise ValueError("Repo id or username must be specified for spaces backend")
            if self.args.token is None:
                raise ValueError("Token must be specified for spaces backend")

        if not torch.cuda.is_available():
            self.device = "cpu"

        self.num_gpus = torch.cuda.device_count()

    def run(self):
        logger.info("Running Text Classification")
        if self.args.train:
            params = ImageClassificationParams(**vars(self.args))
            params = img_clf_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            _ = project.create()
