from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import common_args, text_clf_munge_data
from autotrain.project import AutoTrainProject
from autotrain.trainers.text_classification.params import TextClassificationParams

from . import BaseAutoTrainCommand


def run_text_classification_command_factory(args):
    return RunAutoTrainTextClassificationCommand(args)


class RunAutoTrainTextClassificationCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = [
            {
                "arg": "--text-column",
                "help": "Text column to use",
                "required": False,
                "type": str,
                "default": "text",
            },
            {
                "arg": "--target-column",
                "help": "Target column to use",
                "required": False,
                "type": str,
                "default": "target",
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
        run_text_classification_parser = parser.add_parser(
            "text-classification", description="✨ Run AutoTrain Text Classification"
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
        run_text_classification_parser.set_defaults(func=run_text_classification_command_factory)

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
        logger.info("Running Text Classification")
        if self.args.train:
            params = TextClassificationParams(
                data_path=self.args.data_path,
                train_split=self.args.train_split,
                valid_split=self.args.valid_split,
                text_column=self.args.text_column,
                target_column=self.args.target_column,
                model=self.args.model,
                lr=self.args.lr,
                epochs=self.args.epochs,
                max_seq_length=self.args.max_seq_length,
                batch_size=self.args.batch_size,
                warmup_ratio=self.args.warmup_ratio,
                gradient_accumulation=self.args.gradient_accumulation,
                optimizer=self.args.optimizer,
                scheduler=self.args.scheduler,
                weight_decay=self.args.weight_decay,
                max_grad_norm=self.args.max_grad_norm,
                seed=self.args.seed,
                logging_steps=self.args.logging_steps,
                project_name=self.args.project_name,
                evaluation_strategy=self.args.evaluation_strategy,
                save_total_limit=self.args.save_total_limit,
                save_strategy=self.args.save_strategy,
                auto_find_batch_size=self.args.auto_find_batch_size,
                mixed_precision=self.args.mixed_precision,
                push_to_hub=self.args.push_to_hub,
                repo_id=self.args.repo_id,
                token=self.args.token,
                username=self.args.username,
                log=self.args.log,
            )

            params = text_clf_munge_data(params, local=self.args.backend.startswith("local"))
            project = AutoTrainProject(params=params, backend=self.args.backend)
            _ = project.create()
