import os
import subprocess
import sys
from argparse import ArgumentParser

import torch

from autotrain import logger
from autotrain.backend import EndpointsRunner, SpaceRunner

from . import BaseAutoTrainCommand


def run_text_classification_command_factory(args):
    return RunAutoTrainTextClassificationCommand(args)


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
                "arg": "--max-seq-length",
                "help": "Maximum number of tokens in a sequence to use",
                "required": False,
                "type": int,
                "default": 128,
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
                "arg": "--fp16",
                "help": "FP16 True/False",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--token",
                "help": "Hugging face token",
                "required": False,
                "type": str,
                "default": "",
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
                "arg": "--backend",
                "help": "Backend to use: default or spaces. Spaces backend requires push_to_hub and repo_id",
                "required": False,
                "type": str,
                "default": "default",
            },
            {
                "arg": "--username",
                "help": "Huggingface username to use",
                "required": False,
                "type": str,
            },
        ]
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
            "fp16",
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

        if not torch.cuda.is_available():
            self.device = "cpu"

        self.num_gpus = torch.cuda.device_count()

        if len(str(self.args.token)) < 6:
            self.args.token = os.environ.get("HF_TOKEN", None)

    def run(self):
        from autotrain.trainers.text_classification.__main__ import train as train_text_classification
        from autotrain.trainers.text_classification.params import TextClassificationParams

        logger.info("Running Text Classification")
        if self.args.train:
            params = TextClassificationParams(
                data_path=self.args.data_path,
                train_split=self.args.train_split,
                valid_split=self.args.valid_split,
                text_column=self.args.text_column,
                target_column=self.args.target_column,
                model_name=self.args.model,
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
                fp16=self.args.fp16,
                push_to_hub=self.args.push_to_hub,
                repo_id=self.args.repo_id,
                token=self.args.token,
                username=self.args.username,
            )

            if self.args.backend.startswith("spaces"):
                logger.info("Creating space...")
                sr = SpaceRunner(
                    params=params,
                    backend=self.args.backend,
                )
                space_id = sr.prepare()
                logger.info(f"Training Space created. Check progress at https://hf.co/spaces/{space_id}")
                sys.exit(0)

            if self.args.backend.startswith("ep-"):
                logger.info("Creating training endpoint...")
                sr = EndpointsRunner(
                    params=params,
                    backend=self.args.backend,
                )
                sr.prepare()
                logger.info("Training endpoint created.")
                sys.exit(0)

            params.save(output_dir=self.args.project_name)
            if self.num_gpus == 1:
                train_text_classification(params)
            else:
                cmd = ["accelerate", "launch", "--multi_gpu", "--num_machines", "1", "--num_processes"]
                cmd.append(str(self.num_gpus))
                cmd.append("--mixed_precision")
                if self.args.fp16:
                    cmd.append("fp16")
                else:
                    cmd.append("no")

                cmd.extend(
                    [
                        "-m",
                        "autotrain.trainers.text_classification",
                        "--training_config",
                        os.path.join(self.args.project_name, "training_params.json"),
                    ]
                )

                env = os.environ.copy()
                process = subprocess.Popen(cmd, env=env)
                process.wait()
