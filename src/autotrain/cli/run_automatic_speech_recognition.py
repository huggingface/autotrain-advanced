import argparse
import json
import os
from typing import Optional

from autotrain import logger
from autotrain.cli.run import AutoTrainCLI
from autotrain.cli.utils import get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.automatic_speech_recognition import __main__ as trainer
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams

from . import BaseAutoTrainCommand


def run_ASR_command_factory(args):
    return RunAutoTrainAutomaticSpeechRecognitionCommand(args)


class RunAutoTrainAutomaticSpeechRecognitionCommand(AutoTrainCLI):
    @staticmethod
    def register_subcommand(parser: argparse._SubParsersAction):
        arg_list = get_field_info(AutomaticSpeechRecognitionParams)
        arg_list = [
            {
                "arg": "--train",
                "help": "Command to train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Command to deploy the model (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Command to run inference (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--backend",
                "help": "Backend to use for training",
                "required": False,
                "default": "local",
            },
        ] + arg_list
        run_parser = parser.add_parser(
            "ASR",
            description="✨ Run AutoTrain Automatic Speech Recognition"
        )
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_parser.set_defaults(func=run_ASR_command_factory)

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
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

    def run(self):
        """Run the ASR training."""
        logger.info("Running Automatic Speech Recognition")
        if self.args.train:
            params = AutomaticSpeechRecognitionParams(**vars(self.args))
            project = AutoTrainProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
        
        logger.info("Starting ASR training...")
        
        # Create training config
        training_config = {
            "task": "ASR",
            "model": self.args.model,
            "train_data": self.args.train_data,
            "valid_data": self.args.valid_data,
            "project_name": self.args.project_name,
            "text_column": self.args.text_column,
            "audio_column": self.args.audio_column,
            "max_duration": self.args.max_duration,
            "sampling_rate": self.args.sampling_rate,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "learning_rate": self.args.learning_rate,
            "optimizer": self.args.optimizer,
            "scheduler": self.args.scheduler,
            "mixed_precision": self.args.mixed_precision,
            "weight_decay": self.args.weight_decay,
            "warmup_ratio": self.args.warmup_ratio,
            "early_stopping_patience": self.args.early_stopping_patience,
            "early_stopping_threshold": self.args.early_stopping_threshold,
            "eval_strategy": self.args.eval_strategy,
            "save_total_limit": self.args.save_total_limit,
            "auto_find_batch_size": self.args.auto_find_batch_size,
            "logging_steps": self.args.logging_steps,
        }
        
        # Save config
        config_path = f"{self.args.project_name}/training_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)
        
        
        trainer.train(training_config) 