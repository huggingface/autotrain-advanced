from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.asr.params import WhisperTrainingParams

from . import BaseAutoTrainCommand


def run_asr_command_factory(args):
    return RunAutoTrainASRCommand(args)


class RunAutoTrainASRCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(WhisperTrainingParams)
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
                "help": "Backend",
                "required": False,
                "type": str,
                "default": "local",
            },
        ] + arg_list
        
        run_asr_parser = parser.add_parser(
            "asr", description="✨ Run AutoTrain Automatic Speech Recognition (Whisper)"
        )
        for arg in arg_list:
            if arg["arg"] == "--target_modules":
                run_asr_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg["required"],
                    type=str,
                    nargs="+",
                    default=arg.get("default"),
                )
            elif "action" in arg:
                run_asr_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg["required"],
                    action=arg["action"],
                    default=arg.get("default"),
                )
            else:
                run_asr_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg["required"],
                    type=arg["type"],
                    default=arg.get("default"),
                )
        run_asr_parser.set_defaults(func=run_asr_command_factory)

    def __init__(self, args):
        self.args = args
        self.train = args.train
        self.deploy = args.deploy
        self.inference = args.inference
        self.backend = args.backend
        
        # Convert args to WhisperTrainingParams
        params_dict = vars(args)
        # Remove args that are not part of WhisperTrainingParams
        params_dict.pop("train", None)
        params_dict.pop("deploy", None)
        params_dict.pop("inference", None)
        params_dict.pop("backend", None)
        params_dict.pop("func", None)
        params_dict.pop("command", None)
        
        self.params = WhisperTrainingParams(**params_dict)

    def run(self):
        if not any([self.train, self.deploy, self.inference]):
            logger.warning(
                "No action specified. Please specify one of --train, --deploy, or --inference."
            )
            return
        
        if self.train:
            logger.info("Training ASR model...")
            project = AutoTrainProject(params=self.params, backend=self.backend)
            project.create()
        
        if self.deploy:
            logger.info("Deploying ASR model...")
            # Add deployment logic here
        
        if self.inference:
            logger.info("Running inference with ASR model...")
            # Add inference logic here 