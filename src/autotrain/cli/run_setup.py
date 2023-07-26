import subprocess
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoTrainCommand


def run_app_command_factory(args):
    return RunSetupCommand(args.update_torch)


class RunSetupCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_setup_parser = parser.add_parser(
            "setup",
            description="✨ Run AutoTrain setup",
        )
        run_setup_parser.add_argument(
            "--update-torch",
            action="store_true",
            help="Update PyTorch to latest version",
        )
        run_setup_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, update_torch: bool):
        self.update_torch = update_torch

    def run(self):
        # install latest transformers
        cmd = "pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers.git"
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Installing latest transformers@main")
        _, _ = pipe.communicate()
        logger.info("Successfully installed latest transformers")

        cmd = "pip uninstall -y peft && pip install git+https://github.com/huggingface/peft.git"
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Installing latest peft@main")
        _, _ = pipe.communicate()
        logger.info("Successfully installed latest peft")

        cmd = "pip uninstall -y diffusers && pip install git+https://github.com/huggingface/diffusers.git"
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Installing latest diffusers@main")
        _, _ = pipe.communicate()
        logger.info("Successfully installed latest diffusers")

        cmd = "pip uninstall -y trl && pip install git+https://github.com/lvwerra/trl.git"
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Installing latest trl@main")
        _, _ = pipe.communicate()
        logger.info("Successfully installed latest trl")

        if self.update_torch:
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Installing latest PyTorch")
            _, _ = pipe.communicate()
            logger.info("Successfully installed latest PyTorch")
