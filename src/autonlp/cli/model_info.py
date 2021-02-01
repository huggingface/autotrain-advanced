import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def model_info_command_factory(args):
    return ModelInfoCommand(args.id)


class ModelInfoCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        model_info_parser = parser.add_parser("model_info")
        model_info_parser.add_argument("--id", type=str, default=None, required=True, help="Job/Model ID")
        model_info_parser.set_defaults(func=model_info_command_factory)

    def __init__(self, id: str):
        self._id = id

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Fetching info for model: {self._id}")
        client = AutoNLP()
        try:
            _ = client.get_model_info(model_id=self._id)
        except ValueError:
            logger.error(f"Model {self._id} not found!")
            sys.exit(1)
