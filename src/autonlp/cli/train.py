from argparse import ArgumentParser

from . import BaseAutoNLPCommand
from loguru import logger


def train_command_factory(args):
    return TrainCommand(args.project)


class TrainCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        train_parser = parser.add_parser("train")
        train_parser.add_argument("--project", type=str, default=None, required=True, help="Project Name")
        train_parser.set_defaults(func=train_command_factory)

    def __init__(self, name: str):
        self._name = name

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Starting Training For Project: {self._name}")

        client = AutoNLP()
        project = client.get_project(name=self._name)
        project.train()
