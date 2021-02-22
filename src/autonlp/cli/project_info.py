import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def project_info_command_factory(args):
    return ProjectInfoCommand(args.name)


class ProjectInfoCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        project_info_parser = parser.add_parser("project_info")
        project_info_parser.add_argument("--name", type=str, default=None, required=True, help="Project Name")
        project_info_parser.set_defaults(func=project_info_command_factory)

    def __init__(self, name: str):
        self._name = name

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Fetching info for project: {self._name}")
        client = AutoNLP()
        try:
            project = client.get_project(name=self._name)
        except ValueError:
            logger.error(f"Project {self._name} not found! You can create it using the create_project command.")
            sys.exit(1)
        print(project)
