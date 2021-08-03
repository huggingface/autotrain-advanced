import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def project_info_command_factory(args):
    return ProjectInfoCommand(args.name, args.is_eval, args.username)


class ProjectInfoCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        project_info_parser = parser.add_parser(
            "project_info", description="♻ Refreshes and displays information about a project"
        )
        project_info_parser.add_argument("--name", type=str, default=None, required=True, help="The project's name")
        project_info_parser.add_argument(
            "--username",
            type=str,
            default=None,
            required=False,
            help="The user or org that owns the project, defaults to the selected identity",
        )
        project_info_parser.add_argument(
            "--is_eval", action="store_true", help="Use `--is_eval` flag if this is an evaluation project"
        )
        project_info_parser.set_defaults(func=project_info_command_factory)

    def __init__(self, name: str, is_eval: bool, username: str = None):
        self._name = name
        self._is_eval = is_eval
        self._username = username

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Fetching info for project: {self._name}")
        client = AutoNLP()
        try:
            project = client.get_project(name=self._name, is_eval=self._is_eval, username=self._username)
        except ValueError:
            logger.error(
                f"Project {self._username}/{self._name} not found! You can create it using the create_project command."
            )
            sys.exit(1)
        print(project)
