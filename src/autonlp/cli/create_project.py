from argparse import ArgumentParser

from loguru import logger

from ..languages import SUPPORTED_LANGUAGES
from ..tasks import TASKS
from . import BaseAutoNLPCommand


def create_project_command_factory(args):
    return CreateProjectCommand(args.name, args.task, args.language)


class CreateProjectCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create_project")
        create_project_parser.add_argument("--name", type=str, default=None, required=True, help="Project Name")
        create_project_parser.add_argument(
            "--task", type=str, default=None, required=True, help="Project Task Type", choices=list(TASKS.keys())
        )
        create_project_parser.add_argument(
            "--language", type=str, default=None, required=True, help="Language", choices=SUPPORTED_LANGUAGES
        )
        create_project_parser.set_defaults(func=create_project_command_factory)

    def __init__(self, name: str, task: str, language: str):
        self._name = name
        self._task = task
        self._lang = language

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Creating project: {self._name} with task: {self._task}")
        client = AutoNLP()
        project = client.create_project(name=self._name, task=self._task, language=self._lang)
        print(project)
