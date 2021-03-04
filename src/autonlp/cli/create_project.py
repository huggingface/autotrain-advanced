from argparse import ArgumentParser

from loguru import logger

from ..languages import SUPPORTED_LANGUAGES
from ..tasks import TASKS
from ..utils import RED_TAG as RED
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand


def create_project_command_factory(args):
    return CreateProjectCommand(args.name, args.task, args.language)


class CreateProjectCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create_project", description="✨ Creates a project in AutoNLP.")
        create_project_parser.add_argument("--name", type=str, default=None, required=True, help="The project's name")
        create_project_parser.add_argument(
            "--task",
            metavar="TASK",
            type=str,
            default=None,
            required=True,
            help=f"The project's task type, one of: {list(TASKS.keys())}",
            choices=list(TASKS.keys()),
        )
        create_project_parser.add_argument(
            "--language",
            type=str,
            default=None,
            required=True,
            metavar="LANGUAGE",
            help=f"The project's language, one of: {SUPPORTED_LANGUAGES}",
            choices=SUPPORTED_LANGUAGES,
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
        print(f'Upload files to your project: {RED}autonlp upload --project "{project.name}"{RST}')
