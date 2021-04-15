from argparse import ArgumentParser

from loguru import logger

from ..languages import SUPPORTED_LANGUAGES
from ..tasks import TASKS
from ..utils import RED_TAG as RED
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand


def create_project_command_factory(args):
    if args.max_models <= 0:
        raise ValueError("max_models cannot be 0 or negative")
    if args.max_models > 150:
        raise ValueError("Please choose a value from 1 to 150 for max_models")
    return CreateProjectCommand(args.name, args.task, args.language, args.max_models)


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
            help="The project's language. Please check supported languages in AutoNLP documentation.",
            choices=SUPPORTED_LANGUAGES,
        )
        create_project_parser.add_argument(
            "--max_models",
            type=int,
            default=10,
            required=True,
            metavar="MAX_MODELS",
            help="Maximum number of models you want to train in this project. More models => higher chances of getting awesome models. Also, more models => higher expenses",
        )
        create_project_parser.set_defaults(func=create_project_command_factory)

    def __init__(self, name: str, task: str, language: str, max_models: int):
        self._name = name
        self._task = task
        self._lang = language
        self._max_models = max_models

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Creating project: {self._name} with task: {self._task}")
        client = AutoNLP()
        project = client.create_project(
            name=self._name, task=self._task, language=self._lang, max_models=self._max_models
        )
        print(project)
        print(f'Upload files to your project: {RED}autonlp upload --project "{project.name}"{RST}')
