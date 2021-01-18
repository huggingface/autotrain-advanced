from argparse import ArgumentParser

from . import BaseAutoNLPCommand
from loguru import logger

TASKS = {
    "binary_classification": 1,
    "multi_class_classification": 2,
    "multi_label_classification": 3,
    "entity_extraction": 4,
    "question_answering": 5,
    "translation": 6,
    "multiple_choice": 7,
    "summarization": 8,
    "lm_training": 9,
}


def create_project_command_factory(args):
    return CreateProjectCommand(args.name, args.task)


class CreateProjectCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create_project")
        create_project_parser.add_argument("--name", type=str, default=None, required=True, help="Project Name")
        create_project_parser.add_argument(
            "--task", type=str, default=None, required=True, help="Project Task Type", choices=list(TASKS.keys())
        )
        create_project_parser.set_defaults(func=create_project_command_factory)

    def __init__(self, name: str, task: str):
        self._name = name
        self._task = task

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Creating project: {self._name} with task: {self._task}")
        client = AutoNLP()
        client.create_project(name=self._name, task=self._task)
