from argparse import ArgumentParser

from loguru import logger

from ..tasks import DATASETS_TASKS, TASKS
from . import BaseAutoNLPCommand
from .common import COL_MAPPING_HELP


def create_evaluation_command_factory(args):
    if args.task not in DATASETS_TASKS:
        if args.col_mapping is None:
            raise Exception("`col_mapping` is required if task is not a datasets task")
    return CreateEvaluationCommand(args.task, args.dataset, args.model, args.col_mapping, args.split, args.config)


class CreateEvaluationCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_evaluation_parser = parser.add_parser("evaluate", description="✨ Creates an evaluation in AutoNLP.")
        create_evaluation_parser.add_argument(
            "--task",
            metavar="TASK",
            type=str,
            default=None,
            required=True,
            help=f"The evaluation task type, one of: {list(TASKS.keys())}",
            choices=list(TASKS.keys()) + DATASETS_TASKS,
        )
        create_evaluation_parser.add_argument(
            "--dataset",
            metavar="DATASET",
            type=str,
            default=None,
            required=True,
            help="Dataset from the hub or from datasets libray. e.g. 'imdb', 'username/awesome_dataset', etc",
        )
        create_evaluation_parser.add_argument(
            "--model",
            metavar="MODEL",
            type=str,
            default=None,
            required=True,
            help="Model from the hub, e.g. 'bert-base-uncased', 'facebook/bart-large-mnli', etc. Model must be compatible with task and the dataset",
        )
        create_evaluation_parser.add_argument(
            "--col_mapping",
            type=str,
            default=None,
            required=False,
            help=COL_MAPPING_HELP,
        )
        create_evaluation_parser.add_argument(
            "--split",
            type=str,
            default="test",
            required=False,
            help="Which split of dataset to use for evaluation. If not provided, this will default to 'test'.",
        )
        create_evaluation_parser.add_argument(
            "--config",
            type=str,
            default=None,
            required=False,
            help="Which config of dataset to use for evaluation. If not provided, this will default to None.",
        )

        create_evaluation_parser.set_defaults(func=create_evaluation_command_factory)

    def __init__(self, task: str, dataset: str, model: str, col_mapping: str, split: str, config: str = None):
        self._task = task
        self._model = model
        self._dataset = dataset
        self._col_mapping = col_mapping
        self._split = split
        self._config = config

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Creating evaluation for task: {self._task}")
        client = AutoNLP()
        eval_project = client.create_evaluation(
            task=self._task,
            dataset=self._dataset,
            model=self._model,
            split=self._split,
            col_mapping=self._col_mapping,
            config=self._config,
        )
        print(eval_project)
