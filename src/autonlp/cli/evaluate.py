from argparse import ArgumentParser

from loguru import logger

from ..tasks import TASKS
from ..utils import BOLD_TAG as BLD
from ..utils import CYAN_TAG as CYN
from ..utils import GREEN_TAG as GRN
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand


COL_MAPPING_HELP = f"""\
Expected columns for AutoNLP evaluation tasks:
--------------------------------------------------------

{BLD}col_name1{RST} and {BLD}col_name2{RST} refer to columns in your files.

{BLD}`binary_classification`{RST}:
    {BLD}col_name1{RST} -> {BLD}text{RST}    (The text to classify)
    {BLD}col_name2{RST} -> {BLD}target{RST}  (The label)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}text{RST},{GRN}col_name2{RST}:{CYN}target{RST}'

{BLD}`multi_class_classification`{RST}:
    {BLD}col_name1{RST} -> {BLD}text{RST}    (The text to classify)
    {BLD}col_name2{RST} -> {BLD}target{RST}  (The label)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}text,{GRN}col_name2{RST}:{CYN}target{RST}'

{BLD}`entity_extraction`{RST}:
    {BLD}col_name1{RST} -> {BLD}tokens{RST}  (The tokens to tag)
    {BLD}col_name2{RST} -> {BLD}tags{RST}    (The associated tags)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}tokens{RST},{GRN}col_name2{RST}:{CYN}tags{RST}'

{BLD}`speech_recognition`{RST}:
    {BLD}col_name1{RST} -> {BLD}path{RST}  (The path to the audio file, only the file name matters)
    {BLD}col_name2{RST} -> {BLD}text{RST}  (The matching speech transcription)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}path{RST},{GRN}col_name2{RST}:{CYN}text{RST}'
"""


def create_evaluation_command_factory(args):
    return CreateEvaluationCommand(
        args.task,
        args.dataset,
        args.model,
        args.col_mapping,
        args.subset,
    )


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
            choices=list(TASKS.keys()),
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
            required=True,
            help=COL_MAPPING_HELP,
        )
        create_evaluation_parser.add_argument(
            "--subset",
            type=str,
            default="test",
            required=False,
            help="Which subset of dataset to use for evaluation. If not provided, this will default to 'test'.",
        )

        create_evaluation_parser.set_defaults(func=create_evaluation_command_factory)

    def __init__(self, task: str, dataset: str, model: str, col_mapping: str, subset: str):
        self._task = task
        self._model = model
        self._dataset = dataset
        self._col_mapping = col_mapping
        self._subset = subset

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Creating evaluation for task: {self._task}")
        client = AutoNLP()
        eval_project = client.create_evaluation(
            task=self._task,
            dataset=self._dataset,
            model=self._model,
            col_mapping=self._col_mapping,
            subset=self._subset,
        )
        print(eval_project)
