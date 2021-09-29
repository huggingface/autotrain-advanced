import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional

from loguru import logger

from autonlp.validation import InvalidColMappingError

from ..utils import RED_TAG as RED
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand
from .common import COL_MAPPING_ARG_HELP, COL_MAPPING_HELP


def add_dataset_command_factory(args):
    return AddDatasetCommand(
        project_name=args.project,
        split=args.split,
        col_mapping=args.col_mapping,
        dataset_id=args.dataset_id,
        config_name=args.config_name,
        split_name=args.split_name,
    )


class AddDatasetCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_dataset_parser = parser.add_parser(
            "add_dataset",
            description="📡 Add a dataset from the 🤗 datasets library to your project",
            epilog=COL_MAPPING_HELP,
            formatter_class=RawTextHelpFormatter,
        )
        add_dataset_parser.add_argument(
            "--project", type=str, default=None, required=True, help="The name of the project to upload files to"
        )
        add_dataset_parser.add_argument(
            "--dataset_id", type=str, required=True, help="The name of the dataset, eg 'squad' or 'allenai/scitldr'"
        )
        add_dataset_parser.add_argument("--config_name", type=str, default=None, help="An optional config name")
        add_dataset_parser.add_argument(
            "--split_name", type=str, required=True, help="An optional split name from the datasets library"
        )
        add_dataset_parser.add_argument(
            "--split",
            type=str,
            default=None,
            required=True,
            metavar="SPLIT",
            help=f"The files' split, must be one of {['train', 'valid', 'auto']}",
            choices=["train", "valid", "auto"],
        )
        add_dataset_parser.add_argument(
            "--col_mapping",
            type=str,
            default=None,
            required=True,
            help=COL_MAPPING_ARG_HELP,
        )
        add_dataset_parser.set_defaults(func=add_dataset_command_factory)

    def __init__(
        self,
        project_name: str,
        split: str,
        col_mapping: str,
        dataset_id: str,
        config_name: Optional[str],
        split_name: str,
    ):
        self._name = project_name
        self._split = split
        self._col_mapping = col_mapping
        self._dataset_id = dataset_id
        self._config_name = config_name
        self._split_name = split_name

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Uploading files for project: {self._name}")
        client = AutoNLP()
        try:
            project = client.get_project(name=self._name)
        except ValueError:
            logger.error(f"Project {self._name} not found! You can create it using the create_project command.")
            sys.exit(1)
        splits = self._col_mapping.split(",")
        col_maps = {}
        for s in splits:
            k, v = s.split(":")
            col_maps[k] = v
        logger.info(f"Mapping: {col_maps}")

        try:
            project.add_dataset(
                dataset_id=self._dataset_id,
                config_name=self._config_name,
                dataset_split=self._split_name,
                split=self._split,
                col_mapping=col_maps,
            )
            print(
                f"🎉 Yupee! Dataset {self._dataset_id} has been added to your project.\n"
                f"Once you're done, starting a training here: {RED}autonlp train --project {project.name}{RST}"
            )
        except InvalidColMappingError as err:
            logger.error("❌ The column mapping you provided is incorrect!")
            logger.error("Details:")
            for line in str(err).splitlines():
                logger.error(line)
        except FileNotFoundError as err:
            logger.error(f"❌ Dataset '{self._dataset_id}' not found")
            logger.error("Details:")
            logger.error(str(err))
        except ValueError as err:
            if "Config name is missing" in str(err):
                logger.error(f"❌ You must provide config_name for dataset '{self._dataset_id}'")
            elif "Bad split" in str(err):
                logger.error(f"❌ Split '{self._split_name}' not found")
            else:
                logger.error("❌ Something went wrong!")
            logger.error("Details:")
            logger.error(str(err))
