import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional

from loguru import logger

from autonlp.validation import InvalidColMappingError, InvalidFileError

from ..utils import BOLD_TAG as BLD
from ..utils import RED_TAG as RED
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand
from .common import COL_MAPPING_ARG_HELP, COL_MAPPING_HELP


def upload_command_factory(args):
    return UploadCommand(
        name=args.project,
        split=args.split,
        col_mapping=args.col_mapping,
        files=args.files,
        path_to_audio=args.path_to_audio,
    )


class UploadCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        upload_parser = parser.add_parser(
            "upload",
            description="📡 Upload files to AutoNLP! The files will be uploaded to a private dataset on your 🤗.co account",
            epilog=COL_MAPPING_HELP,
            formatter_class=RawTextHelpFormatter,
        )
        upload_parser.add_argument(
            "--project", type=str, default=None, required=True, help="The name of the project to upload files to"
        )
        upload_parser.add_argument(
            "--files", type=str, required=True, help="Paths to the files to upload, comma-separated"
        )
        upload_parser.add_argument(
            "--split",
            type=str,
            default=None,
            required=True,
            metavar="SPLIT",
            help=f"The files' split, must be one of {['train', 'valid']}",
            choices=["train", "valid"],
        )
        upload_parser.add_argument(
            "--col_mapping",
            type=str,
            default=None,
            required=True,
            help=COL_MAPPING_ARG_HELP,
        )
        upload_parser.add_argument(
            "--path_to_audio",
            type=str,
            default=None,
            required=False,
            help=(
                "Required for speech recognition task. "
                f"Comma-separated paths to {BLD}directories{RST} containing audio files"
            ),
        )
        upload_parser.set_defaults(func=upload_command_factory)

    def __init__(self, name: str, split: str, col_mapping: str, files: str, path_to_audio: Optional[str]):
        self._name = name
        self._split = split
        self._col_mapping = col_mapping
        self._files = files
        self._path_to_audio = path_to_audio

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

        files = self._files.split(",")
        try:
            project.upload(filepaths=files, split=self._split, col_mapping=col_maps, path_to_audio=self._path_to_audio)
            print(
                "🎉 Yupee! Your files have been uploaded.\n"
                f"Once you're done, starting a training here: {RED}autonlp train --project {project.name}{RST}"
            )
        except ValueError as err:
            logger.error("❌ Something went wrong!")
            logger.error("Details:")
            logger.error(str(err))
        except FileNotFoundError as err:
            logger.error("❌ One path you provided is invalid!")
            logger.error("Details:")
            logger.error(str(err))
        except InvalidFileError as err:
            logger.error("❌ Sorry, AutoNLP is not able to process the files you want to upload")
            logger.error("Details:")
            for line in str(err).splitlines():
                logger.error(line)
        except InvalidColMappingError as err:
            logger.error("❌ The column mapping you provided is incorrect!")
            logger.error("Details:")
            for line in str(err).splitlines():
                logger.error(line)
