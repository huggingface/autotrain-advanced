import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from loguru import logger

from autonlp.validation import InvalidColMappingError, InvalidFileError

from ..utils import BOLD_TAG as BLD
from ..utils import CYAN_TAG as CYN
from ..utils import GREEN_TAG as GRN
from ..utils import RED_TAG as RED
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand


COL_MAPPING_ARG_HELP = f"""\
The files' column mapping. Must be like this:
'{GRN}col_name{RST}:{CYN}autonlp_col_name{RST},{GRN}col_name{RST}:{CYN}autonlp_col_name{RST}'
where '{CYN}autonlp_col_name{RST}' corresponds to an expected column in AutoNLP, and
'{GRN}col_name{RST}' is the corresponding column in your files.
"""

COL_MAPPING_HELP = f"""\
Expected columns for AutoNLP tasks:
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

"""


def upload_command_factory(args):
    return UploadCommand(args.project, args.split, args.col_mapping, args.files)


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
        upload_parser.set_defaults(func=upload_command_factory)

    def __init__(self, name: str, split: str, col_mapping: str, files: str):
        self._name = name
        self._split = split
        self._col_mapping = col_mapping
        self._files = files

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
            project.upload(filepaths=files, split=self._split, col_mapping=col_maps)
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

        print(
            "🎉 Yupee! Your files have been uploaded.\n"
            f"Once you're done, starting a training here: {RED}autonlp train --project {project.name}{RST}"
        )
