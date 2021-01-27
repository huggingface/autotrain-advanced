import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def upload_command_factory(args):
    return UploadCommand(args.project, args.split, args.col_mapping, args.files)


class UploadCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        upload_parser = parser.add_parser("upload")
        upload_parser.add_argument("--project", type=str, default=None, required=True, help="Project Name")
        upload_parser.add_argument(
            "--split", type=str, default=None, required=True, help="File Split Type", choices=["train", "valid"]
        )
        upload_parser.add_argument(
            "--col_mapping", type=str, default=None, required=True, help="Column Mapping. E.g. col1:map1,col2:map2"
        )
        upload_parser.add_argument(
            "--files", type=str, required=True, help="Paths to the files to upload, comma-separated"
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
        project.upload(filepaths=files, split=self._split, col_mapping=col_maps)
