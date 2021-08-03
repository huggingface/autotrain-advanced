import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def metrics_command_factory(args):
    return MetricsCommand(args.project, args.username)


class MetricsCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        metrics_parser = parser.add_parser("metrics", description="📊 Fetches models' metrics for a project in AutoNLP")
        metrics_parser.add_argument("--project", type=str, default=None, required=True, help="The project name")
        metrics_parser.add_argument(
            "--username",
            type=str,
            default=None,
            required=False,
            help="The user or org that owns the project, defaults to the selected identity",
        )
        metrics_parser.set_defaults(func=metrics_command_factory)

    def __init__(self, project: str, username: str = None):
        self._project = project
        self._username = username

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        try:
            _ = client.get_metrics(project=self._project, username=self._username)
        except ValueError:
            logger.error("Something bad happened. No metrics found for request")
            sys.exit(1)
