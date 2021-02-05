import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def metrics_command_factory(args):
    if not (args.model_id or args.project):
        logger.error("Either --model_id or --project is required")
        sys.exit(1)
    return MetricsCommand(args.model_id, args.project)


class MetricsCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        metrics_parser = parser.add_parser("metrics")
        metrics_parser.add_argument("--model_id", type=str, default=None, required=False, help="Model ID")
        metrics_parser.add_argument("--project", type=str, default=None, required=False, help="Project ID")
        metrics_parser.set_defaults(func=metrics_command_factory)

    def __init__(self, model_id: str = None, project: str = None):
        self._model_id = model_id
        self._project = project

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        try:
            _ = client.get_metrics(model_id=self._model_id, project=self._project)
        except ValueError:
            logger.error("Something bad happened. No metrics found for request")
            sys.exit(1)
