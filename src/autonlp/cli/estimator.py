import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def estimator_command_factory(args):
    return EstimatorCommand(args.num_train_samples, args.project_name)


class EstimatorCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        estimator_parser = parser.add_parser(
            "estimate", description="💰 Fetches estimated cost for a project in AutoNLP"
        )
        estimator_parser.add_argument(
            "--num_train_samples", type=int, required=True, help="Number of training samples"
        )
        estimator_parser.add_argument(
            "--project_name",
            type=str,
            required=True,
            help="The project's name",
        )
        estimator_parser.set_defaults(func=estimator_command_factory)

    def __init__(self, num_train_samples: int, proj_name: str):
        self._num_train_samples = num_train_samples
        self._proj_name = proj_name

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        try:
            resp = client.estimate(num_train_samples=self._num_train_samples, proj_name=self._proj_name)
            print(f"Cost range: {resp['cost_min']} - {resp['cost_max']} USD")
        except ValueError:
            logger.error("Something bad happened. Couldn't make the estimate.")
            sys.exit(1)
