import sys
from argparse import ArgumentParser

from loguru import logger

from ..languages import SUPPORTED_LANGUAGES
from . import BaseAutoNLPCommand


def estimator_command_factory(args):
    return EstimatorCommand(args.num_train_samples, args.language)


class EstimatorCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        estimator_parser = parser.add_parser(
            "estimate", description="💰 Fetches estimated cost for a project in AutoNLP"
        )
        estimator_parser.add_argument(
            "--num_train_samples", type=int, default=None, required=True, help="Number of training samples"
        )
        estimator_parser.add_argument(
            "--language",
            type=str,
            default=None,
            required=True,
            metavar="LANGUAGE",
            help=f"The project's language, one of: {SUPPORTED_LANGUAGES}",
            choices=SUPPORTED_LANGUAGES,
        )
        estimator_parser.set_defaults(func=estimator_command_factory)

    def __init__(self, num_train_samples: int, language: str):
        self._num_train_samples = num_train_samples
        self._language = language

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        try:
            resp = client.estimate(num_train_samples=self._num_train_samples, language=self._language)
            print(f"Cost range: {resp['cost_min']} - {resp['cost_max']} USD")
            print("NOTE: This is only an estimate and actual cost may vary!")
        except ValueError:
            logger.error("Something bad happened. Couldn't make the estimate.")
            sys.exit(1)
