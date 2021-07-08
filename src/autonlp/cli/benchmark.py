from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def create_benchmark_command_factory(args):
    return CreateBenchmarkCommand(args.dataset, args.submission, args.eval_name)


class CreateBenchmarkCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_benchmark_parser = parser.add_parser("benchmark", description="✨ Creates an evaluation in AutoNLP.")
        create_benchmark_parser.add_argument(
            "--dataset",
            metavar="DATASET",
            type=str,
            default=None,
            required=True,
        )
        create_benchmark_parser.add_argument(
            "--submission",
            metavar="SUBMISSION",
            type=str,
            default=None,
            required=True,
        )
        create_benchmark_parser.add_argument(
            "--eval_name",
            metavar="EVAL_NAME",
            type=str,
            default=None,
            required=True,
        )

        create_benchmark_parser.set_defaults(func=create_benchmark_command_factory)

    def __init__(self, dataset, submission, eval_name):
        self._dataset = dataset
        self._submission = submission
        self._eval_name = eval_name

    def run(self):
        from ..autonlp import AutoNLP

        logger.info("Creating benchmark")
        client = AutoNLP()
        eval_project = client.create_benchmark(
            dataset=self._dataset,
            submission=self._submission,
            eval_name=self._eval_name,
        )
        print(eval_project)
