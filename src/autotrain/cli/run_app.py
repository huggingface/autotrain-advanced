import os
from argparse import ArgumentParser

from . import BaseAutoTrainCommand


def run_app_command_factory(args):
    return RunAutoTrainAppCommand(
        args.port,
        args.host,
        args.task,
    )


class RunAutoTrainAppCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_app_parser = parser.add_parser(
            "app",
            description="✨ Run AutoTrain app",
        )
        run_app_parser.add_argument(
            "--port",
            type=int,
            default=7860,
            help="Port to run the app on",
            required=False,
        )
        run_app_parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Host to run the app on",
            required=False,
        )
        run_app_parser.add_argument(
            "--task",
            type=str,
            required=False,
            help="Task to run",
        )
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, port, host, task):
        self.port = port
        self.host = host
        self.task = task

    def run(self):
        if os.environ.get("TASK") == "Dreambooth" or self.task == "dreambooth":
            from ..apps.dreambooth import main
        elif os.environ.get("TASK") == "LLM":
            from ..apps.llm import main
        elif os.environ.get("TASK") == "TEXT_CLASSIFICATION":
            from ..apps.text_classification import main
        else:
            from ..apps.main import main

        demo = main()
        demo.queue(concurrency_count=10).launch()
