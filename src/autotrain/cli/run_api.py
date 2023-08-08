from argparse import ArgumentParser

from . import BaseAutoTrainCommand


def run_api_command_factory(args):
    return RunAutoTrainAPICommand(
        args.port,
        args.host,
        args.task,
    )


class RunAutoTrainAPICommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_api_parser = parser.add_parser(
            "api",
            description="✨ Run AutoTrain API",
        )
        run_api_parser.add_argument(
            "--port",
            type=int,
            default=7860,
            help="Port to run the api on",
            required=False,
        )
        run_api_parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Host to run the api on",
            required=False,
        )
        run_api_parser.add_argument(
            "--task",
            type=str,
            required=False,
            help="Task to run",
        )
        run_api_parser.set_defaults(func=run_api_command_factory)

    def __init__(self, port, host, task):
        self.port = port
        self.host = host
        self.task = task

    def run(self):
        import uvicorn

        from autotrain.api import api

        uvicorn.run(api, host=self.host, port=self.port)
