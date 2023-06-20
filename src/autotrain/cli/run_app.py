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
        from ..app import main

        demo = main()
        demo.launch()
        # dirname = os.path.dirname(__file__)
        # filename = os.path.join(dirname, "..", "app.py")
        # cmd = [
        #     "streamlit",
        #     "run",
        #     filename,
        #     "--browser.gatherUsageStats",
        #     "false",
        #     "--browser.serverAddress",
        #     self.host,
        #     "--server.port",
        #     str(self.port),
        #     "--theme.base",
        #     "light",
        #     "--server.maxUploadSize",
        #     "10000",
        # ]
        # if "SPACE_ID" in os.environ:
        #     cmd.extend(["--server.enableXsrfProtection", "false"])
        #     cmd.extend(["--server.headless", "true"])
        #     cmd.extend(["--server.enableCORS", "false"])
        #     cmd.extend(["--server.fileWatcherType", "none"])

        # if self.task:
        #     cmd.extend(["--", "--task", self.task])

        # proc = subprocess.Popen(
        #     cmd,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT,
        #     shell=False,
        #     universal_newlines=True,
        #     bufsize=1,
        # )
        # with proc as p:
        #     try:
        #         for line in p.stdout:
        #             print(line, end="")
        #     except KeyboardInterrupt:
        #         print("Killing app")
        #         p.kill()
        #         p.wait()
        #         raise
