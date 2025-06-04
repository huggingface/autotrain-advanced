import os
import signal
import subprocess
import sys
import threading
from argparse import ArgumentParser

from autotrain import logger

from . import BaseAutoTrainCommand


def handle_output(stream, log_file):
    """
    Continuously reads lines from a given stream and writes them to both
    standard output and a log file until the stream is exhausted.

    Args:
        stream (io.TextIOBase): The input stream to read lines from.
        log_file (io.TextIOBase): The log file to write lines to.

    Returns:
        None
    """
    while True:
        line = stream.readline()
        if not line:
            break
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def run_app_command_factory(args):
    return RunAutoTrainAppCommand(args.port, args.host, args.share, args.workers, args.colab)


class RunAutoTrainAppCommand(BaseAutoTrainCommand):
    """
    Command to run the AutoTrain application.

    This command sets up and runs the AutoTrain application with the specified
    configuration options such as port, host, number of workers, and sharing options.

    Methods
    -------
    register_subcommand(parser: ArgumentParser):
        Registers the subcommand and its arguments to the provided parser.

    __init__(port: int, host: str, share: bool, workers: int, colab: bool):
        Initializes the command with the specified parameters.

    run():
        Executes the command to run the AutoTrain application. Handles different
        modes such as running in Colab or sharing via ngrok.
    """

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
            "--workers",
            type=int,
            default=1,
            help="Number of workers to run the app with",
            required=False,
        )
        run_app_parser.add_argument(
            "--share",
            action="store_true",
            help="Share the app on ngrok",
            required=False,
        )
        run_app_parser.add_argument(
            "--colab",
            action="store_true",
            help="Use app in colab",
            required=False,
        )
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, port, host, share, workers, colab):
        self.port = port
        self.host = host
        self.share = share
        self.workers = workers
        self.colab = colab

    def run(self):
        if self.colab:
            from IPython.display import display

            from autotrain.app.colab import colab_app

            elements = colab_app()
            display(elements)
            return

        if self.share:
            from pyngrok import ngrok

            os.system(f"fuser -n tcp -k {self.port}")
            authtoken = os.environ.get("NGROK_AUTH_TOKEN", "")
            if authtoken.strip() == "":
                logger.info("NGROK_AUTH_TOKEN not set")
                raise ValueError("NGROK_AUTH_TOKEN not set. Please set it!")

            ngrok.set_auth_token(authtoken)
            active_tunnels = ngrok.get_tunnels()
            for tunnel in active_tunnels:
                public_url = tunnel.public_url
                ngrok.disconnect(public_url)
            url = ngrok.connect(addr=self.port, bind_tls=True)
            logger.info(f"AutoTrain Public URL: {url}")
            logger.info("Please wait for the app to load...")

        command = f"uvicorn autotrain.app.app:app --host {self.host} --port {self.port}"
        command += f" --workers {self.workers}"

        with open("autotrain.log", "w", encoding="utf-8") as log_file:
            if sys.platform == "win32":
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, bufsize=1
                )

            else:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )

            output_thread = threading.Thread(target=handle_output, args=(process.stdout, log_file))
            output_thread.start()

            try:
                process.wait()
                output_thread.join()
            except KeyboardInterrupt:
                logger.warning("Attempting to terminate the process...")
                if sys.platform == "win32":
                    process.terminate()
                else:
                    # If user cancels (Ctrl+C), terminate the subprocess
                    # Use os.killpg to send SIGTERM to the process group, ensuring all child processes are killed
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                logger.info("Process terminated by user")
