import os
from argparse import ArgumentParser

from autotrain import logger

from . import BaseAutoTrainCommand


def run_app_command_factory(args):
    return RunAutoTrainAppCommand(args.port, args.host, args.share)


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
            "--share",
            action="store_true",
            help="Share the app on ngrok",
            required=False,
        )
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, port, host, share):
        self.port = port
        self.host = host
        self.share = share

    def run(self):
        import uvicorn
        from pyngrok import ngrok

        from autotrain.app import app

        if self.share:
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
            logger.info("***")

        uvicorn.run(app, host=self.host, port=self.port)
