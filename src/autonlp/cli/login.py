from argparse import ArgumentParser

from ..utils import RED_TAG as RED
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand


def login_command_factory(args):
    return LoginCommand(api_key=args.api_key)


class LoginCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login", description="🔐 Logs you in AutoNLP!")
        login_parser.add_argument(
            "--api-key",
            type=str,
            required=True,
            help="Your API key. You can find it under you user profile settings on huggingface.co!",
        )
        login_parser.set_defaults(func=login_command_factory)

    def __init__(self, api_key: str):
        self._api_key = api_key

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        client.login(token=self._api_key)

        print(f"Welcome to 🤗 AutoNLP! Start by creating a project: {RED}autonlp create_project{RST}")
