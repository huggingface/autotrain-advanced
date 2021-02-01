from argparse import ArgumentParser

from . import BaseAutoNLPCommand


def login_command_factory(args):
    return LoginCommand(api_key=args.api_key)


class LoginCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login")
        login_parser.add_argument("--api-key", type=str, required=True, help="API key")
        login_parser.set_defaults(func=login_command_factory)

    def __init__(self, api_key: str):
        self._api_key = api_key

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        client.login(token=self._api_key)
