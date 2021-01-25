from argparse import ArgumentParser

from loguru import logger

from . import BaseAutoNLPCommand


def login_command_factory(args):
    return LoginCommand(username=args.username, api_key=args.api_key)


class LoginCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login")
        login_parser.add_argument("--username", type=str, default=None, required=True, help="Username")
        login_parser.add_argument("--api-key", type="str", required=True, help="API key")
        login_parser.set_defaults(func=login_command_factory)

    def __init__(self, username: str, api_key: str):
        self._username = username
        self._api_key = api_key

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Logging in using username: {self._username}")
        client = AutoNLP()
        client.login(username=self._username, api_key=self._api_key)
