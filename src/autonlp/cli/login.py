from argparse import ArgumentParser

from . import BaseAutoNLPCommand
from loguru import logger


def login_command_factory(args):
    return LoginCommand(args.username)


class LoginCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login")
        login_parser.add_argument("--username", type=str, default=None, required=True, help="Username")
        login_parser.set_defaults(func=login_command_factory)

    def __init__(self, username: str):
        self._username = username

    def run(self):
        from ..autonlp import AutoNLP

        logger.info(f"Logging in using username: {self._username}")
        client = AutoNLP(username=self._username)
        client.login()
