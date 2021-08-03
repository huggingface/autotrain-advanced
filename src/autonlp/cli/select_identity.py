from argparse import ArgumentParser

from loguru import logger

from ..auth import login_from_conf, select_identity
from . import BaseAutoNLPCommand


def select_identity_command_factory(args):
    return SelectidentityCommand(args.identity)


class SelectidentityCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        select_identity_parser = parser.add_parser(
            "select_identity", description="🎭 Sets the default identity for AutoNLP"
        )
        select_identity_parser.add_argument(
            "identity",
            type=str,
            required=True,
            help="The new default identity. Run autonlp list_identities to visualize your options.",
        )
        select_identity_parser.set_defaults(func=select_identity_command_factory)

    def __init__(self, identity: str):
        self._identity = identity

    def run(self, identity: str):
        logger.info("Fetching identities...")
        login_info = login_from_conf()
        if self._identity not in [choice.name for choice in login_info["identities"]]:
            print(f"🛑 Cannot impersonate '{self._identity}'.")
            print("Available choices are:")
            for choice in login_info["identities"]:
                print(f"  > '{choice.name}'")

        select_identity(new_identity=self._identity)
        print(f"✅ Successfully selected '{self._identity}' as the default identity")
