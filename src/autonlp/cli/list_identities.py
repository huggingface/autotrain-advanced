from argparse import ArgumentParser

from loguru import logger
from prettytable import PrettyTable

from ..auth import login_from_conf
from ..utils import GREEN_TAG as GREEN
from ..utils import RESET_TAG as RST
from . import BaseAutoNLPCommand


def list_identities_command_factory(args):
    return ListIdentitiesCommand()


class ListIdentitiesCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        list_identities_parser = parser.add_parser(
            "list_identities", description="🎭 Lists AutoNLP identities the current user can impersonate"
        )
        list_identities_parser.set_defaults(func=list_identities_command_factory)

    def run(self):
        logger.info("Fetching identities...")
        login_info = login_from_conf()
        table = PrettyTable(field_names=["Selected", "Name", "Full name", "Organization"])
        table.add_rows(
            [
                [
                    f"{GREEN}X{RST}" if identity["name"] == login_info["selected_identity"] else " ",
                    identity["name"],
                    identity["full_name"],
                    identity["is_org"],
                ]
                for identity in login_info["identities"]
            ]
        )
        print(table)
