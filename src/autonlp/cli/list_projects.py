from argparse import ArgumentParser

from loguru import logger
from prettytable import PrettyTable

from . import BaseAutoNLPCommand


def list_projects_command_factory(args):
    return ListProjectsCommand(args.username)


class ListProjectsCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        list_projects_parser = parser.add_parser("list_projects")
        list_projects_parser.add_argument(
            "--username",
            type=str,
            default=None,
            required=False,
            help="Username of the owner of the projects, defaults to you.",
        )
        list_projects_parser.set_defaults(func=list_projects_command_factory)

    def __init__(self, username=None):
        self._username = username

    def run(self):
        from ..autonlp import AutoNLP

        logger.info("Fetching projects...")
        client = AutoNLP()
        projects = client.list_projects(username=self._username)
        if projects:
            table = PrettyTable(field_names=["ID", "Name", "Owner", "Task", "Created at", "Last Update"])
            table.add_rows(
                [
                    [
                        proj.proj_id,
                        proj.name,
                        proj.user,
                        proj.task.title().replace("_", " "),
                        proj.created_at.strftime("%Y-%m-%d %H:%M Z"),
                        proj.updated_at.strftime("%Y-%m-%d %H:%M Z"),
                    ]
                    for proj in projects
                ]
            )
            print(table)
        else:
            print("🚫 No projects yet! Create one with the create_project command to see something here")
