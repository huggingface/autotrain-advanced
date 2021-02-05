import argparse

from .create_project import CreateProjectCommand
from .login import LoginCommand
from .metrics import MetricsCommand
from .project_info import ProjectInfoCommand
from .train import TrainCommand
from .upload import UploadCommand


def main():
    parser = argparse.ArgumentParser("AutoNLP CLI", usage="autonlp <command> [<args>]")
    commands_parser = parser.add_subparsers(help="autonlp command helpers")

    # Register commands
    LoginCommand.register_subcommand(commands_parser)
    CreateProjectCommand.register_subcommand(commands_parser)
    ProjectInfoCommand.register_subcommand(commands_parser)
    UploadCommand.register_subcommand(commands_parser)
    TrainCommand.register_subcommand(commands_parser)
    MetricsCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
