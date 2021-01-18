import argparse
from .login import LoginCommand
from .create_project import CreateProjectCommand


def main():
    parser = argparse.ArgumentParser("AutoNLP CLI", usage="autonlp <command> [<args>]")
    commands_parser = parser.add_subparsers(help="autonlp command helpers")

    # Register commands
    LoginCommand.register_subcommand(commands_parser)
    CreateProjectCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
