import argparse

from .. import __version__
from .run_api import RunAutoTrainAPICommand
from .run_app import RunAutoTrainAppCommand
from .run_dreambooth import RunAutoTrainDreamboothCommand
from .run_image_classification import RunAutoTrainImageClassificationCommand
from .run_llm import RunAutoTrainLLMCommand
from .run_seq2seq import RunAutoTrainSeq2SeqCommand
from .run_setup import RunSetupCommand
from .run_spacerunner import RunAutoTrainSpaceRunnerCommand
from .run_tabular import RunAutoTrainTabularCommand
from .run_text_classification import RunAutoTrainTextClassificationCommand
from .run_token_classification import RunAutoTrainTokenClassificationCommand


def main():
    parser = argparse.ArgumentParser(
        "AutoTrain advanced CLI",
        usage="autotrain <command> [<args>]",
        epilog="For more information about a command, run: `autotrain <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display AutoTrain version", action="store_true")
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    RunAutoTrainAppCommand.register_subcommand(commands_parser)
    RunAutoTrainLLMCommand.register_subcommand(commands_parser)
    RunSetupCommand.register_subcommand(commands_parser)
    RunAutoTrainDreamboothCommand.register_subcommand(commands_parser)
    RunAutoTrainAPICommand.register_subcommand(commands_parser)
    RunAutoTrainTextClassificationCommand.register_subcommand(commands_parser)
    RunAutoTrainImageClassificationCommand.register_subcommand(commands_parser)
    RunAutoTrainTabularCommand.register_subcommand(commands_parser)
    RunAutoTrainSpaceRunnerCommand.register_subcommand(commands_parser)
    RunAutoTrainSeq2SeqCommand.register_subcommand(commands_parser)
    RunAutoTrainTokenClassificationCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
