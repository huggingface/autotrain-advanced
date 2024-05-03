import argparse

from autotrain import __version__, logger
from autotrain.cli.run_api import RunAutoTrainAPICommand
from autotrain.cli.run_app import RunAutoTrainAppCommand
from autotrain.cli.run_dreambooth import RunAutoTrainDreamboothCommand
from autotrain.cli.run_image_classification import RunAutoTrainImageClassificationCommand
from autotrain.cli.run_llm import RunAutoTrainLLMCommand
from autotrain.cli.run_seq2seq import RunAutoTrainSeq2SeqCommand
from autotrain.cli.run_setup import RunSetupCommand
from autotrain.cli.run_spacerunner import RunAutoTrainSpaceRunnerCommand
from autotrain.cli.run_tabular import RunAutoTrainTabularCommand
from autotrain.cli.run_text_classification import RunAutoTrainTextClassificationCommand
from autotrain.cli.run_text_regression import RunAutoTrainTextRegressionCommand
from autotrain.cli.run_token_classification import RunAutoTrainTokenClassificationCommand
from autotrain.cli.run_tools import RunAutoTrainToolsCommand
from autotrain.parser import AutoTrainConfigParser


def main():
    parser = argparse.ArgumentParser(
        "AutoTrain advanced CLI",
        usage="autotrain <command> [<args>]",
        epilog="For more information about a command, run: `autotrain <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display AutoTrain version", action="store_true")
    parser.add_argument("--config", help="Optional configuration file", type=str)
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
    RunAutoTrainToolsCommand.register_subcommand(commands_parser)
    RunAutoTrainTextRegressionCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if args.config:
        logger.info(f"Using AutoTrain configuration: {args.config}")
        cp = AutoTrainConfigParser(args.config)
        cp.run()
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
