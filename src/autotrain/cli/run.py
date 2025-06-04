import argparse
from abc import ABC, abstractmethod


class AutoTrainCLI(ABC):
    """
    Base class for all AutoTrain CLI commands.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the CLI command.

        Args:
            args (argparse.Namespace): Command line arguments
        """
        self.args = args

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: argparse._SubParsersAction):
        """
        Register the subcommand with the argument parser.

        Args:
            parser (argparse._SubParsersAction): The subparser to register with
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the command.
        """
        pass 