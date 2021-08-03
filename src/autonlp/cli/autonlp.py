import argparse
import sys

from loguru import logger
from requests import HTTPError

from autonlp.auth import ForbiddenError, NotAuthenticatedError

from .. import __version__
from .benchmark import CreateBenchmarkCommand
from .create_project import CreateProjectCommand
from .estimator import EstimatorCommand
from .evaluate import CreateEvaluationCommand
from .list_identities import ListIdentitiesCommand
from .list_projects import ListProjectsCommand
from .login import LoginCommand
from .metrics import MetricsCommand
from .predict import PredictCommand
from .project_info import ProjectInfoCommand
from .select_identity import SelectidentityCommand
from .train import TrainCommand
from .upload import UploadCommand


def main():
    parser = argparse.ArgumentParser(
        "AutoNLP CLI",
        usage="autonlp <command> [<args>]",
        epilog="For more information about a command, run: `autonlp <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display AutoNLP version", action="store_true")
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    LoginCommand.register_subcommand(commands_parser)
    CreateProjectCommand.register_subcommand(commands_parser)
    ProjectInfoCommand.register_subcommand(commands_parser)
    UploadCommand.register_subcommand(commands_parser)
    TrainCommand.register_subcommand(commands_parser)
    MetricsCommand.register_subcommand(commands_parser)
    ListProjectsCommand.register_subcommand(commands_parser)
    PredictCommand.register_subcommand(commands_parser)
    EstimatorCommand.register_subcommand(commands_parser)
    CreateEvaluationCommand.register_subcommand(commands_parser)
    CreateBenchmarkCommand.register_subcommand(commands_parser)
    ListIdentitiesCommand.register_subcommand(commands_parser)
    SelectidentityCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)

    try:
        command.run()
    except NotAuthenticatedError:
        logger.error("🔐 Not authenticated, please run autonlp login first")
        sys.exit(1)
    except ForbiddenError as err:
        logger.error(f"⛔ Forbidden: {err}")
        sys.exit(1)
    except HTTPError as err:
        status_code = err.response.status_code
        details = err.response.json().get("detail")
        if status_code == 403:
            logger.error(f"⛔ Forbidden: {details}")
        else:
            logger.error(f"❌ [Error: {status_code}] Something failed in AutoNLP backend..")
            if details:
                logger.error(f"{details}")
        sys.exit(1)


if __name__ == "__main__":
    main()
