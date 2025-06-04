import argparse
import json

from autotrain import logger
from autotrain.trainers.common import monitor, pause_space
from autotrain.trainers.generic import utils
from autotrain.trainers.generic.params import GenericParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


@monitor
def run(config):
    """
    Executes a series of operations based on the provided configuration.

    This function performs the following steps:
    1. Converts the configuration dictionary to a GenericParams object if necessary.
    2. Downloads the data repository specified in the configuration.
    3. Uninstalls any existing requirements specified in the configuration.
    4. Installs the necessary requirements specified in the configuration.
    5. Runs a command specified in the configuration.
    6. Pauses the space as specified in the configuration.

    Args:
        config (dict or GenericParams): The configuration for the operations to be performed.
    """
    if isinstance(config, dict):
        config = GenericParams(**config)

    # download the data repo
    logger.info("Downloading data repo...")
    utils.pull_dataset_repo(config)

    logger.info("Unintalling requirements...")
    utils.uninstall_requirements(config)

    # install the requirements
    logger.info("Installing requirements...")
    utils.install_requirements(config)

    # run the command
    logger.info("Running command...")
    utils.run_command(config)

    pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    _config = json.load(open(args.config))
    _config = GenericParams(**_config)
    run(_config)
