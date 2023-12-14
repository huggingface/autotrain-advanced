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
