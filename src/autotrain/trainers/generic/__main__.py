import argparse
import json
import os

from huggingface_hub import HfApi

from autotrain import logger
from autotrain.trainers.generic import utils
from autotrain.trainers.generic.params import GenericParams
from autotrain.utils import monitor


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

    # install the requirements
    logger.info("Installing requirements...")
    utils.install_requirements(config)

    # run the command
    logger.info("Running command...")
    utils.run_command(config)

    if "SPACE_ID" in os.environ:
        # shut down the space
        logger.info("Pausing space...")
        api = HfApi(token=config.token)
        api.pause_space(repo_id=os.environ["SPACE_ID"])

    if "ENDPOINT_ID" in os.environ:
        # shut down the endpoint
        logger.info("Pausing endpoint...")
        utils.pause_endpoint(config)


if __name__ == "__main__":
    args = parse_args()
    _config = json.load(open(args.config))
    _config = GenericParams(**_config)
    run(_config)
