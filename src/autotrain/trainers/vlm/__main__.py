import argparse
import json

from autotrain.trainers.common import monitor
from autotrain.trainers.vlm import utils
from autotrain.trainers.vlm.params import VLMTrainingParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = VLMTrainingParams(**config)

    if not utils.check_model_support(config):
        raise ValueError(f"model `{config.model}` not supported")

    if config.trainer in ("vqa", "captioning"):
        from autotrain.trainers.vlm.train_vlm_generic import train as train_generic

        train_generic(config)

    else:
        raise ValueError(f"trainer `{config.trainer}` not supported")


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = VLMTrainingParams(**training_config)
    train(_config)
