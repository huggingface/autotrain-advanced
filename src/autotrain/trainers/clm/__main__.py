import argparse
import json

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common import monitor


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    if config.trainer == "default":
        from autotrain.trainers.clm.train_clm_default import train as train_default

        train_default(config)

    elif config.trainer == "sft":
        from autotrain.trainers.clm.train_clm_sft import train as train_sft

        train_sft(config)

    elif config.trainer == "reward":
        from autotrain.trainers.clm.train_clm_reward import train as train_reward

        train_reward(config)

    elif config.trainer == "dpo":
        from autotrain.trainers.clm.train_clm_dpo import train as train_dpo

        train_dpo(config)

    elif config.trainer == "orpo":
        from autotrain.trainers.clm.train_clm_orpo import train as train_orpo

        train_orpo(config)

    else:
        raise ValueError(f"trainer `{config.trainer}` not supported")


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = LLMTrainingParams(**training_config)
    train(_config)
