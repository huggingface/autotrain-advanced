import json
import os
import subprocess
from typing import Dict, List, Optional

from datasets import ClassLabel, Dataset, DatasetDict
from huggingface_hub import HfApi, metadata_update

from autotrain import logger


def convert_text_dataset_to_hf(
    task, train: List[Dict[str, str]], valid: Optional[List[Dict[str, str]]] = None
) -> Dataset:
    if task == "text-classification":
        for item in train:
            item["target"] = item["target"].lower().strip()
        label_names = list(set([item["target"] for item in train]))
        logger.info(f"Label names: {label_names}")

    dataset = Dataset.from_list(train)

    if task == "text-classification":
        dataset = dataset.cast_column("target", ClassLabel(names=label_names))

    ddict = {"train": dataset}
    if valid is not None:
        valid_dataset = Dataset.from_list(valid)
        if task == "text-classification":
            for item in valid:
                item["target"] = item["target"].lower().strip()
            valid_dataset = valid_dataset.cast_column("target", ClassLabel(names=label_names))
        ddict["validation"] = valid_dataset
    dataset = DatasetDict(ddict)
    return dataset


def push_data_to_hub(params, dataset) -> str:
    if params.username is None:
        raise ValueError("Username is required for pushing data to Hugging Face Hub.")
    if params.token is None:
        raise ValueError("Token is required for pushing data to Hugging Face Hub.")

    repo_id = f"{params.username}/{params.project_name}"
    dataset.push_to_hub(repo_id, token=params.token, private=True)

    if os.path.exists(f"{params.project_name}/gen_params.json"):
        gen_params = json.load(open(f"{params.project_name}/gen_params.json"))
        if "token" in gen_params:
            gen_params.pop("token")

        if "api" in gen_params:
            gen_params.pop("api")

        if "api_key" in gen_params:
            gen_params.pop("api_key")

        json.dump(
            gen_params,
            open(f"{params.project_name}/gen_params.json", "w"),
            indent=4,
        )

    api = HfApi(token=params.token)
    if os.path.exists(f"{params.project_name}/gen_params.json"):
        api.upload_file(
            path_or_fileobj=f"{params.project_name}/gen_params.json",
            repo_id=f"{params.username}/{params.project_name}",
            repo_type="dataset",
            path_in_repo="gen_params.json",
        )
    metadata = {
        "tags": [
            "autotrain",
            "gen",
            "synthetic",
        ]
    }
    metadata_update(repo_id, metadata, token=params.token, repo_type="dataset", overwrite=True)
    return repo_id


def train(params):
    if params.training_config is None:
        logger.info("No training configuration provided. Skipping training...")
        return
    cmd = f"autotrain --config {params.training_config}"
    logger.info(f"Running AutoTrain: {cmd}")
    cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    process = subprocess.Popen(cmd, env=env)
    process.wait()
    return process.pid
