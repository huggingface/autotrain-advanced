from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict
from huggingface_hub import metadata_update


def convert_text_dataset_to_hf(train: List[Dict[str, str]], valid: Optional[List[Dict[str, str]]] = None) -> Dataset:
    dataset = Dataset.from_list(train)
    ddict = {"train": dataset}
    if valid is not None:
        valid_dataset = Dataset.from_list(valid)
        ddict["validation"] = valid_dataset
    dataset = DatasetDict(ddict)
    return dataset


def push_data_to_hub(dataset: Dataset, dataset_name: str, username: str, token: Optional[str] = None) -> str:
    if username is None:
        raise ValueError("Username is required for pushing data to Hugging Face Hub.")
    if token is None:
        raise ValueError("Token is required for pushing data to Hugging Face Hub.")
    repo_id = f"{username}/{dataset_name}"
    dataset.push_to_hub(repo_id, token=token, private=True)
    metadata = {
        "tags": [
            "autotrain",
            "gen",
            "synthetic",
        ]
    }
    metadata_update(repo_id, metadata, token=token, repo_type="dataset", overwrite=True)
    return repo_id
