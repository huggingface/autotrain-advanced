import os
from uuid import uuid4

from datasets import load_dataset

from autotrain.dataset import AutoTrainDataset
from autotrain.project import Project


RANDOM_ID = str(uuid4())
DATASET = "imdb"
PROJECT_NAME = f"imdb_{RANDOM_ID}"
TASK = "text_binary_classification"
MODEL = "bert-base-uncased"

USERNAME = os.environ["AUTOTRAIN_USERNAME"]
TOKEN = os.environ["HF_TOKEN"]


if __name__ == "__main__":
    dataset = load_dataset(DATASET)
    train = dataset["train"]
    validation = dataset["test"]

    # convert to pandas dataframe
    train_df = train.to_pandas()
    validation_df = validation.to_pandas()

    # prepare dataset for AutoTrain
    dset = AutoTrainDataset(
        train_data=[train_df],
        valid_data=[validation_df],
        task=TASK,
        token=TOKEN,
        project_name=PROJECT_NAME,
        username=USERNAME,
        column_mapping={"text": "text", "label": "label"},
        percent_valid=None,
    )
    dset.prepare()

    #
    # How to get params for a task:
    #
    # from autotrain.params import Params
    # params = Params(task=TASK, training_type="hub_model").get()
    # print(params) to get full list of params for the task

    # define params in proper format
    job1 = {
        "task": TASK,
        "learning_rate": 1e-5,
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "epochs": 5,
    }

    job2 = {
        "task": TASK,
        "learning_rate": 3e-5,
        "optimizer": "adamw_torch",
        "scheduler": "cosine",
        "epochs": 5,
    }

    job3 = {
        "task": TASK,
        "learning_rate": 5e-5,
        "optimizer": "sgd",
        "scheduler": "cosine",
        "epochs": 5,
    }

    jobs = [job1, job2, job3]
    project = Project(dataset=dset, hub_model=MODEL, job_params=jobs)
    project_id = project.create()
    project.approve(project_id)
