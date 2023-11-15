import json
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from autotrain.dataset import AutoTrainDataset
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


HIDDEN_PARAMS = [
    "token",
    "project_name",
    "username",
    "task",
    "backend",
    "repo_id",
    "train_split",
    "valid_split",
    "text_column",
    "rejected_text_column",
    "prompt_text_column",
    "push_to_hub",
    "trainer",
    "model",
    "data_path",
    "image_path",
    "class_image_path",
    "revision",
    "tokenizer",
    "class_prompt",
    "num_class_images",
    "class_labels_conditioning",
    "resume_from_checkpoint",
    "dataloader_num_workers",
    "allow_tf32",
    "prior_generation_precision",
    "local_rank",
    "tokenizer_max_length",
    "rank",
    "xl",
    "checkpoints_total_limit",
    "validation_images",
    "validation_epochs",
    "num_validation_images",
    "validation_prompt",
    "sample_batch_size",
    "log",
    "image_column",
    "target_column",
]

PARAMS = {}
PARAMS["llm"] = LLMTrainingParams(
    target_modules="",
    log="tensorboard",
    fp16=True,
    use_int4=True,
    use_int8=False,
    use_peft=True,
    block_size=1024,
    epochs=3,
).model_dump()

PARAMS["text-classification"] = TextClassificationParams().model_dump()
PARAMS["image-classification"] = ImageClassificationParams().model_dump()
PARAMS["dreambooth"] = DreamBoothTrainingParams(
    prompt="<enter your prompt here>",
    num_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    batch_size=1,
    gradient_accumulation=4,
    lr=1e-4,
).model_dump()
PARAMS["seq2seq"] = Seq2SeqParams().model_dump()
PARAMS["tabular"] = TabularParams().model_dump()

app = FastAPI()
app.mount("/css", StaticFiles(directory="css"), name="css")  # Mounting the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")  # Mounting the static directory
templates = Jinja2Templates(directory="templates")  # Assuming your HTML is in a folder named 'templates'


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    This function is used to render the HTML file
    :param request:
    :return:
    """
    return templates.TemplateResponse("index.html", {"request": request})  # The form.html is your saved html file


@app.get("/params/{task}", response_class=JSONResponse)
async def fetch_params(task: str):
    """
    This function is used to fetch the parameters for a given task
    :param task: str
    :return: JSONResponse
    """
    logger.info(f"Task: {task}")
    if task.startswith("llm"):
        trainer = task.split(":")[1].lower()
        task = task.split(":")[0].lower()

    if task.startswith("tabular"):
        task = "tabular"

    if task in PARAMS:
        task_params = PARAMS[task]
        task_params = {k: v for k, v in task_params.items() if k not in HIDDEN_PARAMS}
        if task == "llm":
            more_hidden_params = []
            if trainer in ("sft", "reward"):
                more_hidden_params = [
                    "model_ref",
                    "dpo_beta",
                    "add_eos_token",
                ]
            elif trainer == "generic":
                more_hidden_params = [
                    "model_ref",
                    "dpo_beta",
                ]
            elif trainer == "dpo":
                more_hidden_params = [
                    "add_eos_token",
                ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "dreambooth":
            more_hidden_params = [
                "epochs",
                "logging",
                "bf16",
            ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        return task_params
    return {"error": "Task not found"}


@app.post("/create_project", response_class=JSONResponse)
async def handle_form(
    project_name: str = Form(...),
    task: str = Form(...),
    token: str = Form(...),
    username: str = Form(...),
    base_model: str = Form(...),
    hardware: str = Form(...),
    data_files_training: List[UploadFile] = File(...),
    data_files_valid: Optional[List[UploadFile]] = File(None),
    params: UploadFile = File(...),
):
    """
    This function is used to handle the form submission
    """
    data = {
        "project_name": project_name,
        "token": token,
        "task": task,
        "base_model": base_model,
        "username": username,
        "hardware": hardware,
        "params": params,
    }
    logger.info(data)
    # check if data_files has train.csv and valid.csv
    # check if params is a valid JSON

    if task.startswith("llm") or task in ("text-classification", "seq2seq"):
        training_files = [f.file for f in data_files_training]
        validation_files = [f.file for f in data_files_valid] if data_files_valid else []

        dset = AutoTrainDataset(
            train_data=training_files,
            task="lm_training",
            token=token,
            project_name=project_name,
            username=username,
            column_mapping={"text": "text"},
            valid_data=validation_files,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()

    else:
        return {"error": "Task not supported yet"}

    params = params.file.read()
    params = json.loads(params)
    params["model_choice"] = base_model
    params["param_choice"] = "manual"
    params["backend"] = hardware

    jobs_df = pd.DataFrame([params])
    logger.info(jobs_df)
    project = AutoTrainProject(dataset=dset, job_params=jobs_df)
    ids = project.create()
    logger.info(ids)
    return {"success": "true", "space_ids": ids}
