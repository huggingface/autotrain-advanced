import collections
import json
import os
from typing import List

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import ModelFilter, list_models

from autotrain import app_utils, logger
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.db import AutoTrainDB
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


HF_TOKEN = os.environ.get("HF_TOKEN", None)
_, _, USERS = app_utils.user_validation()
ENABLE_NGC = int(os.environ.get("ENABLE_NGC", 0))
DB = AutoTrainDB("autotrain.db")
AUTOTRAIN_LOCAL = int(os.environ.get("AUTOTRAIN_LOCAL", 0))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "id_column",
    "target_columns",
]

PARAMS = {}
PARAMS["llm"] = LLMTrainingParams(
    target_modules="",
    log="tensorboard",
    mixed_precision="fp16",
    quantization="int4",
    peft=True,
    block_size=1024,
    epochs=3,
).model_dump()

PARAMS["text-classification"] = TextClassificationParams(
    mixed_precision="fp16",
).model_dump()
PARAMS["image-classification"] = ImageClassificationParams(
    mixed_precision="fp16",
).model_dump()
PARAMS["seq2seq"] = Seq2SeqParams(
    mixed_precision="fp16",
).model_dump()
PARAMS["tabular"] = TabularParams().model_dump()
PARAMS["dreambooth"] = DreamBoothTrainingParams(
    prompt="<enter your prompt here>",
    num_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    batch_size=1,
    gradient_accumulation=4,
    lr=1e-4,
).model_dump()


def get_sorted_models(hub_models):
    hub_models = [{"id": m.modelId, "downloads": m.downloads} for m in hub_models if m.private is False]
    hub_models = sorted(hub_models, key=lambda x: x["downloads"], reverse=True)
    hub_models = [m["id"] for m in hub_models]
    return hub_models


def fetch_models():
    _mc = collections.defaultdict(list)
    hub_models1 = list_models(filter="fill-mask", sort="downloads", direction=-1, limit=100, full=False)
    hub_models2 = list_models(filter="text-classification", sort="downloads", direction=-1, limit=100, full=False)
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)
    _mc["text-classification"] = hub_models

    hub_models = list(list_models(filter="text-generation", sort="downloads", direction=-1, limit=100, full=False))
    hub_models = get_sorted_models(hub_models)
    _mc["llm"] = hub_models

    _filter = ModelFilter(
        task="image-classification",
        library="transformers",
    )
    hub_models = list(list_models(filter=_filter, sort="downloads", direction=-1, limit=100, full=False))
    hub_models = get_sorted_models(hub_models)
    _mc["image-classification"] = hub_models

    hub_models = list(list_models(filter="text-to-image", sort="downloads", direction=-1, limit=100, full=False))
    hub_models = get_sorted_models(hub_models)
    _mc["dreambooth"] = hub_models

    hub_models = list(
        list_models(filter="text2text-generation", sort="downloads", direction=-1, limit=100, full=False)
    )
    hub_models = get_sorted_models(hub_models)
    _mc["seq2seq"] = hub_models

    _mc["tabular-classification"] = [
        "xgboost",
        "random_forest",
        "ridge",
        "logistic_regression",
        "svm",
        "extra_trees",
        "adaboost",
        "decision_tree",
        "knn",
    ]

    _mc["tabular-regression"] = [
        "xgboost",
        "random_forest",
        "ridge",
        "svm",
        "extra_trees",
        "adaboost",
        "decision_tree",
        "knn",
    ]
    return _mc


MODEL_CHOICE = fetch_models()

app = FastAPI()
static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)


async def get_request_data(request: Request):
    # Request headers
    headers = dict(request.headers)

    # Request method
    method = request.method

    # Request URL
    url = str(request.url)

    # Client host information
    client_host = request.client.host

    # Request body
    body = await request.body()
    try:
        body = body.decode("utf-8")
    except UnicodeDecodeError:
        body = str(body)

    return {"headers": headers, "method": method, "url": url, "client_host": client_host, "body": body}


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    This function is used to render the HTML file
    :param request:
    :return:
    """
    if HF_TOKEN is None:
        return templates.TemplateResponse("error.html", {"request": request})
    context = {
        "request": request,
        "valid_users": USERS,
        "enable_ngc": ENABLE_NGC,
        "enable_local": AUTOTRAIN_LOCAL,
    }
    return templates.TemplateResponse("index.html", context)


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


@app.get("/model_choices/{task}", response_class=JSONResponse)
async def fetch_model_choices(task: str):
    if task == "text-classification":
        hub_models = MODEL_CHOICE["text-classification"]
    elif task.startswith("llm"):
        hub_models = MODEL_CHOICE["llm"]
    elif task == "image-classification":
        hub_models = MODEL_CHOICE["image-classification"]
    elif task == "dreambooth":
        hub_models = MODEL_CHOICE["dreambooth"]
    elif task == "seq2seq":
        hub_models = MODEL_CHOICE["seq2seq"]
    elif task == "tabular:classification":
        hub_models = MODEL_CHOICE["tabular-classification"]
    elif task == "tabular:regression":
        hub_models = MODEL_CHOICE["tabular-regression"]
    else:
        raise NotImplementedError
    resp = []
    for hub_model in hub_models:
        resp.append({"id": hub_model, "name": hub_model})
    return resp


@app.post("/create_project", response_class=JSONResponse)
async def handle_form(
    project_name: str = Form(...),
    task: str = Form(...),
    base_model: str = Form(...),
    hardware: str = Form(...),
    params: str = Form(...),
    autotrain_user: str = Form(...),
    data_files_training: List[UploadFile] = File(...),
    data_files_valid: List[UploadFile] = File(...),
):
    """
    This function is used to handle the form submission
    """
    logger.info(f"hardware: {hardware}")
    if hardware == "Local":
        running_jobs = DB.get_running_jobs()
        logger.info(f"Running jobs: {running_jobs}")
        if running_jobs:
            for _pid in running_jobs:
                logger.info(f"Killing PID: {_pid}")
                proc_status = app_utils.get_process_status(_pid)
                proc_status = proc_status.strip().lower()
                if proc_status in ("completed", "error", "zombie"):
                    logger.info(f"Process {_pid} is already completed. Skipping...")
                    try:
                        app_utils.kill_process_by_pid(_pid)
                    except Exception as e:
                        logger.info(f"Error while killing process: {e}")
                    DB.delete_job(_pid)

        running_jobs = DB.get_running_jobs()
        if running_jobs:
            logger.info(f"Running jobs: {running_jobs}")
            raise HTTPException(
                status_code=409, detail="Another job is already running. Please wait for it to finish."
            )

    if HF_TOKEN is None:
        return {"error": "HF_TOKEN not set"}

    params = json.loads(params)
    training_files = [f.file for f in data_files_training if f.filename != ""]
    validation_files = [f.file for f in data_files_valid if f.filename != ""] if data_files_valid else []

    if task.startswith("llm"):
        trainer = task.split(":")[1].lower()
        col_map = {"text": "text"}
        if trainer == "reward":
            col_map["rejected_text"] = "rejected_text"
        if trainer == "dpo":
            col_map["prompt"] = "prompt"
            col_map["rejected_text"] = "rejected_text"
        dset = AutoTrainDataset(
            train_data=training_files,
            task="lm_training",
            token=HF_TOKEN,
            project_name=project_name,
            username=autotrain_user,
            column_mapping=col_map,
            valid_data=validation_files,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
    elif task == "text-classification":
        dset = AutoTrainDataset(
            train_data=training_files,
            task="text_multi_class_classification",
            token=HF_TOKEN,
            project_name=project_name,
            username=autotrain_user,
            column_mapping={"text": "text", "label": "target"},
            valid_data=validation_files,
            percent_valid=None,  # TODO: add to UI
            convert_to_class_label=True,
        )
        dset.prepare()
    elif task == "seq2seq":
        dset = AutoTrainDataset(
            train_data=training_files,
            task="seq2seq",
            token=HF_TOKEN,
            project_name=project_name,
            username=autotrain_user,
            column_mapping={"text": "text", "label": "target"},
            valid_data=validation_files,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
    elif task.startswith("tabular"):
        trainer = task.split(":")[1].lower()
        if trainer == "classification":
            task = "tabular_multi_class_classification"
        elif trainer == "regression":
            task = "tabular_single_column_regression"
        else:
            return {"error": "Unknown subtask"}
        dset = AutoTrainDataset(
            train_data=training_files,
            task=task,
            token=HF_TOKEN,
            project_name=project_name,
            username=autotrain_user,
            column_mapping={"id": "id", "label": ["target"]},
            valid_data=validation_files,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
    elif task == "image-classification":
        task = "image_multi_class_classification"
        dset = AutoTrainImageClassificationDataset(
            train_data=training_files[0],
            token=HF_TOKEN,
            project_name=project_name,
            username=autotrain_user,
            valid_data=validation_files[0] if validation_files else None,
            percent_valid=None,  # TODO: add to UI
        )
        dset.prepare()
    elif task == "dreambooth":
        dset = AutoTrainDreamboothDataset(
            concept_images=data_files_training,
            concept_name=params["prompt"],
            token=HF_TOKEN,
            project_name=project_name,
            username=autotrain_user,
            use_v2=True,
        )
        dset.prepare()
    else:
        return {"error": "Task not supported yet"}

    params["model_choice"] = base_model
    params["param_choice"] = "manual"
    params["backend"] = hardware

    jobs_df = pd.DataFrame([params])
    project = AutoTrainProject(dataset=dset, job_params=jobs_df)
    ids = project.create()
    monitor_url = ""
    if hardware == "Local":
        for _id in ids:
            DB.add_job(_id)
        monitor_url = "Monitor your job locally / in logs"
    elif hardware.startswith("EP"):
        monitor_url = f"https://ui.endpoints.huggingface.co/{autotrain_user}/endpoints/{ids[0]}"
    else:
        monitor_url = f"https://hf.co/spaces/{ids[0]}"
    return {"success": "true", "monitor_url": monitor_url}
