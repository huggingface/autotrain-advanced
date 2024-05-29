import json
import os
import signal
import sys
import time
from typing import List

import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import repo_exists
from nvitop import Device

from autotrain import __version__, logger
from autotrain.app.db import AutoTrainDB
from autotrain.app.models import fetch_models
from autotrain.app.params import AppParams, get_task_params
from autotrain.app.utils import get_running_jobs, get_user_and_orgs, kill_process_by_pid, token_verification
from autotrain.dataset import (
    AutoTrainDataset,
    AutoTrainDreamboothDataset,
    AutoTrainImageClassificationDataset,
    AutoTrainObjectDetectionDataset,
)
from autotrain.help import get_app_help
from autotrain.project import AutoTrainProject


logger.info("Starting AutoTrain...")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
IS_RUNNING_IN_SPACE = "SPACE_ID" in os.environ
ENABLE_NGC = int(os.environ.get("ENABLE_NGC", 0))
ENABLE_NVCF = int(os.environ.get("ENABLE_NVCF", 0))
AUTOTRAIN_LOCAL = int(os.environ.get("AUTOTRAIN_LOCAL", 1))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = AutoTrainDB("autotrain.db")
MODEL_CHOICE = fetch_models()

ui_router = APIRouter()
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)

UI_PARAMS = {
    "mixed_precision": {
        "type": "dropdown",
        "label": "Mixed precision",
        "options": ["fp16", "bf16", "none"],
    },
    "optimizer": {
        "type": "dropdown",
        "label": "Optimizer",
        "options": ["adamw_torch", "adamw", "adam", "sgd"],
    },
    "scheduler": {
        "type": "dropdown",
        "label": "Scheduler",
        "options": ["linear", "cosine", "cosine_warmup", "constant"],
    },
    "evaluation_strategy": {
        "type": "dropdown",
        "label": "Evaluation strategy",
        "options": ["epoch", "steps"],
    },
    "logging_steps": {
        "type": "number",
        "label": "Logging steps",
    },
    "save_total_limit": {
        "type": "number",
        "label": "Save total limit",
    },
    "auto_find_batch_size": {
        "type": "dropdown",
        "label": "Auto find batch size",
        "options": [True, False],
    },
    "warmup_ratio": {
        "type": "number",
        "label": "Warmup proportion",
    },
    "max_grad_norm": {
        "type": "number",
        "label": "Max grad norm",
    },
    "weight_decay": {
        "type": "number",
        "label": "Weight decay",
    },
    "epochs": {
        "type": "number",
        "label": "Epochs",
    },
    "batch_size": {
        "type": "number",
        "label": "Batch size",
    },
    "lr": {
        "type": "number",
        "label": "Learning rate",
    },
    "seed": {
        "type": "number",
        "label": "Seed",
    },
    "gradient_accumulation": {
        "type": "number",
        "label": "Gradient accumulation",
    },
    "block_size": {
        "type": "number",
        "label": "Block size",
    },
    "model_max_length": {
        "type": "number",
        "label": "Model max length",
    },
    "add_eos_token": {
        "type": "dropdown",
        "label": "Add EOS token",
        "options": [True, False],
    },
    "disable_gradient_checkpointing": {
        "type": "dropdown",
        "label": "Disable GC",
        "options": [True, False],
    },
    "use_flash_attention_2": {
        "type": "dropdown",
        "label": "Use flash attention",
        "options": [True, False],
    },
    "log": {
        "type": "dropdown",
        "label": "Logging",
        "options": ["tensorboard", "none"],
    },
    "quantization": {
        "type": "dropdown",
        "label": "Quantization",
        "options": ["int4", "int8", "none"],
    },
    "target_modules": {
        "type": "string",
        "label": "Target modules",
    },
    "merge_adapter": {
        "type": "dropdown",
        "label": "Merge adapter",
        "options": [True, False],
    },
    "peft": {
        "type": "dropdown",
        "label": "PEFT/LoRA",
        "options": [True, False],
    },
    "lora_r": {
        "type": "number",
        "label": "Lora r",
    },
    "lora_alpha": {
        "type": "number",
        "label": "Lora alpha",
    },
    "lora_dropout": {
        "type": "number",
        "label": "Lora dropout",
    },
    "model_ref": {
        "type": "string",
        "label": "Reference model",
    },
    "dpo_beta": {
        "type": "number",
        "label": "DPO beta",
    },
    "max_prompt_length": {
        "type": "number",
        "label": "Prompt length",
    },
    "max_completion_length": {
        "type": "number",
        "label": "Completion length",
    },
    "chat_template": {
        "type": "dropdown",
        "label": "Chat template",
        "options": ["none", "zephyr", "chatml", "tokenizer"],
    },
    "padding": {
        "type": "dropdown",
        "label": "Padding side",
        "options": ["right", "left", "none"],
    },
    "max_seq_length": {
        "type": "number",
        "label": "Max sequence length",
    },
    "early_stopping_patience": {
        "type": "number",
        "label": "Early stopping patience",
    },
    "early_stopping_threshold": {
        "type": "number",
        "label": "Early stopping threshold",
    },
    "max_target_length": {
        "type": "number",
        "label": "Max target length",
    },
    "categorical_columns": {
        "type": "string",
        "label": "Categorical columns",
    },
    "numerical_columns": {
        "type": "string",
        "label": "Numerical columns",
    },
    "num_trials": {
        "type": "number",
        "label": "Number of trials",
    },
    "time_limit": {
        "type": "number",
        "label": "Time limit",
    },
    "categorical_imputer": {
        "type": "dropdown",
        "label": "Categorical imputer",
        "options": ["most_frequent", "none"],
    },
    "numerical_imputer": {
        "type": "dropdown",
        "label": "Numerical imputer",
        "options": ["mean", "median", "none"],
    },
    "numeric_scaler": {
        "type": "dropdown",
        "label": "Numeric scaler",
        "options": ["standard", "minmax", "maxabs", "robust", "none"],
    },
    "vae_model": {
        "type": "string",
        "label": "VAE model",
    },
    "prompt": {
        "type": "string",
        "label": "Prompt",
    },
    "resolution": {
        "type": "number",
        "label": "Resolution",
    },
    "num_steps": {
        "type": "number",
        "label": "Number of steps",
    },
    "checkpointing_steps": {
        "type": "number",
        "label": "Checkpointing steps",
    },
    "use_8bit_adam": {
        "type": "dropdown",
        "label": "Use 8-bit Adam",
        "options": [True, False],
    },
    "xformers": {
        "type": "dropdown",
        "label": "xFormers",
        "options": [True, False],
    },
    "image_square_size": {
        "type": "number",
        "label": "Image square size",
    },
}


def graceful_exit(signum, frame):
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)


logger.info("AutoTrain started successfully")


def user_authentication(request: Request):
    # priority: hf_token env var > oauth_info in session > token in bearer header
    # if "oauth_info" in request.session:
    if HF_TOKEN is not None:
        try:
            _ = token_verification(token=os.environ.get("HF_TOKEN"))
            return HF_TOKEN
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            if IS_RUNNING_IN_SPACE:
                return templates.TemplateResponse("login.html", {"request": request})
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token: HF_TOKEN",
                )

    if IS_RUNNING_IN_SPACE and "oauth_info" in request.session:
        try:
            _ = token_verification(token=request.session["oauth_info"]["access_token"])
            return request.session["oauth_info"]["access_token"]
        except Exception as e:
            request.session.pop("oauth_info", None)
            logger.error(f"Failed to verify token: {e}")
            return templates.TemplateResponse("login.html", {"request": request})

    if IS_RUNNING_IN_SPACE:
        return templates.TemplateResponse("login.html", {"request": request})

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )


@ui_router.get("/", response_class=HTMLResponse)
async def load_index(request: Request, token: str = Depends(user_authentication)):
    """
    This function is used to load the index page
    :return: HTMLResponse
    """
    if os.environ.get("SPACE_ID") == "autotrain-projects/autotrain-advanced":
        return templates.TemplateResponse("duplicate.html", {"request": request})
    try:
        _users = get_user_and_orgs(user_token=token)
    except Exception as e:
        logger.error(f"Failed to get user and orgs: {e}")
        if "oauth_info" in request.session:
            request.session.pop("oauth_info", None)
        return templates.TemplateResponse("login.html", {"request": request})
    context = {
        "request": request,
        "valid_users": _users,
        "enable_ngc": ENABLE_NGC,
        "enable_nvcf": ENABLE_NVCF,
        "enable_local": AUTOTRAIN_LOCAL,
        "version": __version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return templates.TemplateResponse("index.html", context)


@ui_router.get("/logout", response_class=HTMLResponse)
async def oauth_logout(request: Request, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to logout the oauth user
    :return: HTMLResponse
    """
    request.session.pop("oauth_info", None)
    return RedirectResponse("/")


@ui_router.get("/params/{task}/{param_type}", response_class=JSONResponse)
async def fetch_params(task: str, param_type: str, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the parameters for a given task
    :param task: str
    :param param_type: str (basic, full)
    :return: JSONResponse
    """
    logger.info(f"Task: {task}")
    task_params = get_task_params(task, param_type)
    if len(task_params) == 0:
        return {"error": "Task not found"}
    ui_params = {}
    for param in task_params:
        if param in UI_PARAMS:
            ui_params[param] = UI_PARAMS[param]
            ui_params[param]["default"] = task_params[param]
        else:
            logger.info(f"Param {param} not found in UI_PARAMS")

    ui_params = dict(sorted(ui_params.items(), key=lambda x: (x[1]["type"], x[1]["label"])))
    return ui_params


@ui_router.get("/model_choices/{task}", response_class=JSONResponse)
async def fetch_model_choices(
    task: str,
    custom_models: str = Query(None),
    authenticated: bool = Depends(user_authentication),
):
    """
    This function is used to fetch the model choices for a given task
    :param task: str
    :param custom_models: str (optional, comma separated list of custom models, query parameter)
    :return: JSONResponse
    """
    resp = []

    if custom_models is not None:
        custom_models = custom_models.split(",")
        for custom_model in custom_models:
            custom_model = custom_model.strip()
            resp.append({"id": custom_model, "name": custom_model})

    if os.environ.get("AUTOTRAIN_CUSTOM_MODELS", None) is not None:
        custom_models = os.environ.get("AUTOTRAIN_CUSTOM_MODELS")
        custom_models = custom_models.split(",")
        for custom_model in custom_models:
            custom_model = custom_model.strip()
            resp.append({"id": custom_model, "name": custom_model})

    if task == "text-classification":
        hub_models = MODEL_CHOICE["text-classification"]
    elif task.startswith("llm"):
        hub_models = MODEL_CHOICE["llm"]
    elif task.startswith("st:"):
        hub_models = MODEL_CHOICE["sentence-transformers"]
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
    elif task == "token-classification":
        hub_models = MODEL_CHOICE["token-classification"]
    elif task == "text-regression":
        hub_models = MODEL_CHOICE["text-regression"]
    elif task == "image-object-detection":
        hub_models = MODEL_CHOICE["image-object-detection"]
    else:
        raise NotImplementedError

    for hub_model in hub_models:
        resp.append({"id": hub_model, "name": hub_model})
    return resp


@ui_router.post("/create_project", response_class=JSONResponse)
async def handle_form(
    project_name: str = Form(...),
    task: str = Form(...),
    base_model: str = Form(...),
    hardware: str = Form(...),
    params: str = Form(...),
    autotrain_user: str = Form(...),
    column_mapping: str = Form('{"default": "value"}'),
    data_files_training: List[UploadFile] = File(None),
    data_files_valid: List[UploadFile] = File(None),
    hub_dataset: str = Form(""),
    train_split: str = Form(""),
    valid_split: str = Form(""),
    token: str = Depends(user_authentication),
):
    """
    This function is used to create a new project
    :param project_name: str
    :param task: str
    :param base_model: str
    :param hardware: str
    :param params: str
    :param autotrain_user: str
    :param column_mapping: str
    :param data_files_training: List[UploadFile]
    :param data_files_valid: List[UploadFile]
    :param hub_dataset: str
    :param train_split: str
    :param valid_split: str
    :return: JSONResponse
    """
    train_split = train_split.strip()
    if len(train_split) == 0:
        train_split = None

    valid_split = valid_split.strip()
    if len(valid_split) == 0:
        valid_split = None

    logger.info(f"hardware: {hardware}")
    if hardware == "local-ui":
        running_jobs = get_running_jobs(DB)
        if running_jobs:
            raise HTTPException(
                status_code=409, detail="Another job is already running. Please wait for it to finish."
            )

    if repo_exists(f"{autotrain_user}/{project_name}", token=token):
        raise HTTPException(
            status_code=409,
            detail=f"Project {project_name} already exists. Please choose a different name.",
        )

    params = json.loads(params)
    # convert "null" to None
    for key in params:
        if params[key] == "null":
            params[key] = None
    column_mapping = json.loads(column_mapping)

    training_files = [f.file for f in data_files_training if f.filename != ""] if data_files_training else []
    validation_files = [f.file for f in data_files_valid if f.filename != ""] if data_files_valid else []

    if len(training_files) > 0 and len(hub_dataset) > 0:
        raise HTTPException(
            status_code=400, detail="Please either upload a dataset or choose a dataset from the Hugging Face Hub."
        )

    if len(training_files) == 0 and len(hub_dataset) == 0:
        raise HTTPException(
            status_code=400, detail="Please upload a dataset or choose a dataset from the Hugging Face Hub."
        )

    if len(hub_dataset) > 0 and task == "dreambooth":
        raise HTTPException(status_code=400, detail="Dreambooth does not support Hugging Face Hub datasets.")

    if len(hub_dataset) > 0:
        if not train_split:
            raise HTTPException(status_code=400, detail="Please enter a training split.")

    if len(hub_dataset) == 0:
        file_extension = os.path.splitext(data_files_training[0].filename)[1]
        file_extension = file_extension[1:] if file_extension.startswith(".") else file_extension
        if task == "image-classification":
            dset = AutoTrainImageClassificationDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
            )
        elif task == "image-object-detection":
            dset = AutoTrainObjectDetectionDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
            )
        elif task == "dreambooth":
            dset = AutoTrainDreamboothDataset(
                concept_images=data_files_training,
                concept_name=params["prompt"],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                local=hardware.lower() == "local-ui",
            )

        else:
            if task.startswith("llm"):
                dset_task = "lm_training"
            elif task.startswith("st:"):
                dset_task = "sentence_transformers"
            elif task == "text-classification":
                dset_task = "text_multi_class_classification"
            elif task == "text-regression":
                dset_task = "text_single_column_regression"
            elif task == "seq2seq":
                dset_task = "seq2seq"
            elif task.startswith("tabular"):
                if "," in column_mapping["label"]:
                    column_mapping["label"] = column_mapping["label"].split(",")
                else:
                    column_mapping["label"] = [column_mapping["label"]]
                column_mapping["label"] = [col.strip() for col in column_mapping["label"]]
                subtask = task.split(":")[-1].lower()
                if len(column_mapping["label"]) > 1 and subtask == "classification":
                    dset_task = "tabular_multi_label_classification"
                elif len(column_mapping["label"]) == 1 and subtask == "classification":
                    dset_task = "tabular_multi_class_classification"
                elif len(column_mapping["label"]) > 1 and subtask == "regression":
                    dset_task = "tabular_multi_column_regression"
                elif len(column_mapping["label"]) == 1 and subtask == "regression":
                    dset_task = "tabular_single_column_regression"
                else:
                    raise NotImplementedError
            elif task == "token-classification":
                dset_task = "text_token_classification"
            else:
                raise NotImplementedError
            logger.info(f"Task: {dset_task}")
            logger.info(f"Column mapping: {column_mapping}")
            dset_args = dict(
                train_data=training_files,
                task=dset_task,
                token=token,
                project_name=project_name,
                username=autotrain_user,
                column_mapping=column_mapping,
                valid_data=validation_files,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
                ext=file_extension,
            )
            if task in ("text-classification", "token-classification", "st:pair_class"):
                dset_args["convert_to_class_label"] = True
            dset = AutoTrainDataset(**dset_args)
        data_path = dset.prepare()
    else:
        data_path = hub_dataset
    app_params = AppParams(
        job_params_json=json.dumps(params),
        token=token,
        project_name=project_name,
        username=autotrain_user,
        task=task,
        data_path=data_path,
        base_model=base_model,
        column_mapping=column_mapping,
        using_hub_dataset=len(hub_dataset) > 0,
        train_split=None if len(hub_dataset) == 0 else train_split,
        valid_split=None if len(hub_dataset) == 0 else valid_split,
    )
    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=hardware)
    job_id = project.create()
    monitor_url = ""
    if hardware == "local-ui":
        DB.add_job(job_id)
        monitor_url = "Monitor your job locally / in logs"
    elif hardware.startswith("ep-"):
        monitor_url = f"https://ui.endpoints.huggingface.co/{autotrain_user}/endpoints/{job_id}"
    elif hardware.startswith("spaces-"):
        monitor_url = f"https://hf.co/spaces/{job_id}"
    else:
        monitor_url = f"Success! Monitor your job in logs. Job ID: {job_id}"

    return {"success": "true", "monitor_url": monitor_url}


@ui_router.get("/help/{element_id}", response_class=JSONResponse)
async def fetch_help(element_id: str, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the help text for a given element
    :param element_id: str
    :return: JSONResponse
    """
    msg = get_app_help(element_id)
    return {"message": msg}


@ui_router.get("/accelerators", response_class=JSONResponse)
async def available_accelerators(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the number of available accelerators
    :return: JSONResponse
    """
    if AUTOTRAIN_LOCAL == 0:
        return {"accelerators": "Not available in cloud mode."}
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
    elif mps_available:
        num_gpus = 1
    else:
        num_gpus = 0
    return {"accelerators": num_gpus}


@ui_router.get("/is_model_training", response_class=JSONResponse)
async def is_model_training(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the number of running jobs
    :return: JSONResponse
    """
    if AUTOTRAIN_LOCAL == 0:
        return {"model_training": "Not available in cloud mode."}
    running_jobs = get_running_jobs(DB)
    if running_jobs:
        return {"model_training": True, "pids": running_jobs}
    return {"model_training": False, "pids": []}


@ui_router.get("/logs", response_class=JSONResponse)
async def fetch_logs(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the logs
    :return: JSONResponse
    """
    if not AUTOTRAIN_LOCAL:
        return {"logs": "Logs are only available in local mode."}
    log_file = "autotrain.log"
    with open(log_file, "r", encoding="utf-8") as f:
        logs = f.read()
    if len(str(logs).strip()) == 0:
        logs = "No logs available."

    logs = logs.split("\n")
    logs = logs[::-1]
    # remove lines containing /is_model_training & /accelerators
    logs = [log for log in logs if "/ui/" not in log and "/static/" not in log and "nvidia-ml-py" not in log]

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        devices = Device.all()
        device_logs = []
        for device in devices:
            device_logs.append(
                f"Device {device.index}: {device.name()} - {device.memory_used_human()}/{device.memory_total_human()}"
            )
        device_logs.append("-----------------")
        logs = device_logs + logs
    return {"logs": logs}


@ui_router.get("/stop_training", response_class=JSONResponse)
async def stop_training(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to stop the training
    :return: JSONResponse
    """
    running_jobs = get_running_jobs(DB)
    if running_jobs:
        for _pid in running_jobs:
            try:
                kill_process_by_pid(_pid)
            except Exception:
                logger.info(f"Process {_pid} is already completed. Skipping...")
        return {"success": True}
    return {"success": False}
