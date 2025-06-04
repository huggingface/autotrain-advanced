import json
import os
import signal
import sys
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import subprocess
import sqlite3

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
    AutoTrainImageClassificationDataset,
    AutoTrainImageRegressionDataset,
    AutoTrainObjectDetectionDataset,
    AutoTrainVLMDataset,
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
    "eval_strategy": {
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
    "unsloth": {
        "type": "dropdown",
        "label": "Unsloth",
        "options": [True, False],
    },
    "max_doc_stride": {
        "type": "number",
        "label": "Max doc stride",
    },
    "distributed_backend": {
        "type": "dropdown",
        "label": "Distributed backend",
        "options": ["ddp", "deepspeed"],
    },
    "audio_column": {
        "type": "string",
        "label": "Audio column",
    },
    "text_column": {
        "type": "string",
        "label": "Transcription column",
    },
    "max_duration": {
        "type": "number",
        "label": "Max audio duration (seconds)",
    },
    "sampling_rate": {
        "type": "number",
        "label": "Sampling rate (Hz)",
    },
}


def graceful_exit(signum, frame):
    """
    Handles the SIGTERM signal to perform cleanup and exit the program gracefully.

    Args:
        signum (int): The signal number.
        frame (FrameType): The current stack frame (or None).

    Logs:
        Logs the receipt of the SIGTERM signal and the initiation of cleanup.

    Exits:
        Exits the program with status code 0.
    """
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)


logger.info("AutoTrain started successfully")


async def user_authentication(request: Request):
    """
    Authenticates the user based on the HF_TOKEN environment variable.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        str: The authenticated token string.

    Raises:
        HTTPException: If the token is invalid or expired or verification fails.
    """
    if HF_TOKEN is None:
        logger.error("HF_TOKEN environment variable is not set.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="HF_TOKEN environment variable is not set.",
        )
        
    try:
        get_user_and_orgs(user_token=HF_TOKEN)
        return HF_TOKEN
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token: HF_TOKEN",
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
        logger.error(f"Failed to get user and orgs after authentication: {e}")
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
    elif task == "image-regression":
        hub_models = MODEL_CHOICE["image-regression"]
    elif task.startswith("vlm:"):
        hub_models = MODEL_CHOICE["vlm"]
    elif task == "extractive-qa":
        hub_models = MODEL_CHOICE["extractive-qa"]
    elif task == "automatic-speech-recognition":
        hub_models = MODEL_CHOICE["automatic-speech-recognition"]
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
    life_dataset: str = Form(""),
    train_split: str = Form(""),
    valid_split: str = Form(""),
    token: str = Depends(user_authentication),
):
    """
    Handle form submission for creating and managing AutoTrain projects.

    Args:
        project_name (str): The name of the project.
        task (str): The task type (e.g., "image-classification", "text-classification").
        base_model (str): The base model to use for training.
        hardware (str): The hardware configuration (e.g., "local-ui").
        params (str): JSON string of additional parameters.
        autotrain_user (str): The username of the AutoTrain user.
        column_mapping (str): JSON string mapping columns to their roles.
        data_files_training (List[UploadFile]): List of training data files.
        data_files_valid (List[UploadFile]): List of validation data files.
        hub_dataset (str): The Hugging Face Hub dataset identifier.
        life_dataset (str): The LiFE dataset identifier.
        train_split (str): The training split identifier.
        valid_split (str): The validation split identifier.
        token (str): The authentication token.

    Returns:
        dict: A dictionary containing the success status and monitor URL.

    Raises:
        HTTPException: If there are conflicts or validation errors in the form submission.
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

    if len(life_dataset) > 0:
        training_files = []
        validation_files = []
    else:
        training_files = [f.file for f in data_files_training if f.filename != ""] if data_files_training else []
        validation_files = [f.file for f in data_files_valid if f.filename != ""] if data_files_valid else []

    if len(training_files) > 0 and len(hub_dataset) > 0:
        raise HTTPException(
            status_code=400, detail="Please either upload a dataset or choose a dataset from the Hugging Face Hub."
        )

    if len(training_files) == 0 and len(hub_dataset) == 0 and len(life_dataset) == 0:
        raise HTTPException(
            status_code=400, detail="Please upload a dataset or choose a dataset from the Hugging Face Hub."
        )

    if len(hub_dataset) > 0 or len(life_dataset) > 0:
        if not train_split:
            raise HTTPException(status_code=400, detail="Please enter a training split.")

    if len(life_dataset) > 0:
        file_extension = 'life_jsonl'
    else:
        file_extension = os.path.splitext(data_files_training[0].filename)[1]
        file_extension = file_extension[1:] if file_extension.startswith(".") else file_extension

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
        elif task == "image-regression":
            dset = AutoTrainImageRegressionDataset(
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
        elif task.startswith("vlm:"):
            dset = AutoTrainVLMDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                column_mapping=column_mapping,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
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
            elif task == "extractive-qa":
                dset_task = "text_extractive_question_answering"
            elif task == "automatic-speech-recognition":
                dset_task = "automatic_speech_recognition"
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


@ui_router.post("/create_project")
async def handle_form(request: Request):
    form_data = await request.form()
    
    # Get task type
    task = form_data.get("task")
    
    # Create timestamp for unique config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create config directory
    config_dir = f"{task}_training"
    os.makedirs(config_dir, exist_ok=True)
    
    # Create config file path
    config_path = os.path.join(config_dir, f"training_config_{timestamp}.json")
    
    # Get data path and splits
    data_path = form_data.get("data_path")
    train_split = form_data.get("train_split", "train.json")  # Default to train.json
    valid_split = form_data.get("valid_split", "eval.json")   # Default to eval.json
    
    if not data_path:
        return {"status": "error", "message": "Data path is required"}
    
    # Create config dictionary
    config = {
        "model": form_data.get("model"),
        "model_name": form_data.get("model"),
        "data_path": data_path,
        "train_split": train_split,
        "valid_split": valid_split,
        "audio_column": form_data.get("audio_column", "audio"),
        "text_column": form_data.get("text_column", "transcription"),
        "project_name": form_data.get("project_name"),
        "username": form_data.get("username"),
        "num_train_epochs": int(form_data.get("epochs", 3)),
        "per_device_train_batch_size": int(form_data.get("batch_size", 8)),
        "per_device_eval_batch_size": int(form_data.get("batch_size", 8)),
        "learning_rate": float(form_data.get("learning_rate", 5e-5)),
        "max_steps": -1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "fp16": True,
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 100,
        "save_total_limit": 1,
        "output_dir": f"{task}_training/output",
        "push_to_hub": True,
        "hub_model_id": f"{form_data.get('username')}/{form_data.get('project_name')}",
        "max_duration": 30.0,
        "sampling_rate": 16000,
        "using_hub_dataset": False  # Set to False for local datasets
    }
    
    # Save config file
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Start training process
    if task == "automatic_speech_recognition":
        try:
            # First check if any job is already running
            running_jobs = get_running_jobs(DB)
            if running_jobs:
                return {"status": "error", "message": "Another job is already running. Please wait for it to finish."}
            
            # Start the process
            process = subprocess.Popen([
                "python",
                "src/autotrain/trainers/automatic_speech_recognition/__main__.py",
                "--training_config",
                config_path
            ])
            
            # Get process ID
            pid = process.pid
            
            # Add job to database using existing system
            try:
                DB.add_job(pid)
                logger.info(f"Added job with PID {pid} to database")
            except sqlite3.IntegrityError:
                # If PID already exists, try to kill the old process
                try:
                    kill_process_by_pid(pid)
                except:
                    pass
                # Remove old job and add new one
                DB.remove_job(pid)
                DB.add_job(pid)
            
            return {"status": "success", "message": f"Training started with PID: {pid}"}
            
        except Exception as e:
            logger.error(f"Error starting training: {str(e)}")
            return {"status": "error", "message": f"Error starting training: {str(e)}"}
    
    return {"status": "error", "message": "Invalid task type"}
