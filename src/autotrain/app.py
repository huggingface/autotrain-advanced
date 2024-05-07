import json
import os
from typing import List

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import repo_exists
from nvitop import Device

import autotrain.utils as app_utils
from autotrain import __version__, logger
from autotrain.app_params import AppParams
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.db import AutoTrainDB
from autotrain.help import get_app_help
from autotrain.models import fetch_models
from autotrain.oauth import attach_oauth
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


logger.info("Starting AutoTrain...")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
ENABLE_NGC = int(os.environ.get("ENABLE_NGC", 0))
ENABLE_NVCF = int(os.environ.get("ENABLE_NVCF", 0))
AUTOTRAIN_LOCAL = int(os.environ.get("AUTOTRAIN_LOCAL", 1))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = AutoTrainDB("autotrain.db")

if HF_TOKEN is None and "SPACE_ID" not in os.environ:
    logger.error("HF_TOKEN not set")
    raise ValueError("HF_TOKEN environment variable is not set")

HIDDEN_PARAMS = [
    "token",
    "project_name",
    "username",
    "task",
    "backend",
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
    "tokens_column",
    "tags_column",
]

PARAMS = {}
PARAMS["llm"] = LLMTrainingParams(
    target_modules="all-linear",
    log="tensorboard",
    mixed_precision="fp16",
    quantization="int4",
    peft=True,
    block_size=1024,
    epochs=3,
    padding="right",
    chat_template="none",
    max_completion_length=128,
).model_dump()

PARAMS["text-classification"] = TextClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["image-classification"] = ImageClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["seq2seq"] = Seq2SeqParams(
    mixed_precision="fp16",
    target_modules="all-linear",
    log="tensorboard",
).model_dump()
PARAMS["tabular"] = TabularParams(
    categorical_imputer="most_frequent",
    numerical_imputer="median",
    numeric_scaler="robust",
).model_dump()
PARAMS["dreambooth"] = DreamBoothTrainingParams(
    prompt="<enter your prompt here>",
    vae_model="",
    num_steps=500,
    disable_gradient_checkpointing=False,
    mixed_precision="fp16",
    batch_size=1,
    gradient_accumulation=4,
    resolution=1024,
    use_8bit_adam=False,
    xformers=False,
    train_text_encoder=False,
    lr=1e-4,
).model_dump()
PARAMS["token-classification"] = TokenClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["text-regression"] = TextRegressionParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()


MODEL_CHOICE = fetch_models()

app = FastAPI()
if HF_TOKEN is None:
    attach_oauth(app)

static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)

logger.info("AutoTrain started successfully")


@app.get("/", response_class=HTMLResponse)
async def load_index(request: Request):
    """
    This function is used to load the index page
    :return: HTMLResponse
    """
    if os.environ.get("SPACE_ID") == "autotrain-projects/autotrain-advanced":
        return templates.TemplateResponse("duplicate.html", {"request": request})

    if HF_TOKEN is None:
        try:
            if "oauth_info" not in request.session:
                return templates.TemplateResponse("login.html", {"request": request})
        except AssertionError:
            return templates.TemplateResponse("login.html", {"request": request})

    if HF_TOKEN is None:
        token = request.session["oauth_info"]["access_token"]
    else:
        token = HF_TOKEN
    try:
        _users = app_utils.user_validation(user_token=token)
    except requests.exceptions.JSONDecodeError:
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
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/logout", response_class=HTMLResponse)
async def oauth_logout(request: Request):
    """
    This function is used to logout the oauth user
    :return: HTMLResponse
    """
    request.session.pop("oauth_info", None)
    return RedirectResponse("/")


@app.get("/params/{task}/{param_type}", response_class=JSONResponse)
async def fetch_params(task: str, param_type: str):
    """
    This function is used to fetch the parameters for a given task
    :param task: str
    :param param_type: str (basic, full)
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
                    "max_prompt_length",
                    "max_completion_length",
                ]
            elif trainer == "orpo":
                more_hidden_params = [
                    "model_ref",
                    "dpo_beta",
                    "add_eos_token",
                ]
            elif trainer == "generic":
                more_hidden_params = [
                    "model_ref",
                    "dpo_beta",
                    "max_prompt_length",
                    "max_completion_length",
                ]
            elif trainer == "dpo":
                more_hidden_params = [
                    "add_eos_token",
                ]
            if param_type == "basic":
                more_hidden_params.extend(
                    [
                        "padding",
                        "use_flash_attention_2",
                        "disable_gradient_checkpointing",
                        "logging_steps",
                        "evaluation_strategy",
                        "save_total_limit",
                        "auto_find_batch_size",
                        "warmup_ratio",
                        "weight_decay",
                        "max_grad_norm",
                        "seed",
                        "quantization",
                        "merge_adapter",
                        "lora_r",
                        "lora_alpha",
                        "lora_dropout",
                        "max_completion_length",
                    ]
                )
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "text-classification" and param_type == "basic":
            more_hidden_params = [
                "warmup_ratio",
                "weight_decay",
                "max_grad_norm",
                "seed",
                "logging_steps",
                "auto_find_batch_size",
                "save_total_limit",
                "evaluation_strategy",
            ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "text-regression" and param_type == "basic":
            more_hidden_params = [
                "warmup_ratio",
                "weight_decay",
                "max_grad_norm",
                "seed",
                "logging_steps",
                "auto_find_batch_size",
                "save_total_limit",
                "evaluation_strategy",
            ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "image-classification" and param_type == "basic":
            more_hidden_params = [
                "warmup_ratio",
                "weight_decay",
                "max_grad_norm",
                "seed",
                "logging_steps",
                "auto_find_batch_size",
                "save_total_limit",
                "evaluation_strategy",
            ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "seq2seq" and param_type == "basic":
            more_hidden_params = [
                "warmup_ratio",
                "weight_decay",
                "max_grad_norm",
                "seed",
                "logging_steps",
                "auto_find_batch_size",
                "save_total_limit",
                "evaluation_strategy",
                "quantization",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
            ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "token-classification" and param_type == "basic":
            more_hidden_params = [
                "warmup_ratio",
                "weight_decay",
                "max_grad_norm",
                "seed",
                "logging_steps",
                "auto_find_batch_size",
                "save_total_limit",
                "evaluation_strategy",
            ]
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        if task == "dreambooth":
            more_hidden_params = [
                "epochs",
                "logging",
                "bf16",
            ]
            if param_type == "basic":
                more_hidden_params.extend(
                    [
                        "prior_preservation",
                        "prior_loss_weight",
                        "seed",
                        "center_crop",
                        "train_text_encoder",
                        "disable_gradient_checkpointing",
                        "scale_lr",
                        "warmup_steps",
                        "num_cycles",
                        "lr_power",
                        "adam_beta1",
                        "adam_beta2",
                        "adam_weight_decay",
                        "adam_epsilon",
                        "max_grad_norm",
                        "pre_compute_text_embeddings",
                        "text_encoder_use_attention_mask",
                    ]
                )
            task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
        return task_params
    return {"error": "Task not found"}


@app.get("/model_choices/{task}", response_class=JSONResponse)
async def fetch_model_choices(task: str, custom_models: str = Query(None)):
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
    else:
        raise NotImplementedError

    for hub_model in hub_models:
        resp.append({"id": hub_model, "name": hub_model})
    return resp


@app.post("/create_project", response_class=JSONResponse)
async def handle_form(
    request: Request,
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
):
    """
    This function is used to create a new project
    :param request: Request
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
        running_jobs = app_utils.get_running_jobs(DB)
        if running_jobs:
            raise HTTPException(
                status_code=409, detail="Another job is already running. Please wait for it to finish."
            )

    if HF_TOKEN is None:
        token = request.session["oauth_info"]["access_token"]
    else:
        token = HF_TOKEN

    if repo_exists(f"{autotrain_user}/{project_name}", token=token):
        raise HTTPException(
            status_code=409,
            detail=f"Project {project_name} already exists. Please choose a different name.",
        )

    params = json.loads(params)
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

    file_extension = os.path.splitext(data_files_training[0].filename)[1]
    file_extension = file_extension[1:] if file_extension.startswith(".") else file_extension

    if len(hub_dataset) == 0:
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
            elif task == "text-classification":
                dset_task = "text_multi_class_classification"
            elif task == "text-regression":
                dset_task = "text_single_column_regression"
            elif task == "seq2seq":
                dset_task = "seq2seq"
            elif task.startswith("tabular"):
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
            if task in ("text-classification", "token-classification"):
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
        monitor_url = "Success! Monitor your job in logs. Job ID: {job_id}"

    return {"success": "true", "monitor_url": monitor_url}


@app.get("/help/{element_id}", response_class=JSONResponse)
async def fetch_help(element_id: str):
    """
    This function is used to fetch the help text for a given element
    :param element_id: str
    :return: JSONResponse
    """
    msg = get_app_help(element_id)
    return {"message": msg}


@app.get("/accelerators", response_class=JSONResponse)
async def available_accelerators():
    """
    This function is used to fetch the number of available accelerators
    :return: JSONResponse
    """
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
    elif mps_available:
        num_gpus = 1
    else:
        num_gpus = 0
    return {"accelerators": num_gpus}


@app.get("/is_model_training", response_class=JSONResponse)
async def is_model_training():
    """
    This function is used to fetch the number of running jobs
    :return: JSONResponse
    """
    running_jobs = app_utils.get_running_jobs(DB)
    if running_jobs:
        return {"model_training": True, "pids": running_jobs}
    return {"model_training": False, "pids": []}


@app.get("/logs", response_class=JSONResponse)
async def fetch_logs():
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

    devices = Device.all()
    device_logs = []
    for device in devices:
        device_logs.append(
            f"Device {device.index}: {device.name()} - {device.memory_used_human()}/{device.memory_total_human()}"
        )
    device_logs.append("-----------------")
    logs = device_logs + logs
    return {"logs": logs}


@app.get("/stop_training", response_class=JSONResponse)
async def stop_training():
    """
    This function is used to stop the training
    :return: JSONResponse
    """
    running_jobs = app_utils.get_running_jobs(DB)
    if running_jobs:
        for _pid in running_jobs:
            try:
                app_utils.kill_process_by_pid(_pid)
            except Exception:
                logger.info(f"Process {_pid} is already completed. Skipping...")
        return {"success": True}
    return {"success": False}
