import collections
import json
import os
from typing import List

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import ModelFilter, list_models

from autotrain import __version__, app_utils, logger
from autotrain.app_params import AppParams
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.db import AutoTrainDB
from autotrain.help import get_app_help
from autotrain.oauth import attach_oauth
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


HF_TOKEN = os.environ.get("HF_TOKEN", None)
ENABLE_NGC = int(os.environ.get("ENABLE_NGC", 0))
ENABLE_NVCF = int(os.environ.get("ENABLE_NVCF", 0))
DB = AutoTrainDB("autotrain.db")
AUTOTRAIN_LOCAL = int(os.environ.get("AUTOTRAIN_LOCAL", 1))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if HF_TOKEN is None and "SPACE_ID" not in os.environ:
    logger.error("HF_TOKEN not set")
    raise ValueError("HF_TOKEN environment variable is not set")

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
    target_modules="all-linear",
    log="tensorboard",
    mixed_precision="fp16",
    quantization="int4",
    peft=True,
    block_size=1024,
    epochs=3,
    padding="right",
    chat_template="none",
).model_dump()

PARAMS["text-classification"] = TextClassificationParams(
    mixed_precision="fp16",
).model_dump()
PARAMS["image-classification"] = ImageClassificationParams(
    mixed_precision="fp16",
    target_modules="all-linear",
).model_dump()
PARAMS["seq2seq"] = Seq2SeqParams(
    mixed_precision="fp16",
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

    hub_models1 = list_models(filter="fill-mask", sort="downloads", direction=-1, limit=100, full=False)
    hub_models2 = list(
        list_models(filter="token-classification", sort="downloads", direction=-1, limit=100, full=False)
    )
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)
    _mc["token-classification"] = hub_models

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
if HF_TOKEN is None:
    attach_oauth(app)

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
    if os.environ.get("SPACE_ID") == "autotrain-projects/autotrain-advanced":
        return templates.TemplateResponse("duplicate.html", {"request": request})

    # if HF_TOKEN is None and USE_OAUTH == 0:
    #     return templates.TemplateResponse("error.html", {"request": request})

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

    _users = app_utils.user_validation(user_token=token)
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
    request.session.pop("oauth_info", None)
    return RedirectResponse("/")


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
async def fetch_model_choices(task: str, custom_models: str = Query(None)):
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
    data_files_training: List[UploadFile] = File(...),
    data_files_valid: List[UploadFile] = File(...),
):
    """
    This function is used to handle the form submission
    """
    logger.info(f"hardware: {hardware}")
    if hardware == "Local":
        running_jobs = app_utils.get_running_jobs(DB)
        if running_jobs:
            logger.info(f"Running jobs: {running_jobs}")
            raise HTTPException(
                status_code=409, detail="Another job is already running. Please wait for it to finish."
            )

    if HF_TOKEN is None:
        token = request.session["oauth_info"]["access_token"]
    else:
        token = HF_TOKEN

    params = json.loads(params)
    column_mapping = json.loads(column_mapping)

    training_files = [f.file for f in data_files_training if f.filename != ""]
    validation_files = [f.file for f in data_files_valid if f.filename != ""] if data_files_valid else []

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
            local=hardware.lower() == "local",
        )
    elif task == "dreambooth":
        dset = AutoTrainDreamboothDataset(
            concept_images=data_files_training,
            concept_name=params["prompt"],
            token=token,
            project_name=project_name,
            username=autotrain_user,
            local=hardware.lower() == "local",
        )

    else:
        if task.startswith("llm"):
            dset_task = "lm_training"
        elif task == "text-classification":
            dset_task = "text_multi_class_classification"
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
            local=hardware.lower() == "local",
            ext=file_extension,
        )
        if task in ("text-classification", "token-classification"):
            dset_args["convert_to_class_label"] = True
        dset = AutoTrainDataset(**dset_args)
    data_path = dset.prepare()
    app_params = AppParams(
        job_params_json=json.dumps(params),
        token=token,
        project_name=project_name,
        username=autotrain_user,
        task=task,
        data_path=data_path,
        base_model=base_model,
        column_mapping=column_mapping,
    )
    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=hardware)
    job_id = project.create()
    monitor_url = ""
    if hardware == "Local":
        DB.add_job(job_id)
        monitor_url = "Monitor your job locally / in logs"
    elif hardware.startswith("EP"):
        monitor_url = f"https://ui.endpoints.huggingface.co/{autotrain_user}/endpoints/{job_id}"
    else:
        monitor_url = f"https://hf.co/spaces/{job_id}"
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
