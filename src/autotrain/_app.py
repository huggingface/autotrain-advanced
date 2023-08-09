import json
import os
import random
import string
import zipfile

import gradio as gr
import pandas as pd
from huggingface_hub import list_models
from loguru import logger

from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.languages import SUPPORTED_LANGUAGES
from autotrain.params import Params
from autotrain.project import Project
from autotrain.utils import get_project_cost, get_user_token, user_authentication


APP_TASKS = {
    "Natural Language Processing": ["Text Classification", "LLM Finetuning"],
    # "Tabular": TABULAR_TASKS,
    "Computer Vision": ["Image Classification", "Dreambooth"],
}

APP_TASKS_MAPPING = {
    "Text Classification": "text_multi_class_classification",
    "LLM Finetuning": "lm_training",
    "Image Classification": "image_multi_class_classification",
    "Dreambooth": "dreambooth",
}

APP_TASK_TYPE_MAPPING = {
    "text_classification": "Natural Language Processing",
    "lm_training": "Natural Language Processing",
    "image_classification": "Computer Vision",
    "dreambooth": "Computer Vision",
}

ALLOWED_FILE_TYPES = [
    ".csv",
    ".CSV",
    ".jsonl",
    ".JSONL",
    ".zip",
    ".ZIP",
    ".png",
    ".PNG",
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
]


def _login_user(user_token):
    user_info = user_authentication(token=user_token)
    username = user_info["name"]

    user_can_pay = user_info["canPay"]
    orgs = user_info["orgs"]

    valid_orgs = [org for org in orgs if org["canPay"] is True]
    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]
    valid_orgs = [org["name"] for org in valid_orgs]

    valid_can_pay = [username] + valid_orgs if user_can_pay else valid_orgs
    who_is_training = [username] + [org["name"] for org in orgs]
    return user_token, valid_can_pay, who_is_training


def _update_task_type(project_type):
    return gr.Dropdown.update(
        value=APP_TASKS[project_type][0],
        choices=APP_TASKS[project_type],
        visible=True,
    )


def _update_model_choice(task, autotrain_backend):
    # TODO: add tabular and remember, for tabular, we only support AutoTrain
    if autotrain_backend.lower() != "huggingface internal":
        model_choice = ["HuggingFace Hub"]
        return gr.Dropdown.update(
            value=model_choice[0],
            choices=model_choice,
            visible=True,
        )

    if task == "LLM Finetuning":
        model_choice = ["HuggingFace Hub"]
    else:
        model_choice = ["AutoTrain", "HuggingFace Hub"]

    return gr.Dropdown.update(
        value=model_choice[0],
        choices=model_choice,
        visible=True,
    )


def _update_file_type(task):
    task = APP_TASKS_MAPPING[task]
    if task in ("text_multi_class_classification", "lm_training"):
        return gr.Radio.update(
            value="CSV",
            choices=["CSV", "JSONL"],
            visible=True,
        )
    elif task == "image_multi_class_classification":
        return gr.Radio.update(
            value="ZIP",
            choices=["Image Subfolders", "ZIP"],
            visible=True,
        )
    elif task == "dreambooth":
        return gr.Radio.update(
            value="ZIP",
            choices=["Image Folder", "ZIP"],
            visible=True,
        )
    else:
        raise NotImplementedError


def _update_param_choice(model_choice, autotrain_backend):
    logger.info(f"model_choice: {model_choice}")
    choices = ["AutoTrain", "Manual"] if model_choice == "HuggingFace Hub" else ["AutoTrain"]
    choices = ["Manual"] if autotrain_backend != "HuggingFace Internal" else choices
    return gr.Dropdown.update(
        value=choices[0],
        choices=choices,
        visible=True,
    )


def _project_type_update(project_type, task_type, autotrain_backend):
    logger.info(f"project_type: {project_type}, task_type: {task_type}")
    task_choices_update = _update_task_type(project_type)
    model_choices_update = _update_model_choice(task_choices_update["value"], autotrain_backend)
    param_choices_update = _update_param_choice(model_choices_update["value"], autotrain_backend)
    return [
        task_choices_update,
        model_choices_update,
        param_choices_update,
        _update_hub_model_choices(task_choices_update["value"], model_choices_update["value"]),
    ]


def _task_type_update(task_type, autotrain_backend):
    logger.info(f"task_type: {task_type}")
    model_choices_update = _update_model_choice(task_type, autotrain_backend)
    param_choices_update = _update_param_choice(model_choices_update["value"], autotrain_backend)
    return [
        model_choices_update,
        param_choices_update,
        _update_hub_model_choices(task_type, model_choices_update["value"]),
    ]


def _update_col_map(training_data, task):
    task = APP_TASKS_MAPPING[task]
    if task == "text_multi_class_classification":
        data_cols = pd.read_csv(training_data[0].name, nrows=2).columns.tolist()
        return [
            gr.Dropdown.update(visible=True, choices=data_cols, label="Map `text` column", value=data_cols[0]),
            gr.Dropdown.update(visible=True, choices=data_cols, label="Map `target` column", value=data_cols[1]),
            gr.Text.update(visible=False),
        ]
    elif task == "lm_training":
        data_cols = pd.read_csv(training_data[0].name, nrows=2).columns.tolist()
        return [
            gr.Dropdown.update(visible=True, choices=data_cols, label="Map `text` column", value=data_cols[0]),
            gr.Dropdown.update(visible=False),
            gr.Text.update(visible=False),
        ]
    elif task == "dreambooth":
        return [
            gr.Dropdown.update(visible=False),
            gr.Dropdown.update(visible=False),
            gr.Text.update(visible=True, label="Concept Token", interactive=True),
        ]
    else:
        return [
            gr.Dropdown.update(visible=False),
            gr.Dropdown.update(visible=False),
            gr.Text.update(visible=False),
        ]


def _estimate_costs(
    training_data, validation_data, task, user_token, autotrain_username, training_params_txt, autotrain_backend
):
    if autotrain_backend.lower() != "huggingface internal":
        return [
            gr.Markdown.update(
                value="Cost estimation is not available for this backend",
                visible=True,
            ),
            gr.Number.update(visible=False),
        ]
    try:
        logger.info("Estimating costs....")
        if training_data is None:
            return [
                gr.Markdown.update(
                    value="Could not estimate cost. Please add training data",
                    visible=True,
                ),
                gr.Number.update(visible=False),
            ]
        if validation_data is None:
            validation_data = []

        training_params = json.loads(training_params_txt)
        if len(training_params) == 0:
            return [
                gr.Markdown.update(
                    value="Could not estimate cost. Please add atleast one job",
                    visible=True,
                ),
                gr.Number.update(visible=False),
            ]
        elif len(training_params) == 1:
            if "num_models" in training_params[0]:
                num_models = training_params[0]["num_models"]
            else:
                num_models = 1
        else:
            num_models = len(training_params)
        task = APP_TASKS_MAPPING[task]
        num_samples = 0
        logger.info("Estimating number of samples")
        if task in ("text_multi_class_classification", "lm_training"):
            for _f in training_data:
                num_samples += pd.read_csv(_f.name).shape[0]
            for _f in validation_data:
                num_samples += pd.read_csv(_f.name).shape[0]
        elif task == "image_multi_class_classification":
            logger.info(f"training_data: {training_data}")
            if len(training_data) > 1:
                return [
                    gr.Markdown.update(
                        value="Only one training file is supported for image classification",
                        visible=True,
                    ),
                    gr.Number.update(visible=False),
                ]
            if len(validation_data) > 1:
                return [
                    gr.Markdown.update(
                        value="Only one validation file is supported for image classification",
                        visible=True,
                    ),
                    gr.Number.update(visible=False),
                ]
            for _f in training_data:
                zip_ref = zipfile.ZipFile(_f.name, "r")
                for _ in zip_ref.namelist():
                    num_samples += 1
            for _f in validation_data:
                zip_ref = zipfile.ZipFile(_f.name, "r")
                for _ in zip_ref.namelist():
                    num_samples += 1
        elif task == "dreambooth":
            num_samples = len(training_data)
        else:
            raise NotImplementedError

        logger.info(f"Estimating costs for: num_models: {num_models}, task: {task}, num_samples: {num_samples}")
        estimated_cost = get_project_cost(
            username=autotrain_username,
            token=user_token,
            task=task,
            num_samples=num_samples,
            num_models=num_models,
        )
        logger.info(f"Estimated_cost: {estimated_cost}")
        return [
            gr.Markdown.update(
                value=f"Estimated cost: ${estimated_cost:.2f}. Note: clicking on 'Create Project' will start training and incur charges!",
                visible=True,
            ),
            gr.Number.update(visible=False),
        ]
    except Exception as e:
        logger.error(e)
        logger.error("Could not estimate cost, check inputs")
        return [
            gr.Markdown.update(
                value="Could not estimate cost, check inputs",
                visible=True,
            ),
            gr.Number.update(visible=False),
        ]


def get_job_params(param_choice, training_params, task):
    if param_choice == "autotrain":
        if len(training_params) > 1:
            raise ValueError("❌ Only one job parameter is allowed for AutoTrain.")
        training_params[0].update({"task": task})
    elif param_choice.lower() == "manual":
        for i in range(len(training_params)):
            training_params[i].update({"task": task})
            if "hub_model" in training_params[i]:
                # remove hub_model from training_params
                training_params[i].pop("hub_model")
    return training_params


def _update_project_name():
    random_project_name = "-".join(
        ["".join(random.choices(string.ascii_lowercase + string.digits, k=4)) for _ in range(3)]
    )
    # check if training tracker exists
    if os.path.exists(os.path.join("/tmp", "training")):
        return [
            gr.Text.update(value=random_project_name, visible=True, interactive=True),
            gr.Button.update(interactive=False),
        ]
    return [
        gr.Text.update(value=random_project_name, visible=True, interactive=True),
        gr.Button.update(interactive=True),
    ]


def _update_hub_model_choices(task, model_choice):
    task = APP_TASKS_MAPPING[task]
    logger.info(f"Updating hub model choices for task: {task}, model_choice: {model_choice}")
    if model_choice.lower() == "autotrain":
        return gr.Dropdown.update(
            visible=False,
            interactive=False,
        )
    if task == "text_multi_class_classification":
        hub_models1 = list_models(filter="fill-mask", sort="downloads", direction=-1, limit=100)
        hub_models2 = list_models(filter="text-classification", sort="downloads", direction=-1, limit=100)
        hub_models = list(hub_models1) + list(hub_models2)
    elif task == "lm_training":
        hub_models = list(list_models(filter="text-generation", sort="downloads", direction=-1, limit=100))
    elif task == "image_multi_class_classification":
        hub_models = list(list_models(filter="image-classification", sort="downloads", direction=-1, limit=100))
    elif task == "dreambooth":
        hub_models = list(list_models(filter="text-to-image", sort="downloads", direction=-1, limit=100))
    else:
        raise NotImplementedError
    # sort by number of downloads in descending order
    hub_models = [{"id": m.modelId, "downloads": m.downloads} for m in hub_models if m.private is False]
    hub_models = sorted(hub_models, key=lambda x: x["downloads"], reverse=True)

    if task == "dreambooth":
        choices = ["stabilityai/stable-diffusion-xl-base-1.0"] + [m["id"] for m in hub_models]
        value = choices[0]
        return gr.Dropdown.update(
            choices=choices,
            value=value,
            visible=True,
            interactive=True,
        )

    return gr.Dropdown.update(
        choices=[m["id"] for m in hub_models],
        value=hub_models[0]["id"],
        visible=True,
        interactive=True,
    )


def _update_backend(backend):
    if backend != "Hugging Face Internal":
        return [
            gr.Dropdown.update(
                visible=True,
                interactive=True,
                choices=["HuggingFace Hub"],
                value="HuggingFace Hub",
            ),
            gr.Dropdown.update(
                visible=True,
                interactive=True,
                choices=["Manual"],
                value="Manual",
            ),
        ]
    return [
        gr.Dropdown.update(
            visible=True,
            interactive=True,
        ),
        gr.Dropdown.update(
            visible=True,
            interactive=True,
        ),
    ]


def _create_project(
    autotrain_username,
    valid_can_pay,
    project_name,
    user_token,
    task,
    training_data,
    validation_data,
    col_map_text,
    col_map_label,
    concept_token,
    training_params_txt,
    hub_model,
    estimated_cost,
    autotrain_backend,
):
    task = APP_TASKS_MAPPING[task]
    valid_can_pay = valid_can_pay.split(",")
    can_pay = autotrain_username in valid_can_pay
    logger.info(f"🚨🚨🚨Creating project: {project_name}")
    logger.info(f"🚨Task: {task}")
    logger.info(f"🚨Training data: {training_data}")
    logger.info(f"🚨Validation data: {validation_data}")
    logger.info(f"🚨Training params: {training_params_txt}")
    logger.info(f"🚨Hub model: {hub_model}")
    logger.info(f"🚨Estimated cost: {estimated_cost}")
    logger.info(f"🚨:Can pay: {can_pay}")

    if can_pay is False and estimated_cost > 0:
        raise gr.Error("❌ You do not have enough credits to create this project. Please add a valid payment method.")

    training_params = json.loads(training_params_txt)
    if len(training_params) == 0:
        raise gr.Error("Please add atleast one job")
    elif len(training_params) == 1:
        if "num_models" in training_params[0]:
            param_choice = "autotrain"
        else:
            param_choice = "manual"
    else:
        param_choice = "manual"

    if task == "image_multi_class_classification":
        training_data = training_data[0].name
        if validation_data is not None:
            validation_data = validation_data[0].name
        dset = AutoTrainImageClassificationDataset(
            train_data=training_data,
            token=user_token,
            project_name=project_name,
            username=autotrain_username,
            valid_data=validation_data,
            percent_valid=None,  # TODO: add to UI
        )
    elif task == "text_multi_class_classification":
        training_data = [f.name for f in training_data]
        if validation_data is None:
            validation_data = []
        else:
            validation_data = [f.name for f in validation_data]
        dset = AutoTrainDataset(
            train_data=training_data,
            task=task,
            token=user_token,
            project_name=project_name,
            username=autotrain_username,
            column_mapping={"text": col_map_text, "label": col_map_label},
            valid_data=validation_data,
            percent_valid=None,  # TODO: add to UI
        )
    elif task == "lm_training":
        training_data = [f.name for f in training_data]
        if validation_data is None:
            validation_data = []
        else:
            validation_data = [f.name for f in validation_data]
        dset = AutoTrainDataset(
            train_data=training_data,
            task=task,
            token=user_token,
            project_name=project_name,
            username=autotrain_username,
            column_mapping={"text": col_map_text},
            valid_data=validation_data,
            percent_valid=None,  # TODO: add to UI
        )
    elif task == "dreambooth":
        dset = AutoTrainDreamboothDataset(
            concept_images=training_data,
            concept_name=concept_token,
            token=user_token,
            project_name=project_name,
            username=autotrain_username,
        )
    else:
        raise NotImplementedError

    dset.prepare()
    project = Project(
        dataset=dset,
        param_choice=param_choice,
        hub_model=hub_model,
        job_params=get_job_params(param_choice, training_params, task),
    )
    if autotrain_backend.lower() == "huggingface internal":
        project_id = project.create()
        project.approve(project_id)
        return gr.Markdown.update(
            value=f"Project created successfully. Monitor progess on the [dashboard](https://ui.autotrain.huggingface.co/{project_id}/trainings).",
            visible=True,
        )
    else:
        project.create(local=True)


def get_variable_name(var, namespace):
    for name in namespace:
        if namespace[name] is var:
            return name
    return None


def disable_create_project_button():
    return gr.Button.update(interactive=False)


def main():
    with gr.Blocks(theme="freddyaboulton/dracula_revamped") as demo:
        gr.Markdown("## 🤗 AutoTrain Advanced")
        user_token = os.environ.get("HF_TOKEN", "")

        if len(user_token) == 0:
            user_token = get_user_token()

        if user_token is None:
            gr.Markdown(
                """Please login with a write [token](https://huggingface.co/settings/tokens).
                Pass your HF token in an environment variable called `HF_TOKEN` and then restart this app.
                """
            )
            return demo

        user_token, valid_can_pay, who_is_training = _login_user(user_token)

        if user_token is None or len(user_token) == 0:
            gr.Error("Please login with a write token.")

        user_token = gr.Textbox(
            value=user_token, type="password", lines=1, max_lines=1, visible=False, interactive=False
        )
        valid_can_pay = gr.Textbox(value=",".join(valid_can_pay), visible=False, interactive=False)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    autotrain_username = gr.Dropdown(
                        label="AutoTrain Username",
                        choices=who_is_training,
                        value=who_is_training[0] if who_is_training else "",
                    )
                    autotrain_backend = gr.Dropdown(
                        label="AutoTrain Backend",
                        choices=["HuggingFace Internal", "HuggingFace Spaces"],
                        value="HuggingFace Internal",
                        interactive=True,
                    )
                with gr.Row():
                    project_name = gr.Textbox(label="Project name", value="", lines=1, max_lines=1, interactive=True)
                    project_type = gr.Dropdown(
                        label="Project Type", choices=list(APP_TASKS.keys()), value=list(APP_TASKS.keys())[0]
                    )
                    task_type = gr.Dropdown(
                        label="Task",
                        choices=APP_TASKS[list(APP_TASKS.keys())[0]],
                        value=APP_TASKS[list(APP_TASKS.keys())[0]][0],
                        interactive=True,
                    )
                    model_choice = gr.Dropdown(
                        label="Model Choice",
                        choices=["AutoTrain", "HuggingFace Hub"],
                        value="AutoTrain",
                        visible=True,
                        interactive=True,
                    )
                hub_model = gr.Dropdown(
                    label="Hub Model",
                    value="",
                    visible=False,
                    interactive=True,
                    elem_id="hub_model",
                )
        gr.Markdown("<hr>")
        with gr.Row():
            with gr.Column():
                with gr.Tabs(elem_id="tabs"):
                    with gr.TabItem("Data"):
                        with gr.Column():
                            # file_type_training = gr.Radio(
                            #     label="File Type",
                            #     choices=["CSV", "JSONL"],
                            #     value="CSV",
                            #     visible=True,
                            #     interactive=True,
                            # )
                            training_data = gr.File(
                                label="Training Data",
                                file_types=ALLOWED_FILE_TYPES,
                                file_count="multiple",
                                visible=True,
                                interactive=True,
                                elem_id="training_data_box",
                            )
                            with gr.Accordion("Validation Data (Optional)", open=False):
                                validation_data = gr.File(
                                    label="Validation Data (Optional)",
                                    file_types=ALLOWED_FILE_TYPES,
                                    file_count="multiple",
                                    visible=True,
                                    interactive=True,
                                    elem_id="validation_data_box",
                                )
                            with gr.Row():
                                col_map_text = gr.Dropdown(
                                    label="Text Column", choices=[], visible=False, interactive=True
                                )
                                col_map_target = gr.Dropdown(
                                    label="Target Column", choices=[], visible=False, interactive=True
                                )
                                concept_token = gr.Text(
                                    value="", visible=False, interactive=True, lines=1, max_lines=1
                                )
                    with gr.TabItem("Params"):
                        with gr.Row():
                            source_language = gr.Dropdown(
                                label="Source Language",
                                choices=SUPPORTED_LANGUAGES[:-1],
                                value="en",
                                visible=True,
                                interactive=True,
                                elem_id="source_language",
                            )
                            num_models = gr.Slider(
                                label="Number of Models",
                                minimum=1,
                                maximum=25,
                                value=5,
                                step=1,
                                visible=True,
                                interactive=True,
                                elem_id="num_models",
                            )
                            target_language = gr.Dropdown(
                                label="Target Language",
                                choices=["fr"],
                                value="fr",
                                visible=False,
                                interactive=True,
                                elem_id="target_language",
                            )
                            image_size = gr.Number(
                                label="Image Size",
                                value=512,
                                visible=False,
                                interactive=True,
                                elem_id="image_size",
                            )

                        with gr.Row():
                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=5e-5,
                                visible=False,
                                interactive=True,
                                elem_id="learning_rate",
                            )
                            batch_size = gr.Number(
                                label="Train Batch Size",
                                value=32,
                                visible=False,
                                interactive=True,
                                elem_id="train_batch_size",
                            )
                            num_epochs = gr.Number(
                                label="Number of Epochs",
                                value=3,
                                visible=False,
                                interactive=True,
                                elem_id="num_train_epochs",
                            )
                        with gr.Row():
                            gradient_accumulation_steps = gr.Number(
                                label="Gradient Accumulation Steps",
                                value=1,
                                visible=False,
                                interactive=True,
                                elem_id="gradient_accumulation_steps",
                            )
                            percentage_warmup_steps = gr.Number(
                                label="Percentage of Warmup Steps",
                                value=0.1,
                                visible=False,
                                interactive=True,
                                elem_id="percentage_warmup",
                            )
                            weight_decay = gr.Number(
                                label="Weight Decay",
                                value=0.01,
                                visible=False,
                                interactive=True,
                                elem_id="weight_decay",
                            )
                        with gr.Row():
                            lora_r = gr.Number(
                                label="LoraR",
                                value=16,
                                visible=False,
                                interactive=True,
                                elem_id="lora_r",
                            )
                            lora_alpha = gr.Number(
                                label="LoraAlpha",
                                value=32,
                                visible=False,
                                interactive=True,
                                elem_id="lora_alpha",
                            )
                            lora_dropout = gr.Number(
                                label="Lora Dropout",
                                value=0.1,
                                visible=False,
                                interactive=True,
                                elem_id="lora_dropout",
                            )
                        with gr.Row():
                            db_num_steps = gr.Number(
                                label="Num Steps",
                                value=500,
                                visible=False,
                                interactive=True,
                                elem_id="num_steps",
                            )
                        with gr.Row():
                            optimizer = gr.Dropdown(
                                label="Optimizer",
                                choices=["adamw_torch", "adamw_hf", "sgd", "adafactor", "adagrad"],
                                value="adamw_torch",
                                visible=False,
                                interactive=True,
                                elem_id="optimizer",
                            )
                            scheduler = gr.Dropdown(
                                label="Scheduler",
                                choices=["linear", "cosine"],
                                value="linear",
                                visible=False,
                                interactive=True,
                                elem_id="scheduler",
                            )

                        add_job_button = gr.Button(
                            value="Add Job",
                            visible=True,
                            interactive=True,
                            elem_id="add_job",
                        )
                        # clear_jobs_button = gr.Button(
                        #     value="Clear Jobs",
                        #     visible=True,
                        #     interactive=True,
                        #     elem_id="clear_jobs",
                        # )
                gr.Markdown("<hr>")
                estimated_costs_md = gr.Markdown(value="Estimated Costs: N/A", visible=True, interactive=False)
                estimated_costs_num = gr.Number(value=0, visible=False, interactive=False)
                create_project_button = gr.Button(
                    value="Create Project",
                    visible=True,
                    interactive=True,
                    elem_id="create_project",
                )
            with gr.Column():
                param_choice = gr.Dropdown(
                    label="Param Choice",
                    choices=["AutoTrain"],
                    value="AutoTrain",
                    visible=True,
                    interactive=True,
                )
                training_params_txt = gr.Text(value="[]", visible=False, interactive=False)
                training_params_md = gr.DataFrame(visible=False, interactive=False)

        final_output = gr.Markdown(value="", visible=True, interactive=False)
        hyperparameters = [
            hub_model,
            num_models,
            source_language,
            target_language,
            learning_rate,
            batch_size,
            num_epochs,
            gradient_accumulation_steps,
            lora_r,
            lora_alpha,
            lora_dropout,
            optimizer,
            scheduler,
            percentage_warmup_steps,
            weight_decay,
            db_num_steps,
            image_size,
        ]

        def _update_params(params_data):
            _task = params_data[task_type]
            _task = APP_TASKS_MAPPING[_task]
            params = Params(
                task=_task,
                param_choice="autotrain" if params_data[param_choice] == "AutoTrain" else "manual",
                model_choice="autotrain" if params_data[model_choice] == "AutoTrain" else "hub_model",
            )
            params = params.get()
            visible_params = []
            for param in hyperparameters:
                if param.elem_id in params.keys():
                    visible_params.append(param.elem_id)
            op = [h.update(visible=h.elem_id in visible_params) for h in hyperparameters]
            op.append(add_job_button.update(visible=True))
            op.append(training_params_md.update(visible=False))
            op.append(training_params_txt.update(value="[]"))
            return op

        autotrain_backend.change(
            _project_type_update,
            inputs=[project_type, task_type, autotrain_backend],
            outputs=[task_type, model_choice, param_choice, hub_model],
        )

        project_type.change(
            _project_type_update,
            inputs=[project_type, task_type, autotrain_backend],
            outputs=[task_type, model_choice, param_choice, hub_model],
        )
        task_type.change(
            _task_type_update,
            inputs=[task_type, autotrain_backend],
            outputs=[model_choice, param_choice, hub_model],
        )
        model_choice.change(
            _update_param_choice,
            inputs=[model_choice, autotrain_backend],
            outputs=param_choice,
        ).then(
            _update_hub_model_choices,
            inputs=[task_type, model_choice],
            outputs=hub_model,
        )

        param_choice.change(
            _update_params,
            inputs=set([task_type, param_choice, model_choice] + hyperparameters + [add_job_button]),
            outputs=hyperparameters + [add_job_button, training_params_md, training_params_txt],
        )
        task_type.change(
            _update_params,
            inputs=set([task_type, param_choice, model_choice] + hyperparameters + [add_job_button]),
            outputs=hyperparameters + [add_job_button, training_params_md, training_params_txt],
        )
        model_choice.change(
            _update_params,
            inputs=set([task_type, param_choice, model_choice] + hyperparameters + [add_job_button]),
            outputs=hyperparameters + [add_job_button, training_params_md, training_params_txt],
        )

        def _add_job(params_data):
            _task = params_data[task_type]
            _task = APP_TASKS_MAPPING[_task]
            _param_choice = "autotrain" if params_data[param_choice] == "AutoTrain" else "manual"
            _model_choice = "autotrain" if params_data[model_choice] == "AutoTrain" else "hub_model"
            if _model_choice == "hub_model" and params_data[hub_model] is None:
                logger.error("Hub model is None")
                return
            _training_params = {}
            params = Params(task=_task, param_choice=_param_choice, model_choice=_model_choice)
            params = params.get()
            for _param in hyperparameters:
                if _param.elem_id in params.keys():
                    _training_params[_param.elem_id] = params_data[_param]
            _training_params_md = json.loads(params_data[training_params_txt])
            if _param_choice == "autotrain":
                if len(_training_params_md) > 0:
                    _training_params_md[0] = _training_params
                    _training_params_md = _training_params_md[:1]
                else:
                    _training_params_md.append(_training_params)
            else:
                _training_params_md.append(_training_params)
            params_df = pd.DataFrame(_training_params_md)
            # remove hub_model column
            if "hub_model" in params_df.columns:
                params_df = params_df.drop(columns=["hub_model"])
            return [
                gr.DataFrame.update(value=params_df, visible=True),
                gr.Textbox.update(value=json.dumps(_training_params_md), visible=False),
            ]

        add_job_button.click(
            _add_job,
            inputs=set(
                [task_type, param_choice, model_choice] + hyperparameters + [training_params_md, training_params_txt]
            ),
            outputs=[training_params_md, training_params_txt],
        )
        col_map_components = [
            col_map_text,
            col_map_target,
            concept_token,
        ]
        training_data.change(
            _update_col_map,
            inputs=[training_data, task_type],
            outputs=col_map_components,
        )
        task_type.change(
            _update_col_map,
            inputs=[training_data, task_type],
            outputs=col_map_components,
        )
        estimate_costs_inputs = [
            training_data,
            validation_data,
            task_type,
            user_token,
            autotrain_username,
            training_params_txt,
            autotrain_backend,
        ]
        estimate_costs_outputs = [estimated_costs_md, estimated_costs_num]
        training_data.change(_estimate_costs, inputs=estimate_costs_inputs, outputs=estimate_costs_outputs)
        validation_data.change(_estimate_costs, inputs=estimate_costs_inputs, outputs=estimate_costs_outputs)
        training_params_txt.change(_estimate_costs, inputs=estimate_costs_inputs, outputs=estimate_costs_outputs)
        task_type.change(_estimate_costs, inputs=estimate_costs_inputs, outputs=estimate_costs_outputs)
        add_job_button.click(_estimate_costs, inputs=estimate_costs_inputs, outputs=estimate_costs_outputs)

        create_project_button.click(disable_create_project_button, None, create_project_button).then(
            _create_project,
            inputs=[
                autotrain_username,
                valid_can_pay,
                project_name,
                user_token,
                task_type,
                training_data,
                validation_data,
                col_map_text,
                col_map_target,
                concept_token,
                training_params_txt,
                hub_model,
                estimated_costs_num,
                autotrain_backend,
            ],
            outputs=final_output,
        )

        demo.load(
            _update_project_name,
            outputs=[project_name, create_project_button],
        )

    return demo
