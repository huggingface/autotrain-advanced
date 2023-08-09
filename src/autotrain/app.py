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
from autotrain import allowed_file_types

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

BACKEND_CHOICES = {
    "A10G Large": 3.15,
    "A10G Small": 1.05,
    "A100 Large": 4.13,
    "T4 Medium": 0.9,
    "T4 Small": 0.6,
    "CPU Upgrade": 0.03,
    "CPU": 0.0,
    "Local": 0.0,
    "AutoTrain": -1,
}

MODEL_CHOICES = [
    "HuggingFace Hub",
    "AutoTrain",
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
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        autotrain_username = gr.Dropdown(
                            label="AutoTrain Username",
                            choices=who_is_training,
                            value=who_is_training[0] if who_is_training else "",
                            interactive=True,
                        )
                        autotrain_backend = gr.Dropdown(
                            label="AutoTrain Backend",
                            choices=list(BACKEND_CHOICES.keys()),
                            value=list(BACKEND_CHOICES.keys())[0],
                            interactive=True,
                        )
                    with gr.Row():
                        project_name = gr.Textbox(
                            label="Project name",
                            value="",
                            lines=1,
                            max_lines=1,
                            interactive=True,
                        )
                        task_type = gr.Dropdown(
                            label="Task",
                            choices=list(APP_TASKS_MAPPING.keys()),
                            value=list(APP_TASKS_MAPPING.keys())[0],
                            interactive=True,
                        )
                        model_choice = gr.Dropdown(
                            label="Model Choice",
                            choices=MODEL_CHOICES,
                            value=MODEL_CHOICES[0],
                            visible=True,
                            interactive=True,
                        )
                        param_choice = gr.Dropdown(
                            label="Parameter Choice",
                            choices=["Manual", "AutoTrain"],
                            value="Manual",
                            visible=True,
                            interactive=True,
                            elem_id="param_choice",
                        )
                    with gr.Row():
                        hub_model = gr.Dropdown(
                            label="Hub Model",
                            value="",
                            visible=True,
                            interactive=True,
                            elem_id="hub_model",
                        )
        gr.Markdown("<hr>")
        with gr.Row():
            with gr.Column():
                txtcl_training_data = gr.File(
                    label="Training Data",
                    file_types=allowed_file_types.TEXT_CLASSIFICATION,
                    file_count="multiple",
                    visible=True,
                    interactive=True,
                    elem_id="txtcl_training_data",
                )
                with gr.Accordion(open=False, label="Validation Data [optional]", elem_id="txtcl_data_accordion"):
                    txtcl_validation_data = gr.File(
                        label="Validation Data",
                        file_types=allowed_file_types.TEXT_CLASSIFICATION,
                        file_count="multiple",
                        visible=True,
                        interactive=True,
                        elem_id="txtcl_validation_data",
                    )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        txtcl_col_map_text = gr.Dropdown(
                            label="Text Column",
                            choices=[],
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_col_map_text",
                        )
                        txtcl_col_map_target = gr.Dropdown(
                            label="Target Column",
                            choices=[],
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_col_map_target",
                        )
                        llm_col_map_text = gr.Dropdown(
                            label="Text Column",
                            choices=[],
                            visible=False,
                            interactive=True,
                            elem_id="llm_col_map_text",
                        )
                    with gr.Row():
                        txtcl_hyp_scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["cosine", "linear", "constant"],
                            value="linear",
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_scheduler",
                        )
                        llm_hyp_scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["cosine", "linear", "constant"],
                            value="linear",
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_scheduler",
                        )
                        txtcl_hyp_optimizer = gr.Dropdown(
                            label="Optimizer",
                            choices=["adamw_torch", "adamw_hf", "sgd", "adafactor", "adagrad"],
                            value="adamw_torch",
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_optimizer",
                        )
                        llm_hyp_optimizer = gr.Dropdown(
                            label="Optimizer",
                            choices=["adamw_torch", "adamw_hf", "sgd", "adafactor", "adagrad"],
                            value="adamw_torch",
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_optimizer",
                        )
                    with gr.Row():
                        llm_hyp_trainer_type = gr.Dropdown(
                            label="Trainer Type",
                            choices=["Default", "SFT"],
                            value="SFT",
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_trainer_type",
                        )
                    with gr.Row():
                        llm_hyp_use_peft = gr.Checkbox(
                            label="Use PEFT",
                            value=True,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_use_peft",
                        )
                        llm_hyp_fp16 = gr.Checkbox(
                            label="FP16",
                            value=True,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_fp16",
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        txtcl_hyp_learning_rate = gr.Number(
                            label="Learning Rate",
                            value=5e-5,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_learning_rate",
                        )
                        llm_hyp_learning_rate = gr.Number(
                            label="Learning Rate",
                            value=1e-4,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_learning_rate",
                        )
                        txtcl_hyp_num_train_epochs = gr.Number(
                            label="Epochs",
                            value=3,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_num_train_epochs",
                        )
                        llm_hyp_num_train_epochs = gr.Number(
                            label="Epochs",
                            value=3,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_num_train_epochs",
                        )
                    with gr.Row():
                        txtcl_hyp_max_seq_length = gr.Number(
                            label="Max Seq Length",
                            value=512,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_max_seq_length",
                        )
                        llm_hyp_max_seq_length = gr.Number(
                            label="Max Seq Length",
                            value=1024,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_max_seq_length",
                        )
                        txtcl_hyp_batch_size = gr.Number(
                            label="Batch Size",
                            value=8,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_batch_size",
                        )
                        llm_hyp_batch_size = gr.Number(
                            label="Batch Size",
                            value=2,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_batch_size",
                        )
                    with gr.Row():
                        txtcl_hyp_percentage_warmup = gr.Number(
                            label="Warmup Steps %",
                            value=0.1,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_percentage_warmup",
                        )
                        llm_hyp_percentage_warmup = gr.Number(
                            label="Warmup Steps %",
                            value=0.1,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_percentage_warmup",
                        )
                        txtcl_hyp_weight_decay = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_weight_decay",
                        )
                        llm_hyp_weight_decay = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_weight_decay",
                        )
                    with gr.Row():
                        txtcl_hyp_gradient_accumulation_steps = gr.Number(
                            label="Grad Acc Steps",
                            value=1,
                            visible=True,
                            interactive=True,
                            elem_id="txtcl_hyp_gradient_accumulation_steps",
                        )
                        llm_hyp_gradient_accumulation_steps = gr.Number(
                            label="Grad Acc Steps",
                            value=1,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_gradient_accumulation_steps",
                        )
                        llm_hyp_lora_r = gr.Number(
                            label="LoRA R",
                            value=16,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_lora_r",
                            precision=0,
                        )
                        llm_hyp_lora_alpha = gr.Number(
                            label="LoRA Alpha",
                            value=32,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_lora_alpha",
                            precision=0,
                        )
                        llm_hyp_lora_dropout = gr.Number(
                            label="LoRA Dropout",
                            value=0.05,
                            visible=False,
                            interactive=True,
                            elem_id="llm_hyp_lora_dropout",
                        )

        add_job_button = gr.Button(value="Add Job", elem_id="add_job_button")

        jobs_df = gr.DataFrame()

        hyperparameters = [
            # text classification hyperparameters
            txtcl_hyp_scheduler,
            txtcl_hyp_optimizer,
            txtcl_hyp_learning_rate,
            txtcl_hyp_num_train_epochs,
            txtcl_hyp_max_seq_length,
            txtcl_hyp_batch_size,
            txtcl_hyp_percentage_warmup,
            txtcl_hyp_weight_decay,
            txtcl_hyp_gradient_accumulation_steps,
            # llm hyperparameters
            llm_hyp_scheduler,
            llm_hyp_optimizer,
            llm_hyp_learning_rate,
            llm_hyp_num_train_epochs,
            llm_hyp_max_seq_length,
            llm_hyp_batch_size,
            llm_hyp_percentage_warmup,
            llm_hyp_weight_decay,
            llm_hyp_gradient_accumulation_steps,
            llm_hyp_trainer_type,
            llm_hyp_use_peft,
            llm_hyp_fp16,
            llm_hyp_lora_r,
            llm_hyp_lora_alpha,
            llm_hyp_lora_dropout,
        ]

        column_mappings = [
            txtcl_col_map_text,
            txtcl_col_map_target,
            llm_col_map_text,
        ]

        # handle all change events here

        # autotrain_backend change
        # change in backend to AutoTrain should remove all the jobs,
        # and change the param choice to AutoTrain
        def _handle_backend_change(components):
            non_interactive_hyperparameters = []
            if components[autotrain_backend] == "AutoTrain":
                for _hyperparameter in hyperparameters:
                    non_interactive_hyperparameters.append(_hyperparameter.elem_id)
            op = [h.update(interactive=h.elem_id not in non_interactive_hyperparameters) for h in hyperparameters]
            op.append(jobs_df.update(value=pd.DataFrame()))
            op.append(param_choice.update(value="AutoTrain", interactive=False))
            return op

        _inps = [autotrain_backend] + hyperparameters
        _outs = hyperparameters + [jobs_df, param_choice]
        autotrain_backend.change(
            _handle_backend_change,
            inputs=set(_inps),
            outputs=_outs,
        )

        # task_type change
        # change in task type should change all the hyperparameters and column mappings
        def _handle_task_type_change(components):
            interactive_hyperparameters = []
            visible_hyperparameters = []
            if APP_TASKS_MAPPING[components[task_type]] == "text_classification":
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id.startswith("txtcl"):
                        interactive_hyperparameters.append(_hyperparameter.elem_id)
                        visible_hyperparameters.append(_hyperparameter.elem_id)
            elif APP_TASKS_MAPPING[components[task_type]] == "lm_training":
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id.startswith("llm"):
                        interactive_hyperparameters.append(_hyperparameter.elem_id)
                        visible_hyperparameters.append(_hyperparameter.elem_id)
            op = [
                h.update(
                    interactive=h.elem_id in interactive_hyperparameters,
                    visible=h.elem_id in visible_hyperparameters,
                )
                for h in hyperparameters
            ]
            visible_column_mappings = []
            if APP_TASKS_MAPPING[components[task_type]] == "text_classification":
                for _column_mapping in column_mappings:
                    if _column_mapping.elem_id.startswith("txtcl"):
                        visible_column_mappings.append(_column_mapping.elem_id)
            elif APP_TASKS_MAPPING[components[task_type]] == "lm_training":
                for _column_mapping in column_mappings:
                    if _column_mapping.elem_id.startswith("llm"):
                        visible_column_mappings.append(_column_mapping.elem_id)
            op.extend(
                [
                    c.update(
                        visible=c.elem_id in visible_column_mappings,
                    )
                    for c in column_mappings
                ]
            )
            return op

        _inps = [task_type] + hyperparameters
        _outs = hyperparameters + column_mappings
        task_type.change(
            _handle_task_type_change,
            inputs=set(_inps),
            outputs=_outs,
        )

    return demo
