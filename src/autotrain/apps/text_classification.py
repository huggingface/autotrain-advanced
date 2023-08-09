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
    "bert-base-uncased",
    "AutoTrain",
]


def _update_project_name():
    random_project_name = "-".join(
        ["".join(random.choices(string.ascii_lowercase + string.digits, k=4)) for _ in range(3)]
    )
    return gr.Text.update(value=random_project_name, visible=True, interactive=True)


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
        gr.Markdown("### 🚀 Text Classification")
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
                    with gr.Row():
                        project_name = gr.Textbox(
                            label="Project name",
                            value="",
                            lines=1,
                            max_lines=1,
                            interactive=True,
                        )
                        model_choice = gr.Dropdown(
                            label="Model Choice",
                            choices=MODEL_CHOICES,
                            value=MODEL_CHOICES[0],
                            visible=True,
                            interactive=True,
                            allow_custom_value=True,
                        )
                        autotrain_backend = gr.Dropdown(
                            label="Backend",
                            choices=list(BACKEND_CHOICES.keys()),
                            value=list(BACKEND_CHOICES.keys())[0],
                            interactive=True,
                        )
        gr.Markdown("<hr>")
        with gr.Row():
            with gr.Column():
                with gr.Tabs(elem_id="tabs"):
                    with gr.TabItem("Training Data"):
                        training_data = gr.File(
                            label="Training Data",
                            file_types=allowed_file_types.TEXT_CLASSIFICATION,
                            file_count="multiple",
                            visible=True,
                            interactive=True,
                            elem_id="training_data",
                        )
                    with gr.TabItem("Validation Data (Optional)"):
                        validation_data = gr.File(
                            label="Validation Data",
                            file_types=allowed_file_types.TEXT_CLASSIFICATION,
                            file_count="multiple",
                            visible=True,
                            interactive=True,
                            elem_id="validation_data",
                        )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        col_map_text = gr.Dropdown(
                            label="Text Column",
                            choices=[],
                            visible=True,
                            interactive=True,
                            elem_id="col_map_text",
                        )
                        col_map_target = gr.Dropdown(
                            label="Target Column",
                            choices=[],
                            visible=True,
                            interactive=True,
                            elem_id="col_map_target",
                        )
                    with gr.Row():
                        hyp_scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["cosine", "linear", "constant"],
                            value="linear",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_scheduler",
                        )
                        hyp_optimizer = gr.Dropdown(
                            label="Optimizer",
                            choices=["adamw_torch", "adamw_hf", "sgd", "adafactor", "adagrad"],
                            value="adamw_torch",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_optimizer",
                        )

            with gr.Column():
                with gr.Group():
                    param_choice = gr.Dropdown(
                        label="Parameter Choice",
                        choices=["Manual", "AutoTrain"],
                        value="Manual",
                        visible=True,
                        interactive=True,
                        elem_id="param_choice",
                    )
                    with gr.Row():
                        hyp_language = gr.Dropdown(
                            label="Language",
                            choices=SUPPORTED_LANGUAGES,
                            value="en",
                            visible=False,
                            interactive=False,
                            elem_id="hyp_language",
                        )
                    with gr.Row():
                        hyp_num_jobs = gr.Number(
                            label="Num Jobs",
                            value=5,
                            visible=False,
                            interactive=False,
                            elem_id="hyp_num_jobs",
                            precision=0,
                        )
                    with gr.Row():
                        hyp_learning_rate = gr.Number(
                            label="Learning Rate",
                            value=5e-5,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_learning_rate",
                        )
                        hyp_num_train_epochs = gr.Number(
                            label="Epochs",
                            value=3,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_num_train_epochs",
                        )
                    with gr.Row():
                        hyp_max_seq_length = gr.Number(
                            label="Max Seq Length",
                            value=512,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_max_seq_length",
                        )
                        hyp_batch_size = gr.Number(
                            label="Batch Size",
                            value=8,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_batch_size",
                        )
                    with gr.Row():
                        hyp_percentage_warmup = gr.Number(
                            label="Warmup Steps %",
                            value=0.1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_percentage_warmup",
                        )
                        hyp_weight_decay = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_weight_decay",
                        )
                    with gr.Row():
                        hyp_gradient_accumulation_steps = gr.Number(
                            label="Grad Acc Steps",
                            value=1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_gradient_accumulation_steps",
                        )

        with gr.Row():
            add_job_button = gr.Button(value="Add Job", elem_id="add_job_button")
            clear_jobs_button = gr.Button(value="Clear Jobs", elem_id="clear_jobs_button")
            start_training_button = gr.Button(value="Start Training", elem_id="start_training_button")

        jobs_df = gr.DataFrame(visible=False, interactive=False)

        hyperparameters = [
            hyp_scheduler,
            hyp_optimizer,
            hyp_learning_rate,
            hyp_num_train_epochs,
            hyp_max_seq_length,
            hyp_batch_size,
            hyp_percentage_warmup,
            hyp_weight_decay,
            hyp_gradient_accumulation_steps,
            hyp_language,
            hyp_num_jobs,
        ]

        # handle all change events here

        # model_choice change
        # change in model_choice to AutoTrain should remove all the jobs,
        # and change the param choice to AutoTrain
        def _handle_model_choice_change(components):
            op = []
            op.append(jobs_df.update(value=pd.DataFrame(), visible=False, interactive=False))
            if components[model_choice] == "AutoTrain":
                op.append(param_choice.update(value="AutoTrain", interactive=False))
                op.append(autotrain_backend.update(value="AutoTrain", interactive=False))
            else:
                op.append(param_choice.update(value="Manual", interactive=True))
                op.append(autotrain_backend.update(value=list(BACKEND_CHOICES.keys())[0], interactive=True))
            return op

        model_choice.change(
            _handle_model_choice_change,
            inputs=set([model_choice]),
            outputs=[jobs_df, param_choice, autotrain_backend],
        )

        def _handle_param_choice_change(components):
            hyperparam_visibility = {}
            if components[param_choice] == "AutoTrain":
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id in ["hyp_num_jobs", "hyp_language"]:
                        hyperparam_visibility[_hyperparameter.elem_id] = True
                    else:
                        hyperparam_visibility[_hyperparameter.elem_id] = False
            else:
                for _hyperparameter in hyperparameters:
                    if _hyperparameter.elem_id in ["hyp_num_jobs", "hyp_language"]:
                        hyperparam_visibility[_hyperparameter.elem_id] = False
                    else:
                        hyperparam_visibility[_hyperparameter.elem_id] = True
            op = [
                h.update(
                    interactive=hyperparam_visibility.get(h.elem_id, False),
                    visible=hyperparam_visibility.get(h.elem_id, False),
                )
                for h in hyperparameters
            ]
            op.append(jobs_df.update(value=pd.DataFrame(), visible=False, interactive=False))
            return op

        param_choice.change(
            _handle_param_choice_change,
            inputs=set([param_choice]),
            outputs=hyperparameters + [jobs_df],
        )

        def _add_job(components):
            if len(str(components[col_map_text].strip())) == 0:
                raise gr.Error("Text column cannot be empty.")
            if len(str(components[col_map_target].strip())) == 0:
                raise gr.Error("Target column cannot be empty.")
            if components[col_map_text] == components[col_map_target]:
                raise gr.Error("Text and Target column cannot be the same.")

        add_job_button.click(
            _add_job,
            inputs=set([param_choice, col_map_text, col_map_target] + hyperparameters),
            outputs=jobs_df,
        )

        demo.load(_update_project_name, outputs=project_name)

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
