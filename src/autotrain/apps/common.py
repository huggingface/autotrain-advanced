import os

import gradio as gr

from autotrain import allowed_file_types
from autotrain.apps.utils import BACKEND_CHOICES, _login_user
from autotrain.utils import get_user_token


def user_validation():
    user_token = os.environ.get("HF_TOKEN", "")

    if len(user_token) == 0:
        user_token = get_user_token()

    if user_token is None:
        gr.Markdown(
            """Please login with a write [token](https://huggingface.co/settings/tokens).
            Pass your HF token in an environment variable called `HF_TOKEN` and then restart this app.
            """
        )
        raise ValueError("Please login with a write token.")

    user_token, valid_can_pay, who_is_training = _login_user(user_token)

    if user_token is None or len(user_token) == 0:
        gr.Error("Please login with a write token.")

    user_token = gr.Textbox(value=user_token, type="password", lines=1, max_lines=1, visible=False, interactive=False)
    valid_can_pay = gr.Textbox(value=",".join(valid_can_pay), visible=False, interactive=False)

    return user_token, valid_can_pay, who_is_training


def base_components(who_is_training):
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
                        choices=[],
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

    return autotrain_username, project_name, model_choice, autotrain_backend


def train_valid_components():
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
    return training_data, validation_data
