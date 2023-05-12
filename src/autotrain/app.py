import gradio as gr
import os

from autotrain import help
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.params import Params
from autotrain.project import Project
from autotrain.tasks import COLUMN_MAPPING
from autotrain.utils import app_error_handler, get_project_cost, get_user_token, user_authentication
from loguru import logger


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


with gr.Blocks(css=".tabitem {padding: 25px}") as demo:
    gr.Markdown("## 🤗 AutoTrain Advanced")
    user_token = os.environ.get("HF_TOKEN", "")
    
    if len(user_token) == 0:
        user_token = get_user_token()
    
    if user_token is None:
        gr.Markdown(
            """Please login with a write [token](https://huggingface.co/settings/tokens).
            You can also pass your HF token in an environment variable called `HF_TOKEN` to avoid having to enter it every time.
            """
        )
        user_token_input = gr.Textbox(label="HuggingFace Token", value="", type="password", lines=1, max_lines=1)
        user_token = gr.Markdown(visible=)
        user_token_input.submit(_login_user, inputs=[user_token_input])
    else:
        _login_user(user_token)
    # if user_token is None:
    #     gr.Error("Please login with a write token.")

    # if len(user_token) == 0:
    #     gr.Error("Please login with a write token.")

    # logger.info(who_is_training)
