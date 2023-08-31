import copy
import random
import string

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import list_models

from autotrain import logger
from autotrain.utils import user_authentication


THEME = "freddyaboulton/dracula_revamped"

BACKEND_CHOICES = {
    "A10G Large": 3.15,
    "A10G Small": 1.05,
    "A100 Large": 4.13,
    "T4 Medium": 0.9,
    "T4 Small": 0.6,
    "CPU Upgrade": 0.03,
    "CPU (Free)": 0.0,
    # "Local": 0.0,
    # "AutoTrain": -1,
}


def estimate_cost():
    pass


def _update_hub_model_choices(task):
    logger.info(f"Updating hub model choices for task: {task}")
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
    elif task == "tabular":
        return gr.Dropdown.update(
            choices=[],
            visible=False,
            interactive=False,
        )
    else:
        raise NotImplementedError
    # sort by number of downloads in descending order
    hub_models = [{"id": m.modelId, "downloads": m.downloads} for m in hub_models if m.private is False]
    hub_models = sorted(hub_models, key=lambda x: x["downloads"], reverse=True)

    if task == "dreambooth":
        choices = [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-1-base",
        ]
        value = choices[0]
        return gr.Dropdown.update(
            choices=choices,
            value=value,
            visible=True,
            interactive=True,
        )

    # if task in ("text_multi_class_classification", "image_multi_class_classification"):
    #     choices = ["AutoTrain"] + [m["id"] for m in hub_models]
    # else:
    choices = [m["id"] for m in hub_models]

    return gr.Dropdown.update(
        choices=choices,
        value=choices[0],
        visible=True,
        interactive=True,
    )


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


def fetch_training_params_df(
    param_choice, jobs_df, training_params, model_choice, autotrain_backend, hide_model_param=False
):
    if param_choice == "AutoTrain":
        # create a new dataframe from dict
        _training_params_df = pd.DataFrame([training_params])
    else:
        # add row to the dataframe
        if len(jobs_df) == 0:
            _training_params_df = pd.DataFrame([training_params])
        else:
            _training_params_df = copy.deepcopy(jobs_df)
            _training_params_df.columns = [f"hyp_{c}" for c in _training_params_df.columns]
            # convert dataframe to list of dicts
            _training_params_df = _training_params_df.to_dict(orient="records")
            # append new dict to the list
            _training_params_df.append(training_params)
            _training_params_df = pd.DataFrame(_training_params_df)
            # drop rows with all nan values
            _training_params_df.replace("", np.nan, inplace=True)
            # Drop rows with all missing values
            _training_params_df = _training_params_df.dropna(how="all")
            # Drop columns with all missing values
            _training_params_df = _training_params_df.dropna(axis=1, how="all")

    # remove hyp_ from column names
    _training_params_df.columns = [c[len("hyp_") :] for c in _training_params_df.columns]
    _training_params_df = _training_params_df.reset_index(drop=True)
    if not hide_model_param:
        _training_params_df.loc[:, "model_choice"] = model_choice
        _training_params_df.loc[:, "param_choice"] = param_choice
    _training_params_df.loc[:, "backend"] = autotrain_backend
    return _training_params_df


def clear_jobs(jobs_df):
    return gr.DataFrame.update(visible=False, interactive=False, value=pd.DataFrame())


def handle_model_choice_change(model_choice):
    op = []
    op.append(gr.DataFrame.update(value=pd.DataFrame(), visible=False, interactive=False))
    if model_choice == "AutoTrain":
        op.append(gr.Dropdown.update(value="AutoTrain", interactive=False))
        op.append(gr.Dropdown.update(value="AutoTrain", interactive=False))
    else:
        op.append(gr.Dropdown.update(value="Manual", interactive=True))
        op.append(gr.Dropdown.update(value=list(BACKEND_CHOICES.keys())[0], interactive=True))
    return op
