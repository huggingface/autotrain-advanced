import json
import os
import random
import string
import subprocess

import ipywidgets as widgets
import requests
import yaml

from autotrain.app.models import fetch_models
from autotrain.app.params import get_task_params


def generate_random_string():
    prefix = "autotrain"
    part1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    part2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{prefix}-{part1}-{part2}"


def colab_app():
    MODEL_CHOICES = fetch_models()
    TASK_NAMES = [
        "LLM SFT",
        "LLM ORPO",
        "LLM Generic",
        "LLM DPO",
        "LLM Reward",
        "Text Classification",
        "Text Regression",
        "Sequence to Sequence",
        "Token Classification",
        "DreamBooth LoRA",
        "Image Classification",
        "Object Detection",
        "Tabular Classification",
        "Tabular Regression",
    ]

    TASK_MAP = {
        "LLM SFT": "llm:sft",
        "LLM ORPO": "llm:orpo",
        "LLM Generic": "llm:generic",
        "LLM DPO": "llm:dpo",
        "LLM Reward": "llm:reward",
        "Text Classification": "text-classification",
        "Text Regression": "text-regression",
        "Sequence to Sequence": "seq2seq",
        "Token Classification": "token-classification",
        "DreamBooth LoRA": "dreambooth",
        "Image Classification": "image-classification",
        "Object Detection": "image-object-detection",
        "Tabular Classification": "tabular:classification",
        "Tabular Regression": "tabular:regression",
    }

    hf_token_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Hugging Face Write Token</h5>")
    hf_token = widgets.Password(
        value="", description="", disabled=False, layout=widgets.Layout(margin="0 0 0 0", width="200px")
    )

    hf_user_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Hugging Face Username</h5>")
    hf_user = widgets.Text(
        value="", description="", disabled=False, layout=widgets.Layout(margin="0 0 0 0", width="200px")
    )

    base_model_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Base Model</h5>")
    base_model = widgets.Text(value=MODEL_CHOICES["llm"][0], disabled=False, layout=widgets.Layout(width="420px"))

    project_name_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Project Name</h5>")
    project_name = widgets.Text(
        value=generate_random_string(),
        description="",
        disabled=False,
        layout=widgets.Layout(margin="0 0 0 0", width="200px"),
    )

    task_dropdown_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Task</h5>")
    task_dropdown = widgets.Dropdown(
        options=TASK_NAMES,
        value=TASK_NAMES[0],
        description="",
        disabled=False,
        layout=widgets.Layout(margin="0 0 0 0", width="200px"),
    )

    dataset_path_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Path</h5>")
    dataset_path = widgets.Text(
        value="", description="", disabled=False, layout=widgets.Layout(margin="0 0 0 0", width="200px")
    )

    train_split_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Train Split</h5>")
    train_split = widgets.Text(
        value="", description="", disabled=False, layout=widgets.Layout(margin="0 0 0 0", width="200px")
    )

    valid_split_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Valid Split</h5>")
    valid_split = widgets.Text(
        value="",
        placeholder="optional",
        description="",
        disabled=False,
        layout=widgets.Layout(margin="0 0 0 0", width="200px"),
    )

    dataset_source_dropdown_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Source</h5>")
    dataset_source_dropdown = widgets.Dropdown(
        options=["Hugging Face Hub", "Local"],
        value="Hugging Face Hub",
        description="",
        disabled=False,
        layout=widgets.Layout(margin="0 0 0 0", width="200px"),
    )

    col_mapping_label = widgets.HTML("<h5 style='margin-bottom: 0; margin-top: 0;'>Column Mapping</h5>")
    col_mapping = widgets.Text(
        value='{"text": "text"}',
        placeholder="",
        description="",
        disabled=False,
        layout=widgets.Layout(margin="0 0 0 0", width="420px"),
    )

    parameters_dropdown = widgets.Dropdown(
        options=["Basic", "Full"], value="Basic", description="", disabled=False, layout=widgets.Layout(width="400px")
    )

    parameters = widgets.Textarea(
        value=json.dumps(get_task_params("llm:sft", param_type="basic"), indent=4),
        description="",
        disabled=False,
        layout=widgets.Layout(height="400px", width="400px"),
    )

    start_training_button = widgets.Button(
        description="Start Training",
        layout=widgets.Layout(width="1000px"),
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click to start training",
        icon="check",  # (FontAwesome names without the `fa-` prefix)
    )

    spacer = widgets.Box(layout=widgets.Layout(width="20px"))
    title_hbox0 = widgets.HTML("<h3>Hugging Face Credentials</h3>")
    title_hbox1 = widgets.HTML("<h3>Project Details</h3>")
    title_hbox2 = widgets.HTML("<h3>Dataset Details</h3>")
    title_hbox3 = widgets.HTML("<h3>Parameters</h3>")

    hbox0 = widgets.HBox(
        [
            widgets.VBox([hf_token_label, hf_token]),
            spacer,
            widgets.VBox([hf_user_label, hf_user]),
        ]
    )
    hbox1 = widgets.HBox(
        [
            widgets.VBox([project_name_label, project_name]),
            spacer,
            widgets.VBox([task_dropdown_label, task_dropdown]),
        ]
    )
    hbox2_1 = widgets.HBox(
        [
            widgets.VBox([dataset_source_dropdown_label, dataset_source_dropdown]),
            spacer,
            widgets.VBox([dataset_path_label, dataset_path]),
        ]
    )
    hbox2_2 = widgets.HBox(
        [
            widgets.VBox([train_split_label, train_split]),
            spacer,
            widgets.VBox([valid_split_label, valid_split]),
        ]
    )
    hbox2_3 = widgets.HBox(
        [
            widgets.VBox([col_mapping_label, col_mapping]),
        ]
    )
    hbox3 = widgets.VBox([parameters_dropdown, parameters])

    vbox0 = widgets.VBox([title_hbox0, hbox0])
    vbox1 = widgets.VBox([title_hbox1, base_model_label, base_model, hbox1])
    vbox2 = widgets.VBox([title_hbox2, hbox2_1, hbox2_2, hbox2_3])
    vbox3 = widgets.VBox([title_hbox3, hbox3])

    left_column = widgets.VBox([vbox0, vbox1, vbox2], layout=widgets.Layout(width="500px"))
    right_column = widgets.VBox([vbox3], layout=widgets.Layout(width="500px", align_items="flex-end"))

    separator = widgets.HTML('<div style="border-left: 1px solid black; height: 100%;"></div>')

    logo = widgets.Image(
        value=requests.get(
            "https://raw.githubusercontent.com/huggingface/autotrain-advanced/main/src/autotrain/app/static/logo.png"
        ).content,
        format="png",
        layout=widgets.Layout(width="300px"),
    )
    logo_box = widgets.VBox([logo], layout=widgets.Layout(align_items="flex-start"))

    _main_layout = widgets.HBox([left_column, separator, right_column])
    main_layout = widgets.VBox([logo_box, _main_layout, start_training_button])

    def on_dataset_change(change):
        if change["new"] == "Local":
            dataset_path.value = "data/"
            train_split.value = "train"
            valid_split.value = ""
        else:
            dataset_path.value = ""
            train_split.value = ""
            valid_split.value = ""

    def update_parameters(*args):
        task = TASK_MAP[task_dropdown.value]
        param_type = parameters_dropdown.value.lower()
        parameters.value = json.dumps(get_task_params(task, param_type), indent=4)

    def update_col_mapping(*args):
        task = TASK_MAP[task_dropdown.value]
        if task in ["llm:sft", "llm:generic"]:
            col_mapping.value = '{"text": "text"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = True
        elif task in ["llm:dpo", "llm:orpo"]:
            col_mapping.value = '{"prompt": "prompt", "text": "text", "rejected_text": "rejected_text"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = True
        elif task == "llm:reward":
            col_mapping.value = '{"text": "text", "rejected_text": "rejected_text"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = True
        elif task == "text-classification":
            col_mapping.value = '{"text": "text", "label": "target"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "text-regression":
            col_mapping.value = '{"text": "text", "label": "target"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "token-classification":
            col_mapping.value = '{"text": "tokens", "label": "tags"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "seq2seq":
            col_mapping.value = '{"text": "text", "label": "target"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "dreambooth":
            col_mapping.value = '{"image": "image"}'
            dataset_source_dropdown.value = "Local"
            dataset_source_dropdown.disabled = True
            valid_split.disabled = True
        elif task == "image-classification":
            col_mapping.value = '{"image": "image", "label": "label"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "image-object-detection":
            col_mapping.value = '{"image": "image", "objects": "objects"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "tabular:classification":
            col_mapping.value = '{"id": "id", "label": ["target"]}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "tabular:regression":
            col_mapping.value = '{"id": "id", "label": ["target"]}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        else:
            col_mapping.value = "Enter column mapping..."

    def update_base_model(*args):
        if TASK_MAP[task_dropdown.value] == "text-classification":
            base_model.value = MODEL_CHOICES["text-classification"][0]
        elif TASK_MAP[task_dropdown.value].startswith("llm"):
            base_model.value = MODEL_CHOICES["llm"][0]
        elif TASK_MAP[task_dropdown.value] == "image-classification":
            base_model.value = MODEL_CHOICES["image-classification"][0]
        elif TASK_MAP[task_dropdown.value] == "dreambooth":
            base_model.value = MODEL_CHOICES["dreambooth"][0]
        elif TASK_MAP[task_dropdown.value] == "seq2seq":
            base_model.value = MODEL_CHOICES["seq2seq"][0]
        elif TASK_MAP[task_dropdown.value] == "tabular:classification":
            base_model.value = MODEL_CHOICES["tabular-classification"][0]
        elif TASK_MAP[task_dropdown.value] == "tabular:regression":
            base_model.value = MODEL_CHOICES["tabular-regression"][0]
        elif TASK_MAP[task_dropdown.value] == "token-classification":
            base_model.value = MODEL_CHOICES["token-classification"][0]
        elif TASK_MAP[task_dropdown.value] == "text-regression":
            base_model.value = MODEL_CHOICES["text-regression"][0]
        elif TASK_MAP[task_dropdown.value] == "image-object-detection":
            base_model.value = MODEL_CHOICES["image-object-detection"][0]
        else:
            base_model.value = "Enter base model..."

    def start_training(b):
        os.environ["HF_USERNAME"] = hf_user.value
        os.environ["HF_TOKEN"] = hf_token.value
        train_split_value = train_split.value.strip() if train_split.value.strip() != "" else None
        valid_split_value = valid_split.value.strip() if valid_split.value.strip() != "" else None
        params_val = json.loads(parameters.value)
        if task_dropdown.value.startswith("llm"):
            params_val["trainer"] = task_dropdown.value.split(":")[1]

        if TASK_MAP[task_dropdown.value] != "dreambooth":
            config = {
                "task": TASK_MAP[task_dropdown.value].split(":")[0],
                "base_model": base_model.value,
                "project_name": project_name.value,
                "log": "tensorboard",
                "backend": "local",
                "data": {
                    "path": dataset_path.value,
                    "train_split": train_split_value,
                    "valid_split": valid_split_value,
                    "col_mapping": json.loads(col_mapping.value),
                },
                "params": params_val,
                "hub": {"username": "${{HF_USERNAME}}", "token": "${{HF_TOKEN}}"},
            }
        else:
            config = {
                "task": TASK_MAP[task_dropdown.value],
                "base_model": base_model.value,
                "project_name": project_name.value,
                "backend": "local",
                "data": {
                    "path": dataset_path.value,
                    "prompt": params_val["prompt"],
                },
                "params": params_val,
                "hub": {"username": "${HF_USERNAME}", "token": "${HF_TOKEN}"},
            }

        with open("config.yml", "w") as f:
            yaml.dump(config, f)

        cmd = "autotrain --config config.yml"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        _ = process.poll()

    start_training_button.on_click(start_training)

    # Observe changes in task_dropdown to update col_mapping
    dataset_source_dropdown.observe(on_dataset_change, names="value")
    task_dropdown.observe(update_col_mapping, names="value")
    task_dropdown.observe(update_parameters, names="value")
    task_dropdown.observe(update_base_model, names="value")
    parameters_dropdown.observe(update_parameters, names="value")

    return main_layout
