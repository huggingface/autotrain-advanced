import json
import os
import random
import string
import subprocess

import ipywidgets as widgets
import yaml

from autotrain.app.models import fetch_models
from autotrain.app.params import get_task_params


def generate_random_string():
    prefix = "autotrain"
    part1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    part2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{prefix}-{part1}-{part2}"


def colab_app():
    if not os.path.exists("data"):
        os.makedirs("data")
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
        "ST Pair",
        "ST Pair Classification",
        "ST Pair Scoring",
        "ST Triplet",
        "ST Question Answering",
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
        "ST Pair": "st:pair",
        "ST Pair Classification": "st:pair_class",
        "ST Pair Scoring": "st:pair_score",
        "ST Triplet": "st:triplet",
        "ST Question Answering": "st:qa",
    }

    def _get_params(task, param_type):
        _p = get_task_params(task, param_type=param_type)
        _p["push_to_hub"] = True
        _p = json.dumps(_p, indent=4)
        return _p

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
        value=_get_params("llm:sft", "basic"),
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

    _main_layout = widgets.HBox([left_column, separator, right_column])
    main_layout = widgets.VBox([_main_layout, start_training_button])

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
        parameters.value = _get_params(task, param_type)

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
        elif task == "st:pair":
            col_mapping.value = '{"sentence1": "anchor", "sentence2": "positive"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "st:pair_class":
            col_mapping.value = '{"sentence1": "premise", "sentence2": "hypothesis", "target": "label"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "st:pair_score":
            col_mapping.value = '{"sentence1": "sentence1", "sentence2": "sentence2", "target": "score"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "st:triplet":
            col_mapping.value = '{"sentence1": "anchor", "sentence2": "positive", "sentence3": "negative"}'
            dataset_source_dropdown.disabled = False
            valid_split.disabled = False
        elif task == "st:qa":
            col_mapping.value = '{"sentence1": "query", "sentence1": "answer"}'
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
        elif TASK_MAP[task_dropdown.value].startswith("st:"):
            base_model.value = MODEL_CHOICES["sentence-transformers"][0]
        else:
            base_model.value = "Enter base model..."

    def start_training(b):
        start_training_button.disabled = True
        try:
            print("Training is starting... Please wait!")
            os.environ["HF_USERNAME"] = hf_user.value
            os.environ["HF_TOKEN"] = hf_token.value
            train_split_value = train_split.value.strip() if train_split.value.strip() != "" else None
            valid_split_value = valid_split.value.strip() if valid_split.value.strip() != "" else None
            params_val = json.loads(parameters.value)
            if task_dropdown.value.startswith("llm") or task_dropdown.value.startswith("sentence-transformers"):
                params_val["trainer"] = task_dropdown.value.split(":")[1]
                params_val = {k: v for k, v in params_val.items() if k != "trainer"}

            if TASK_MAP[task_dropdown.value] == "dreambooth":
                prompt = params_val.get("prompt")
                if prompt is None:
                    raise ValueError("Prompt is required for DreamBooth task")
                if not isinstance(prompt, str):
                    raise ValueError("Prompt should be a string")
                params_val = {k: v for k, v in params_val.items() if k != "prompt"}
            else:
                prompt = None

            push_to_hub = params_val.get("push_to_hub", True)
            if "push_to_hub" in params_val:
                params_val = {k: v for k, v in params_val.items() if k != "push_to_hub"}

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
                        "column_mapping": json.loads(col_mapping.value),
                    },
                    "params": params_val,
                    "hub": {
                        "username": "${{HF_USERNAME}}",
                        "token": "${{HF_TOKEN}}",
                        "push_to_hub": push_to_hub,
                    },
                }
            else:
                config = {
                    "task": TASK_MAP[task_dropdown.value],
                    "base_model": base_model.value,
                    "project_name": project_name.value,
                    "backend": "local",
                    "data": {
                        "path": dataset_path.value,
                        "prompt": prompt,
                    },
                    "params": params_val,
                    "hub": {
                        "username": "${HF_USERNAME}",
                        "token": "${HF_TOKEN}",
                        "push_to_hub": push_to_hub,
                    },
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

            poll_res = process.poll()
            if poll_res != 0:
                start_training_button.disabled = False
                raise Exception(f"Training failed with exit code: {poll_res}")
            print("Training completed successfully!")
            start_training_button.disabled = False
        except Exception as e:
            print("An error occurred while starting training!")
            print(f"Error: {e}")
            start_training_button.disabled = False

    start_training_button.on_click(start_training)
    dataset_source_dropdown.observe(on_dataset_change, names="value")
    task_dropdown.observe(update_col_mapping, names="value")
    task_dropdown.observe(update_parameters, names="value")
    task_dropdown.observe(update_base_model, names="value")
    parameters_dropdown.observe(update_parameters, names="value")
    return main_layout
