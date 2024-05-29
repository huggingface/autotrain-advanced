import json
from dataclasses import dataclass
from typing import Optional

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


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
    "objects_column",
    "sentence1_column",
    "sentence2_column",
    "sentence3_column",
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
PARAMS["st"] = SentenceTransformersParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["image-classification"] = ImageClassificationParams(
    mixed_precision="fp16",
    log="tensorboard",
).model_dump()
PARAMS["image-object-detection"] = ObjectDetectionParams(
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


@dataclass
class AppParams:
    job_params_json: str
    token: str
    project_name: str
    username: str
    task: str
    data_path: str
    base_model: str
    column_mapping: dict
    train_split: Optional[str] = None
    valid_split: Optional[str] = None
    using_hub_dataset: Optional[bool] = False
    api: Optional[bool] = False

    def __post_init__(self):
        if self.using_hub_dataset and not self.train_split:
            raise ValueError("train_split is required when using a hub dataset")

    def munge(self):
        if self.task == "text-classification":
            return self._munge_params_text_clf()
        elif self.task == "seq2seq":
            return self._munge_params_seq2seq()
        elif self.task == "image-classification":
            return self._munge_params_img_clf()
        elif self.task == "image-object-detection":
            return self._munge_params_img_obj_det()
        elif self.task.startswith("tabular"):
            return self._munge_params_tabular()
        elif self.task == "dreambooth":
            return self._munge_params_dreambooth()
        elif self.task.startswith("llm"):
            return self._munge_params_llm()
        elif self.task == "token-classification":
            return self._munge_params_token_clf()
        elif self.task == "text-regression":
            return self._munge_params_text_reg()
        elif self.task.startswith("st:"):
            return self._munge_params_sent_transformers()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _munge_common_params(self):
        _params = json.loads(self.job_params_json)
        _params["token"] = self.token
        _params["project_name"] = f"{self.project_name}"
        if "push_to_hub" not in _params:
            _params["push_to_hub"] = True
        _params["data_path"] = self.data_path
        _params["username"] = self.username
        return _params

    def _munge_params_sent_transformers(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["sentence1_column"] = "autotrain_sentence1"
            _params["sentence2_column"] = "autotrain_sentence2"
            _params["sentence3_column"] = "autotrain_sentence3"
            _params["target_column"] = "autotrain_target"
            _params["valid_split"] = "validation"
        else:
            _params["sentence1_column"] = self.column_mapping.get(
                "sentence1" if not self.api else "sentence1_column", "sentence1"
            )
            _params["sentence2_column"] = self.column_mapping.get(
                "sentence2" if not self.api else "sentence2_column", "sentence2"
            )
            _params["sentence3_column"] = self.column_mapping.get(
                "sentence3" if not self.api else "sentence3_column", "sentence3"
            )
            _params["target_column"] = self.column_mapping.get("target" if not self.api else "target_column", "target")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        trainer = self.task.split(":")[1]
        _params["trainer"] = trainer.lower()
        return SentenceTransformersParams(**_params)

    def _munge_params_llm(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["prompt_text_column"] = "autotrain_prompt"
            _params["rejected_text_column"] = "autotrain_rejected_text"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["prompt_text_column"] = self.column_mapping.get(
                "prompt" if not self.api else "prompt_text_column", "prompt"
            )
            _params["rejected_text_column"] = self.column_mapping.get(
                "rejected_text" if not self.api else "rejected_text_column", "rejected_text"
            )
            _params["train_split"] = self.train_split
        _params["log"] = "tensorboard"

        trainer = self.task.split(":")[1]
        if trainer != "generic":
            _params["trainer"] = trainer.lower()

        if "quantization" in _params:
            if _params["quantization"] in ("none", "no"):
                _params["quantization"] = None

        return LLMTrainingParams(**_params)

    def _munge_params_text_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
        return TextClassificationParams(**_params)

    def _munge_params_text_reg(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
        return TextRegressionParams(**_params)

    def _munge_params_token_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["tokens_column"] = "autotrain_text"
            _params["tags_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["tokens_column"] = self.column_mapping.get("text" if not self.api else "tokens_column", "text")
            _params["tags_column"] = self.column_mapping.get("label" if not self.api else "tags_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return TokenClassificationParams(**_params)

    def _munge_params_seq2seq(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["text_column"] = "autotrain_text"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["text_column"] = self.column_mapping.get("text" if not self.api else "text_column", "text")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return Seq2SeqParams(**_params)

    def _munge_params_img_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["image_column"] = "autotrain_image"
            _params["target_column"] = "autotrain_label"
            _params["valid_split"] = "validation"
        else:
            _params["image_column"] = self.column_mapping.get("image" if not self.api else "image_column", "image")
            _params["target_column"] = self.column_mapping.get("label" if not self.api else "target_column", "label")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return ImageClassificationParams(**_params)

    def _munge_params_img_obj_det(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["log"] = "tensorboard"
        if not self.using_hub_dataset:
            _params["image_column"] = "autotrain_image"
            _params["objects_column"] = "autotrain_objects"
            _params["valid_split"] = "validation"
        else:

            _params["image_column"] = self.column_mapping.get("image" if not self.api else "image_column", "image")
            _params["objects_column"] = self.column_mapping.get(
                "objects" if not self.api else "objects_column", "objects"
            )
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split

        return ObjectDetectionParams(**_params)

    def _munge_params_tabular(self):
        _params = self._munge_common_params()
        if not self.using_hub_dataset:
            _params["id_column"] = "autotrain_id"
            _params["valid_split"] = "validation"
            if len(self.column_mapping["label"]) == 1:
                _params["target_columns"] = ["autotrain_label"]
            else:
                _params["target_columns"] = [
                    "autotrain_label_" + str(i) for i in range(len(self.column_mapping["label"]))
                ]
        else:
            _params["id_column"] = self.column_mapping.get("id" if not self.api else "id_column", "id")
            _params["train_split"] = self.train_split
            _params["valid_split"] = self.valid_split
            _params["target_columns"] = self.column_mapping.get("label" if not self.api else "target_columns", "label")

        if len(_params["categorical_imputer"].strip()) == 0 or _params["categorical_imputer"].lower() == "none":
            _params["categorical_imputer"] = None
        if len(_params["numerical_imputer"].strip()) == 0 or _params["numerical_imputer"].lower() == "none":
            _params["numerical_imputer"] = None
        if len(_params["numeric_scaler"].strip()) == 0 or _params["numeric_scaler"].lower() == "none":
            _params["numeric_scaler"] = None

        if "classification" in self.task:
            _params["task"] = "classification"
        else:
            _params["task"] = "regression"

        return TabularParams(**_params)

    def _munge_params_dreambooth(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["image_path"] = self.data_path

        if "weight_decay" in _params:
            _params["adam_weight_decay"] = _params["weight_decay"]
            _params.pop("weight_decay")

        return DreamBoothTrainingParams(**_params)


def get_task_params(task, param_type):
    if task.startswith("llm"):
        trainer = task.split(":")[1].lower()
        task = task.split(":")[0].lower()

    if task.startswith("st:"):
        trainer = task.split(":")[1].lower()
        task = task.split(":")[0].lower()

    if task.startswith("tabular"):
        task = "tabular"

    if task not in PARAMS:
        return {}

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
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "st" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "evaluation_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
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
            "early_stopping_patience",
            "early_stopping_threshold",
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
            "early_stopping_patience",
            "early_stopping_threshold",
        ]
        task_params = {k: v for k, v in task_params.items() if k not in more_hidden_params}
    if task == "image-object-detection" and param_type == "basic":
        more_hidden_params = [
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "auto_find_batch_size",
            "save_total_limit",
            "evaluation_strategy",
            "early_stopping_patience",
            "early_stopping_threshold",
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
            "early_stopping_patience",
            "early_stopping_threshold",
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
            "early_stopping_patience",
            "early_stopping_threshold",
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
