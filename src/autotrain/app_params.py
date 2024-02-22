import json
from dataclasses import dataclass

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


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

    def munge(self):
        if self.task == "text-classification":
            return self._munge_params_text_clf()
        elif self.task == "seq2seq":
            return self._munge_params_seq2seq()
        elif self.task == "image-classification":
            return self._munge_params_img_clf()
        elif self.task.startswith("tabular"):
            return self._munge_params_tabular()
        elif self.task == "dreambooth":
            return self._munge_params_dreambooth()
        elif self.task.startswith("llm"):
            return self._munge_params_llm()
        elif self.task == "token-classification":
            return self._munge_params_token_clf()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _munge_common_params(self):
        _params = json.loads(self.job_params_json)
        _params["token"] = self.token
        _params["project_name"] = f"{self.project_name}"
        _params["push_to_hub"] = True
        _params["repo_id"] = f"{self.username}/{self.project_name}"
        _params["data_path"] = self.data_path
        _params["username"] = self.username
        return _params

    def _munge_params_llm(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["text_column"] = "autotrain_text"
        _params["prompt_text_column"] = "autotrain_prompt"
        _params["rejected_text_column"] = "autotrain_rejected_text"
        _params["log"] = "tensorboard"

        trainer = self.task.split(":")[1]
        if trainer != "generic":
            _params["trainer"] = trainer.lower()

        return LLMTrainingParams(**_params)

    def _munge_params_text_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["text_column"] = "autotrain_text"
        _params["target_column"] = "autotrain_label"
        _params["valid_split"] = "validation"

        return TextClassificationParams(**_params)

    def _munge_params_token_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["text_column"] = "autotrain_text"
        _params["target_column"] = "autotrain_label"
        _params["valid_split"] = "validation"

        return TokenClassificationParams(**_params)

    def _munge_params_seq2seq(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["text_column"] = "autotrain_text"
        _params["target_column"] = "autotrain_label"
        _params["valid_split"] = "validation"

        return Seq2SeqParams(**_params)

    def _munge_params_img_clf(self):
        _params = self._munge_common_params()
        _params["model"] = self.base_model
        _params["image_column"] = "autotrain_image"
        _params["target_column"] = "autotrain_label"
        _params["valid_split"] = "validation"

        return ImageClassificationParams(**_params)

    def _munge_params_tabular(self):
        _params = self._munge_common_params()
        _params["id_column"] = "autotrain_id"
        _params["valid_split"] = "validation"
        if len(self.column_mapping["label"]) == 1:
            _params["target_columns"] = ["autotrain_label"]
        else:
            _params["target_columns"] = ["autotrain_label_" + str(i) for i in range(len(self.column_mapping["label"]))]

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
