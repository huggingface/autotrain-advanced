import os
from typing import Optional

from pydantic import BaseModel, Field

from autotrain import logger


class BaseGenParams(BaseModel):
    """
    Base class for all AutoTrain gen parameters.
    """

    class Config:
        protected_namespaces = ()

    def save(self, output_dir):
        """
        Save parameters to a json file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "gen_params.json")
        # save formatted json
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    def __str__(self):
        """
        String representation of the parameters.
        """
        data = self.model_dump()
        data["token"] = "*****" if data.get("token") else None
        return str(data)

    def __init__(self, **data):
        """
        Initialize the parameters, check for unused/extra parameters and warn the user.
        """
        super().__init__(**data)

        if len(self.project_name) > 0:
            if not self.project_name.replace("-", "").isalnum():
                raise ValueError("project_name must be alphanumeric but can contain hyphens")

        if len(self.project_name) > 50:
            raise ValueError("project_name cannot be more than 50 characters")

        defaults = set(self.model_fields.keys())
        supplied = set(data.keys())
        not_supplied = defaults - supplied
        if not_supplied:
            logger.warning(f"Parameters not supplied by user and set to default: {', '.join(not_supplied)}")
        unused = supplied - set(self.model_fields)
        if unused:
            logger.warning(f"Parameters supplied but not used: {', '.join(unused)}")


class AutoTrainGenParams(BaseGenParams):
    gen_model: str = Field("meta-llama/Meta-Llama-3.1-8B-Instruct", title="The model to be used for generation.")
    project_name: str = Field("autotrain-datagen", title="Name of the project.")
    prompt: str = Field(None, title="Prompt to be used for text generation.")
    task: str = Field(None, title="Task type, e.g., text-classification, summarization.")
    token: Optional[str] = Field(None, title="Authentication token for accessing the model.")
    training_config: Optional[str] = Field(None, title="Path to the training configuration file.")
    valid_size: Optional[float] = Field(0.2, title="Validation set size as a fraction of the total dataset.")
    username: Optional[str] = Field(None, title="Username of the person running the training.")
    push_to_hub: Optional[bool] = Field(True, title="Whether to push the model to Hugging Face Hub.")
    backend: Optional[str] = Field("huggingface", title="Backend to be used, e.g., huggingface, local.")
    api: Optional[str] = Field(None, title="API endpoint to be used.")
    api_key: Optional[str] = Field(None, title="API key for authentication.")
    sample: Optional[str] = Field(None, title="Sample dataset for generation.")
    min_samples: Optional[int] = Field(200, title="Minimum number of samples required for training.")
    # text specific
    min_words: Optional[int] = Field(25, title="Minimum number of words in the generated text.")
