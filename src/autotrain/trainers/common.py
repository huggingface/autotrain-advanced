"""
Common classes and functions for all trainers.
"""
import os

from pydantic import BaseModel

from autotrain import logger


class AutoTrainParams(BaseModel):
    """
    Base class for all AutoTrain parameters.
    """

    class Config:
        protected_namespaces = ()

    def save(self, output_dir):
        """
        Save parameters to a json file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_params.json")
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

        # Parameters not supplied by the user
        defaults = set(self.model_fields.keys())
        supplied = set(data.keys())
        not_supplied = defaults - supplied
        if not_supplied:
            logger.warning(f"Parameters not supplied by user and set to default: {', '.join(not_supplied)}")

        # Parameters that were supplied but not used
        # This is a naive implementation. It might catch some internal Pydantic params.
        unused = supplied - set(self.model_fields)
        if unused:
            logger.warning(f"Parameters supplied but not used: {', '.join(unused)}")
