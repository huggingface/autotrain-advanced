import os

from pydantic import BaseModel

from autotrain import logger


class AutoTrainParams(BaseModel):
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_params.json")
        # save formatted json
        with open(path, "w") as f:
            f.write(self.json(indent=4))

    def __str__(self):
        data = self.dict()
        data["token"] = "*****" if data.get("token") else None
        return str(data)

    def __init__(self, **data):
        super().__init__(**data)

        # Parameters not supplied by the user
        defaults = {f.name for f in self.__fields__.values() if f.default == self.__dict__[f.name]}
        supplied = set(data.keys())
        not_supplied = defaults - supplied
        if not_supplied:
            logger.warning(f"Parameters not supplied by user and set to default: {', '.join(not_supplied)}")

        # Parameters that were supplied but not used
        # This is a naive implementation. It might catch some internal Pydantic params.
        unused = supplied - set(self.__fields__)
        if unused:
            logger.warning(f"Parameters supplied but not used: {', '.join(unused)}")
