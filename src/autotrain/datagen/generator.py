from pydantic import BaseModel


class BaseDataGenerator(BaseModel):
    def load_data(self):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError

    def pre_generate_data(self):
        raise NotImplementedError

    def generate_data(self):
        raise NotImplementedError
