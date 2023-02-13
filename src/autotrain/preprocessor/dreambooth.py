from dataclasses import dataclass


@dataclass
class DreamboothPreprocessor:
    def __init__(self, config):
        self.config = config

    def preprocess(self, data):
        pass
