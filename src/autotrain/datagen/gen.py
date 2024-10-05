from dataclasses import dataclass

from autotrain import logger
from autotrain.datagen.params import AutoTrainGenParams


@dataclass
class AutoTrainGen:
    params: AutoTrainGenParams

    def __post_init__(self):
        logger.info(self.params)
        if self.params.task in ("text-classification", "seq2seq"):
            from autotrain.datagen.text import TextDataGenerator

            self.gen = TextDataGenerator(self.params)
        else:
            raise NotImplementedError

    def run(self):
        self.gen.run()
