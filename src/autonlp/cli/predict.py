from argparse import ArgumentParser

from . import BaseAutoNLPCommand


def predict_command_factory(args):
    return PredictCommand(args.model_id, args.sentence)


class PredictCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        predict_parser = parser.add_parser("predict")
        predict_parser.add_argument("--model_id", type=int, default=None, required=True, help="Model ID")
        # TODO: add more types and checks when we have more tasks
        predict_parser.add_argument("--sentence", type=str, default=None, required=True, help="Input sentence")
        predict_parser.set_defaults(func=predict_command_factory)

    def __init__(self, model_id: int, sentence: str):
        self._model_id = model_id
        self._sentence = sentence

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        prediction = client.predict(model_id=self._model_id, input_text=self._sentence)
        print(prediction)
