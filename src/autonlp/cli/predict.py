import json
from argparse import ArgumentParser

from ..utils import BOLD_TAG, GREEN_TAG, RESET_TAG, YELLOW_TAG
from . import BaseAutoNLPCommand


def predict_command_factory(args):
    return PredictCommand(model_id=args.model_id, input=args.input)


class PredictCommand(BaseAutoNLPCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        predict_parser = parser.add_parser("predict")
        predict_parser.add_argument(
            "--model_id",
            type=int,
            required=True,
            help="ID of the model to use for prediction",
        )
        predict_parser.add_argument(
            "--input", type=str, required=True, help="Text to pass to the model for prediction."
        )
        predict_parser.set_defaults(func=predict_command_factory)

    def __init__(self, model_id: int, input: SyntaxError):
        self._model_id = model_id
        self._input = input

    def run(self):
        from ..autonlp import AutoNLP

        client = AutoNLP()
        inference_api_response = client.predict(model_id=self._model_id, input_text=self._input)
        if "error" in inference_api_response:
            if "currently loading" in inference_api_response["error"] and "estimated_time" in inference_api_response:
                print(
                    f"⌛ Model is loading... Estimated loading time: {inference_api_response['estimated_time']} seconds"
                )
            elif "does not exist" in inference_api_response["error"]:
                print(f"❓ Model #{self._model_id} does not exist!")
            else:
                print(f"❌ Something went wrong when fetching predictions for model #{self._model_id}!")
                print(inference_api_response)
        else:
            print(f"✨ {BOLD_TAG}Predictions for Model #{self._model_id}:{RESET_TAG}")
            print(f"- Input text: {GREEN_TAG}'{self._input}'{RESET_TAG}")
            print(YELLOW_TAG + json.dumps(inference_api_response) + RESET_TAG)
