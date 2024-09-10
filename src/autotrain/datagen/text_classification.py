from autotrain.datagen.generator import BaseDataGenerator
from autotrain.datagen.clients import Client
import json
from autotrain import logger
import re
import ijson
import random

SYSTEM_PROMPT = """
You are an AI bot that generates data for text classification tasks.
You do not repeat the question asked by user. You do not generate code.
Only thing you generate is text data in the specified format.
The user provides a problem statement and you generate the data.
For text classification task, the user provides different classes.
If the user has not provided the classes, generate the classes as well but limit the number of classes to 10.
"""

DATA_PROMPT = """
The dataset for text classification is in JSON format.
Each line should be a JSON object with the following keys: text and target.
Make sure each text sample has atleast 25 words.
The target must always be a string.
Don't write what you are doing. Just generate the data.
Each line of the output consists of a dictionary with two keys: text and target and nothing else.
"""


def fix_invalid_json(json_string):
    # Escape backslashes that are not already escaped
    json_string = re.sub(r'(?<!\\)\\(?![\\/"bfnrt])', r"\\\\", json_string)

    # Fix unescaped double quotes within strings (if needed)
    json_string = re.sub(r'(?<!\\)"', r'\\"', json_string)

    # Handle any additional common invalid JSON fixes (optional)

    return json_string


class TextClassificationDataGenerator:
    def __init__(self, prompt: str, data_path: str, api: str):
        self.prompt = prompt
        self.data_path = data_path
        self.data_prompt = DATA_PROMPT
        self.system_prompt = SYSTEM_PROMPT

    def _check_classes(self):
        ask = SYSTEM_PROMPT + DATA_PROMPT
        formatted_message = [{"role": "system", "content": ask}]
        formatted_message.append({"role": "user", "content": self.prompt})

        client = Client("huggingface", model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
        clean_result = []

        response_format = {
            "type": "json",
            "value": {
                "properties": {
                    "data": {
                        "type": "array",
                        "maxItems": 10,
                        "minItems": 1,
                        "items": {
                            "type": "array",
                            "properties": {
                                "text": {"type": "string"},
                                "target": {"type": "string"},
                            },
                            "required": ["text", "target"],
                        },
                    }
                },
                "required": ["data"],
            },
        }

        counter = 0
        while True:
            # generate random unique seed based on counter
            random_seed = random.randint(0, 1000000) + counter
            message = client.chat_completion(
                messages=formatted_message,
                max_tokens=4096,
                seed=random_seed,
                response_format=response_format,
            )
            result = message.choices[0].message.content

            items = []
            parser = ijson.parse(result)

            current_item = None
            current_key = None

            try:
                for prefix, event, value in parser:
                    if event == "start_map":
                        current_item = {}
                    elif event == "map_key":
                        current_key = value
                    elif event == "string" and current_key:
                        current_item[current_key] = value
                    elif event == "end_map" and current_item:
                        items.append(current_item)
                        current_item = None
            except ijson.common.IncompleteJSONError:
                # Handle incomplete JSON data
                print("Incomplete JSON encountered. Returning parsed data.")

            clean_result.append(items)

        # flatten the list
        clean_result = [item for sublist in clean_result for item in sublist]
        print(clean_result)
        print(len(clean_result))
        return clean_result


if __name__ == "__main__":
    prompt = "i want to train a model to classify detect sentiment in hindi text"
    data_path = "data/text_classification.jsonl"
    api = ""

    generator = TextClassificationDataGenerator(prompt, data_path, api)
    generator._check_classes()
