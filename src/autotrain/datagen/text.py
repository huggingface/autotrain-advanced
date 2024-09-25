import hashlib
import os
import random
import time

import ijson

from autotrain import logger
from autotrain.datagen import utils
from autotrain.datagen.clients import Client
from autotrain.datagen.params import AutoTrainGenParams


TEXT_CLASSIFICATION_SYSTEM_PROMPT = """
You are an AI bot that generates data for text classification tasks.
You do not repeat the question asked by user. You do not generate code.
Only thing you generate is text data in the specified format.
The user provides a problem statement and you generate the data.
For text classification task, the user provides different classes.
If the user has not provided the classes, generate the classes as well but limit the number of classes to 10.
"""

TEXT_CLASSIFICATION_DATA_PROMPT = """
The dataset for text classification is in JSON format.
Each line should be a JSON object with the following keys: text and target.
Make sure each text sample has atleast {min_words} words.
The target must always be a string.
Don't write what you are doing. Just generate the data.
Each line of the output consists of a dictionary with two keys: text and target and nothing else.
"""

SEQ2SEQ_SYSTEM_PROMPT = """
You are an AI bot that generates data for sequence-to-sequence tasks.
You do not repeat the question asked by user. You do not generate code.
Only thing you generate is text data in the specified format.
The user provides a problem statement and you generate the data.
For sequence-to-sequence task, the user provides the input and output format.
If the user has not provided the input and output format, generate the format as well.
"""

SEQ2SEQ_DATA_PROMPT = """
The dataset for sequence-to-sequence is in JSON format.
Each line should be a JSON object with the following keys: text and target.
Make sure each text sample has atleast {min_words} words.
Both text and target sentences must always be a string.
Don't write what you are doing. Just generate the data.
Each line of the output consists of a dictionary with two keys: text and target and nothing else.
"""


class TextDataGenerator:
    def __init__(self, params: AutoTrainGenParams):
        self.params = params
        if self.params.task == "text-classification":
            self.system_prompt = TEXT_CLASSIFICATION_SYSTEM_PROMPT
            self.data_prompt = TEXT_CLASSIFICATION_DATA_PROMPT
        elif self.params.task == "seq2seq":
            self.system_prompt = SEQ2SEQ_SYSTEM_PROMPT
            self.data_prompt = SEQ2SEQ_DATA_PROMPT
        else:
            raise NotImplementedError

        self.params.save(output_dir=self.params.project_name)

        self.data_prompt = self.data_prompt.format(min_words=self.params.min_words)

    def run(self):
        ask = self.system_prompt + self.data_prompt
        formatted_message = [{"role": "system", "content": ask}]
        formatted_message.append({"role": "user", "content": self.params.prompt})
        logger.info("Prompting the model. Using prompt:")
        logger.info(formatted_message)

        client = Client(self.params.backend, model_name=self.params.gen_model, api_key=self.params.api_key)
        clean_result = []

        if self.params.task in ["text-classification", "seq2seq"]:
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
        else:
            raise NotImplementedError

        counter = 0
        while True:
            current_time = time.time()
            random_number = random.randint(0, 1000000)
            seed_input = f"{current_time}-{counter}-{random_number}"
            random_seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest(), 16) % (10**8)
            message = client.chat_completion(
                messages=formatted_message,
                max_tokens=4096,
                seed=random_seed,
                response_format=response_format,
            )

            if message is None:
                logger.warning("Failed to generate data. Retrying...")
                continue

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
                logger.warning("Incomplete JSON encountered. Returning parsed data.")

            clean_result.append(items)
            counter += 1
            num_items_collected = len([item for sublist in clean_result for item in sublist])
            logger.info(f"Collected {num_items_collected} items.")
            if num_items_collected >= self.params.min_samples:
                break

        # flatten the list
        clean_result = [item for sublist in clean_result for item in sublist]

        valid_data = None
        if self.params.valid_size != 0:
            valid_size = int(self.params.valid_size * len(clean_result))
            random.shuffle(clean_result)
            valid_data = clean_result[:valid_size]
            train_data = clean_result[valid_size:]

            logger.info(f"Train data size: {len(train_data)}")
            logger.info(f"Valid data size: {len(valid_data)}")

        hf_dataset = utils.convert_text_dataset_to_hf(self.params.task, train_data, valid_data)
        hf_dataset.save_to_disk(os.path.join(self.params.project_name, "autotrain-data"))

        if self.params.push_to_hub:
            logger.info("Pushing the data to Hugging Face Hub.")
            utils.push_data_to_hub(params=self.params, dataset=hf_dataset)

        utils.train(params=self.params)
