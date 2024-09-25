import json
import time
from dataclasses import dataclass
from typing import Optional

import outlines
import torch
import transformers
from huggingface_hub import InferenceClient

from autotrain import logger


@dataclass
class _TransformersClient:
    model_name: str

    def __post_init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def chat_completion(self, messages, max_tokens, stream, seed, response_format):
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            seed=seed,
            response_format=response_format,
            stream=stream,
        )
        return outputs[0]["generated_text"][-1]["content"]


@dataclass
class TransformersClient:
    model_name: str

    def __post_init__(self):
        self.pipeline = outlines.models.transformers(
            self.model_name,
            # device_map="auto",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def chat_completion(self, messages, max_tokens, stream, seed, response_format):
        # dump response_format dict to json
        response_format = json.dumps(response_format)
        generator = outlines.generate.json(self.pipeline, response_format)
        outputs = generator(
            messages,
            max_tokens=max_tokens,
            seed=seed,
        )
        print(outputs)
        return outputs[0]["generated_text"][-1]["content"]


@dataclass
class Client:
    name: str
    model_name: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.name == "hf-inference-api":
            if self.model_name is None:
                raise ValueError("Model name is required for Huggingface")
            self.client = InferenceClient
        elif self.name == "transformers":
            if self.model_name is None:
                raise ValueError("Model name is required for Transformers")
            self.client = TransformersClient
        else:
            raise ValueError("Client not supported")

    def __str__(self):
        return f"Client: {self.name}"

    def __repr__(self):
        return f"Client: {self.name}"

    def _huggingface(self):
        if self.api_key:
            return self.client(self.model_name, token=self.api_key)
        return self.client(self.model_name)

    def _transformers(self):
        return self.client(self.model_name)

    def chat_completion(self, messages, max_tokens=500, seed=42, response_format=None, retries=3, delay=5):
        if self.name == "hf-inference-api":
            _client = self._huggingface()
        elif self.name == "transformers":
            _client = self._transformers()
        else:
            raise ValueError("Client not supported")
        for attempt in range(retries):
            try:
                message = _client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=False,
                    seed=seed,
                    response_format=response_format,
                )
                return message
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return None
