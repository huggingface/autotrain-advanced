from dataclasses import dataclass
from typing import Optional

from huggingface_hub import InferenceClient
from autotrain import logger
import time


@dataclass
class Client:
    name: str
    model_name: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.name == "huggingface":
            if self.model_name is None:
                raise ValueError("Model name is required for Huggingface")
            self.client = InferenceClient
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

    def chat_completion(self, messages, max_tokens=500, seed=42, response_format=None, retries=3, delay=5):
        for attempt in range(retries):
            try:
                _client = self._huggingface()
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
