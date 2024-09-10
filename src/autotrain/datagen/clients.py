from huggingface_hub import InferenceClient
from dataclasses import dataclass
from typing import Optional

"""
from huggingface_hub import InferenceClient

client = InferenceClient(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")
"""


@dataclass
class Client:
    name: str
    model_name: Optional[str] = None
    token: Optional[str] = None

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
        if self.token:
            return self.client(self.model_name, token=self.token)
        return self.client(self.model_name)

    def chat_completion(self, messages, max_tokens=500, seed=42, response_format=None):
        _client = self._huggingface()
        message = _client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
            seed=seed,
            response_format=response_format,
        )
        return message
