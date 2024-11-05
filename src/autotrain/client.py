from dataclasses import dataclass
from typing import Optional
import os
import requests

"""
{
  "project_name": "string",
  "task": "llm:sft",
  "base_model": "string",
  "hardware": "spaces-a10g-large",
  "params": {
    "block_size": -1,
    "model_max_length": 2048,
    "padding": "right",
    "use_flash_attention_2": false,
    "disable_gradient_checkpointing": false,
    "logging_steps": -1,
    "eval_strategy": "epoch",
    "save_total_limit": 1,
    "auto_find_batch_size": false,
    "mixed_precision": "string",
    "lr": 0.00003,
    "epochs": 1,
    "batch_size": 2,
    "warmup_ratio": 0.1,
    "gradient_accumulation": 4,
    "optimizer": "adamw_torch",
    "scheduler": "linear",
    "weight_decay": 0,
    "max_grad_norm": 1,
    "seed": 42,
    "chat_template": "string",
    "quantization": "int4",
    "target_modules": "all-linear",
    "merge_adapter": false,
    "peft": false,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "unsloth": false,
    "distributed_backend": "string"
  },
  "username": "string",
  "column_mapping": {
    "text_column": "string"
  },
  "hub_dataset": "string",
  "train_split": "string",
  "valid_split": "string"
}
"""

@dataclass
class Client:
    host: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None

    def __post_init__(self):
        if self.host is None:
            self.host = "https://autotrain-projects-autotrain-advanced.hf.space/"
        
        if self.token is None:
            self.token = os.environ.get("HF_TOKEN")
        
        if self.username is None:
            self.username = os.environ.get("HF_USERNAME")

        if self.token is None or self.username is None:
            raise ValueError("Please provide a valid username and token")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
    def __str__(self):
        return f"Client(host={self.host}, token=****, username={self.username})"
    
    def __repr__(self):
        return self.__str__()   
    
    def create(self, project_name: str, task: str, base_model: str, hardware: str, params: dict, column_mapping: dict, hub_dataset: str, train_split: str, valid_split: str):
        url = f"{self.host}/api/create_project"
        data = {
            "project_name": project_name,
            "task": task,
            "base_model": base_model,
            "hardware": hardware,
            "params": params,
            "username": self.username,
            "column_mapping": column_mapping,
            "hub_dataset": hub_dataset,
            "train_split": train_split,
            "valid_split": valid_split
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
