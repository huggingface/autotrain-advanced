from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


@dataclass
class TextGenerationInference:
    model_path: str = "gpt2"
    use_int4: Optional[bool] = False
    use_int8: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    num_beams: Optional[int] = 1
    max_new_tokens: Optional[int] = 1024
    do_sample: Optional[bool] = True

    def __post_init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=self.use_int4,
            load_in_8bit=self.use_int8,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=self.num_return_sequences,
            num_beams=self.num_beams,
            max_length=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
        )

    def chat(self, prompt):
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
