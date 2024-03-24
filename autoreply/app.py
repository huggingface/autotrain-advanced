from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def load_model(self, model_path):
        if model_path not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype='auto'
            ).eval().to('cuda')
            self.models[model_path] = model
            self.tokenizers[model_path] = tokenizer
        return self.models[model_path], self.tokenizers[model_path]

model_manager = ModelManager()



app = FastAPI()

class PredictionRequest(BaseModel):
    model_path: str
    messages: list

@app.post("/predict/")
async def generate_text(request: PredictionRequest):
    # Load the model and tokenizer or get them if already loaded
    model, tokenizer = model_manager.load_model(request.model_path)
    
    try:
        input_ids = tokenizer.apply_chat_template(
            conversation=request.messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        )
        
        # Generate response
        output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=80)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
