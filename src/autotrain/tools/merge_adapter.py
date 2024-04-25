import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain import logger
from autotrain.trainers.common import ALLOW_REMOTE_CODE


def merge_llm_adapter(
    base_model_path, adapter_path, token, output_folder=None, pad_to_multiple_of=None, push_to_hub=False
):

    if output_folder is None and push_to_hub is False:
        raise ValueError("You must specify either --output_folder or --push_to_hub")

    logger.info("Loading adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=token,
    )
    if pad_to_multiple_of:
        base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)
    else:
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        token=token,
    )
    model = model.merge_and_unload()

    if output_folder is not None:
        logger.info("Saving target model...")
        model.save_pretrained(output_folder)
        tokenizer.save_pretrained(output_folder)
        logger.info(f"Model saved to {output_folder}")

    if push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        model.push_to_hub(adapter_path)
        tokenizer.push_to_hub(adapter_path)
        logger.info(f"Model pushed to Hugging Face Hub as {adapter_path}")
