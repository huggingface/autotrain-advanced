from diffusers.utils import convert_all_state_dict_to_peft, convert_state_dict_to_kohya
from safetensors.torch import load_file, save_file

from autotrain import logger


def convert_to_kohya(input_path, output_path):
    logger.info(f"Converting Lora state dict from {input_path} to Kohya state dict at {output_path}")
    lora_state_dict = load_file(input_path)
    peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
    kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
    save_file(kohya_state_dict, output_path)
    logger.info(f"Kohya state dict saved at {output_path}")
