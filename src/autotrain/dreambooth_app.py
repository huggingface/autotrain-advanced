import os
import pty
import random
import shutil
import string
import subprocess

import gradio as gr
from huggingface_hub import HfApi, whoami


# ❯ autotrain dreambooth --help
# usage: autotrain <command> [<args>] dreambooth [-h] --model MODEL [--revision REVISION] [--tokenizer TOKENIZER] --image-path IMAGE_PATH
#                                                [--class-image-path CLASS_IMAGE_PATH] --prompt PROMPT [--class-prompt CLASS_PROMPT]
#                                                [--num-class-images NUM_CLASS_IMAGES] [--class-labels-conditioning CLASS_LABELS_CONDITIONING]
#                                                [--prior-preservation] [--prior-loss-weight PRIOR_LOSS_WEIGHT] --output OUTPUT [--seed SEED]
#                                                --resolution RESOLUTION [--center-crop] [--train-text-encoder] [--batch-size BATCH_SIZE]
#                                                [--sample-batch-size SAMPLE_BATCH_SIZE] [--epochs EPOCHS] [--num-steps NUM_STEPS]
#                                                [--checkpointing-steps CHECKPOINTING_STEPS] [--resume-from-checkpoint RESUME_FROM_CHECKPOINT]
#                                                [--gradient-accumulation GRADIENT_ACCUMULATION] [--gradient-checkpointing] [--lr LR] [--scale-lr]
#                                                [--scheduler SCHEDULER] [--warmup-steps WARMUP_STEPS] [--num-cycles NUM_CYCLES] [--lr-power LR_POWER]
#                                                [--dataloader-num-workers DATALOADER_NUM_WORKERS] [--use-8bit-adam] [--adam-beta1 ADAM_BETA1]
#                                                [--adam-beta2 ADAM_BETA2] [--adam-weight-decay ADAM_WEIGHT_DECAY] [--adam-epsilon ADAM_EPSILON]
#                                                [--max-grad-norm MAX_GRAD_NORM] [--allow-tf32]
#                                                [--prior-generation-precision PRIOR_GENERATION_PRECISION] [--local-rank LOCAL_RANK] [--xformers]
#                                                [--pre-compute-text-embeddings] [--tokenizer-max-length TOKENIZER_MAX_LENGTH]
#                                                [--text-encoder-use-attention-mask] [--rank RANK] [--xl] [--fp16] [--bf16] [--hub-token HUB_TOKEN]
#                                                [--hub-model-id HUB_MODEL_ID] [--push-to-hub] [--validation-prompt VALIDATION_PROMPT]
#                                                [--num-validation-images NUM_VALIDATION_IMAGES] [--validation-epochs VALIDATION_EPOCHS]
#                                                [--checkpoints-total-limit CHECKPOINTS_TOTAL_LIMIT] [--validation-images VALIDATION_IMAGES]
#                                                [--logging]

REPO_ID = os.environ.get("SPACE_ID")
ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
]
WELCOME_TEXT = """
Welcome to the AutoTrain DreamBooth! This app allows you to train a DreamBooth model using AutoTrain.
The app runs on HuggingFace Spaces. Your data is not stored anywhere.
The trained model (LoRA) will be pushed to your HuggingFace Hub account.

You need to use your HuggingFace Hub write [token](https://huggingface.co/settings/tokens) to push the model to your account.

NOTE: This space requires GPU to train. Please make sure you have GPU enabled in space settings.
Please make sure to shutdown / pause the space to avoid any additional charges.
"""

STEPS = """
1. [Duplicate](https://huggingface.co/spaces/autotrain-projects/dreambooth?duplicate=true) this space
2. Upgrade the space to GPU
3. Enter your HuggingFace Hub write token
4. Upload images and adjust prompt (remember the prompt!)
5. Click on Train and wait for the training to finish
6. Go to your HuggingFace Hub account to find the trained model

NOTE: For any issues or feature requests, please open an issue [here](https://github.com/huggingface/autotrain-advanced/issues)
"""


def _update_project_name():
    random_project_name = "-".join(
        ["".join(random.choices(string.ascii_lowercase + string.digits, k=4)) for _ in range(3)]
    )
    # check if training tracker exists
    if os.path.exists(os.path.join("/tmp", "training")):
        return [
            gr.Text.update(value=random_project_name, visible=True, interactive=True),
            gr.Button.update(interactive=False),
        ]
    return [
        gr.Text.update(value=random_project_name, visible=True, interactive=True),
        gr.Button.update(interactive=True),
    ]


def run_command(cmd):
    cmd = [str(c) for c in cmd]
    print(f"Running command: {' '.join(cmd)}")
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, stdout=slave, stderr=slave)
    os.close(slave)

    while p.poll() is None:
        try:
            output = os.read(master, 1024).decode()
        except OSError:
            # Handle exception here, e.g. the pty was closed
            break
        else:
            print(output, end="")


def _run_training(
    hub_token,
    project_name,
    model,
    images,
    prompt,
    learning_rate,
    num_steps,
    batch_size,
    gradient_accumulation_steps,
    prior_preservation,
    scale_lr,
    use_8bit_adam,
    train_text_encoder,
    gradient_checkpointing,
    center_crop,
    prior_loss_weight,
    num_cycles,
    lr_power,
    adam_beta1,
    adam_beta2,
    adam_weight_decay,
    adam_epsilon,
    max_grad_norm,
    warmup_steps,
    scheduler,
    resolution,
    fp16,
):
    if REPO_ID == "autotrain-projects/dreambooth":
        return gr.Markdown.update(
            value="❌ Please [duplicate](https://huggingface.co/spaces/autotrain-projects/dreambooth?duplicate=true) this space before training."
        )

    api = HfApi(token=hub_token)

    if os.path.exists(os.path.join("/tmp", "training")):
        return gr.Markdown.update(value="❌ Another training job is already running in this space.")

    with open(os.path.join("/tmp", "training"), "w") as f:
        f.write("training")

    hub_model_id = whoami(token=hub_token)["name"] + "/" + str(project_name).strip()

    image_path = "/tmp/data"
    os.makedirs(image_path, exist_ok=True)
    output_dir = "/tmp/model"
    os.makedirs(output_dir, exist_ok=True)

    for image in images:
        shutil.copy(image.name, image_path)
    cmd = [
        "autotrain",
        "dreambooth",
        "--model",
        model,
        "--output",
        output_dir,
        "--image-path",
        image_path,
        "--prompt",
        prompt,
        "--resolution",
        "1024",
        "--batch-size",
        batch_size,
        "--num-steps",
        num_steps,
        "--gradient-accumulation",
        gradient_accumulation_steps,
        "--lr",
        learning_rate,
        "--scheduler",
        scheduler,
        "--warmup-steps",
        warmup_steps,
        "--num-cycles",
        num_cycles,
        "--lr-power",
        lr_power,
        "--adam-beta1",
        adam_beta1,
        "--adam-beta2",
        adam_beta2,
        "--adam-weight-decay",
        adam_weight_decay,
        "--adam-epsilon",
        adam_epsilon,
        "--max-grad-norm",
        max_grad_norm,
        "--prior-loss-weight",
        prior_loss_weight,
        "--push-to-hub",
        "--hub-token",
        hub_token,
        "--hub-model-id",
        hub_model_id,
    ]

    if prior_preservation:
        cmd.append("--prior-preservation")
    if scale_lr:
        cmd.append("--scale-lr")
    if use_8bit_adam:
        cmd.append("--use-8bit-adam")
    if train_text_encoder:
        cmd.append("--train-text-encoder")
    if gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if center_crop:
        cmd.append("--center-crop")
    if fp16:
        cmd.append("--fp16")

    try:
        run_command(cmd)
        # delete the training tracker file in /tmp/
        os.remove(os.path.join("/tmp", "training"))
        # switch off space
        if REPO_ID is not None:
            api.pause_space(repo_id=REPO_ID)
        return gr.Markdown.update(value=f"✅ Training finished! Model pushed to {hub_model_id}")
    except Exception as e:
        print(e)
        print("Error running command")
        # delete the training tracker file in /tmp/
        os.remove(os.path.join("/tmp", "training"))
        return gr.Markdown.update(value="❌ Error running command. Please try again.")


def main():
    with gr.Blocks(theme="freddyaboulton/dracula_revamped") as demo:
        gr.Markdown("## 🤗 AutoTrain DreamBooth")
        gr.Markdown(WELCOME_TEXT)
        with gr.Accordion("Steps", open=False):
            gr.Markdown(STEPS)
        hub_token = gr.Textbox(
            label="Hub Token",
            value="",
            lines=1,
            max_lines=1,
            interactive=True,
            type="password",
        )

        with gr.Row():
            with gr.Column():
                project_name = gr.Textbox(
                    label="Project name",
                    value="",
                    lines=1,
                    max_lines=1,
                    interactive=True,
                )
                model = gr.Dropdown(
                    label="Model",
                    choices=MODELS,
                    value=MODELS[0],
                    visible=True,
                    interactive=True,
                    elem_id="model",
                    allow_custom_values=True,
                )
                images = gr.File(
                    label="Images",
                    file_types=ALLOWED_FILE_TYPES,
                    file_count="multiple",
                    visible=True,
                    interactive=True,
                )

            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="photo of sks dog",
                    lines=1,
                )
                with gr.Row():
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=1e-4,
                        visible=True,
                        interactive=True,
                        elem_id="learning_rate",
                    )
                    num_steps = gr.Number(
                        label="Number of Steps",
                        value=500,
                        visible=True,
                        interactive=True,
                        elem_id="num_steps",
                        precision=0,
                    )
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        visible=True,
                        interactive=True,
                        elem_id="batch_size",
                        precision=0,
                    )
                with gr.Row():
                    gradient_accumulation_steps = gr.Number(
                        label="Gradient Accumulation Steps",
                        value=4,
                        visible=True,
                        interactive=True,
                        elem_id="gradient_accumulation_steps",
                        precision=0,
                    )
                    resolution = gr.Number(
                        label="Resolution",
                        value=1024,
                        visible=True,
                        interactive=True,
                        elem_id="resolution",
                        precision=0,
                    )
                    scheduler = gr.Dropdown(
                        label="Scheduler",
                        choices=["cosine", "linear", "constant"],
                        value="constant",
                        visible=True,
                        interactive=True,
                        elem_id="scheduler",
                    )
            with gr.Column():
                with gr.Group():
                    fp16 = gr.Checkbox(
                        label="FP16",
                        value=True,
                        visible=True,
                        interactive=True,
                        elem_id="fp16",
                    )
                    prior_preservation = gr.Checkbox(
                        label="Prior Preservation",
                        value=False,
                        visible=True,
                        interactive=True,
                        elem_id="prior_preservation",
                    )
                    scale_lr = gr.Checkbox(
                        label="Scale LR",
                        value=False,
                        visible=True,
                        interactive=True,
                        elem_id="scale_lr",
                    )
                    use_8bit_adam = gr.Checkbox(
                        label="Use 8bit Adam",
                        value=True,
                        visible=True,
                        interactive=True,
                        elem_id="use_8bit_adam",
                    )
                    train_text_encoder = gr.Checkbox(
                        label="Train Text Encoder",
                        value=False,
                        visible=True,
                        interactive=True,
                        elem_id="train_text_encoder",
                    )
                    gradient_checkpointing = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=False,
                        visible=True,
                        interactive=True,
                        elem_id="gradient_checkpointing",
                    )
                    center_crop = gr.Checkbox(
                        label="Center Crop",
                        value=False,
                        visible=True,
                        interactive=True,
                        elem_id="center_crop",
                    )
        with gr.Accordion("Advanced Parameters", open=False):
            with gr.Row():
                prior_loss_weight = gr.Number(
                    label="Prior Loss Weight",
                    value=1.0,
                    visible=True,
                    interactive=True,
                    elem_id="prior_loss_weight",
                )
                num_cycles = gr.Number(
                    label="Num Cycles",
                    value=1,
                    visible=True,
                    interactive=True,
                    elem_id="num_cycles",
                    precision=0,
                )
                lr_power = gr.Number(
                    label="LR Power",
                    value=1,
                    visible=True,
                    interactive=True,
                    elem_id="lr_power",
                )

                adam_beta1 = gr.Number(
                    label="Adam Beta1",
                    value=0.9,
                    visible=True,
                    interactive=True,
                    elem_id="adam_beta1",
                )
                adam_beta2 = gr.Number(
                    label="Adam Beta2",
                    value=0.999,
                    visible=True,
                    interactive=True,
                    elem_id="adam_beta2",
                )
                adam_weight_decay = gr.Number(
                    label="Adam Weight Decay",
                    value=1e-2,
                    visible=True,
                    interactive=True,
                    elem_id="adam_weight_decay",
                )
                adam_epsilon = gr.Number(
                    label="Adam Epsilon",
                    value=1e-8,
                    visible=True,
                    interactive=True,
                    elem_id="adam_epsilon",
                )
                max_grad_norm = gr.Number(
                    label="Max Grad Norm",
                    value=1,
                    visible=True,
                    interactive=True,
                    elem_id="max_grad_norm",
                )
                warmup_steps = gr.Number(
                    label="Warmup Steps",
                    value=0,
                    visible=True,
                    interactive=True,
                    elem_id="warmup_steps",
                    precision=0,
                )

        train_button = gr.Button(value="Train", elem_id="train")
        output_md = gr.Markdown("## Output")
        inputs = [
            hub_token,
            project_name,
            model,
            images,
            prompt,
            learning_rate,
            num_steps,
            batch_size,
            gradient_accumulation_steps,
            prior_preservation,
            scale_lr,
            use_8bit_adam,
            train_text_encoder,
            gradient_checkpointing,
            center_crop,
            prior_loss_weight,
            num_cycles,
            lr_power,
            adam_beta1,
            adam_beta2,
            adam_weight_decay,
            adam_epsilon,
            max_grad_norm,
            warmup_steps,
            scheduler,
            resolution,
            fp16,
        ]

        train_button.click(_run_training, inputs=inputs, outputs=output_md)
        demo.load(
            _update_project_name,
            outputs=[project_name, train_button],
        )
    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
