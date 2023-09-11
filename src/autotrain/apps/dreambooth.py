from functools import partial

import gradio as gr
import pandas as pd

from autotrain.apps import common
from autotrain.apps import utils as app_utils
from autotrain.dataset import AutoTrainDreamboothDataset
from autotrain.project import AutoTrainProject


ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]

MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
]


def start_training(
    jobs_df,
    model_choice,
    training_data,
    project_name,
    autotrain_username,
    user_token,
    prompt,
):
    if len(jobs_df) == 0:
        raise gr.Error("Please add at least one job.")
    dset = AutoTrainDreamboothDataset(
        concept_images=training_data,
        concept_name=prompt,
        token=user_token,
        project_name=project_name,
        username=autotrain_username,
    )
    dset.prepare()
    project = AutoTrainProject(dataset=dset, job_params=jobs_df)
    ids = project.create()
    return gr.Markdown.update(
        value=f"Training started for {len(ids)} jobs. You can view the status of your jobs at ids: {', '.join(ids)}",
        visible=True,
    )


def main():
    with gr.Blocks(theme=app_utils.THEME) as demo:
        gr.Markdown("### 🚀 DreamBooth")
        user_token, valid_can_pay, who_is_training = common.user_validation()
        autotrain_username, project_name, model_choice, autotrain_backend = common.base_components(who_is_training)
        with gr.Row():
            training_data = gr.File(
                label="Images",
                file_types=ALLOWED_FILE_TYPES,
                file_count="multiple",
                visible=True,
                interactive=True,
            )
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        hyp_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="photo of sks dog",
                            lines=1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_prompt",
                        )
                    with gr.Row():
                        hyp_resolution = gr.Number(
                            label="Resolution",
                            value=1024,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_resolution",
                            precision=0,
                        )
                        hyp_scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["cosine", "linear", "constant"],
                            value="linear",
                            visible=True,
                            interactive=True,
                            elem_id="hyp_scheduler",
                        )
                    with gr.Row():
                        with gr.Group():
                            hyp_gradient_checkpointing = gr.Checkbox(
                                label="Gradient Checkpointing",
                                value=False,
                                visible=True,
                                interactive=True,
                                elem_id="hyp_gradient_checkpointing",
                            )
                            hyp_xformers = gr.Checkbox(
                                label="Enable XFormers",
                                value=True,
                                visible=True,
                                interactive=True,
                                elem_id="hyp_xformers",
                            )

                            with gr.Row():
                                hyp_prior_preservation = gr.Checkbox(
                                    label="Prior Preservation",
                                    value=False,
                                    visible=True,
                                    interactive=True,
                                    elem_id="hyp_prior_preservation",
                                )
                                hyp_scale_lr = gr.Checkbox(
                                    label="Scale LR",
                                    value=False,
                                    visible=True,
                                    interactive=True,
                                    elem_id="hyp_scale_lr",
                                )
                            with gr.Row():
                                hyp_use_8bit_adam = gr.Checkbox(
                                    label="Use 8bit Adam",
                                    value=True,
                                    visible=True,
                                    interactive=True,
                                    elem_id="hyp_use_8bit_adam",
                                )
                                hyp_train_text_encoder = gr.Checkbox(
                                    label="Train Text Encoder",
                                    value=False,
                                    visible=True,
                                    interactive=True,
                                    elem_id="hyp_train_text_encoder",
                                )
                            with gr.Row():
                                hyp_fp16 = gr.Checkbox(
                                    label="FP16",
                                    value=True,
                                    visible=True,
                                    interactive=True,
                                    elem_id="hyp_fp16",
                                )
                                hyp_center_crop = gr.Checkbox(
                                    label="Center Crop",
                                    value=False,
                                    visible=True,
                                    interactive=True,
                                    elem_id="hyp_center_crop",
                                )

            with gr.Column():
                with gr.Group():
                    param_choice = gr.Dropdown(
                        label="Parameter Choice",
                        choices=["Manual"],
                        value="Manual",
                        visible=True,
                        interactive=True,
                        elem_id="param_choice",
                    )
                    with gr.Row():
                        hyp_lr = gr.Number(
                            label="Learning Rate",
                            value=1e-4,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_lr",
                        )
                        hyp_num_steps = gr.Number(
                            label="Number of Steps",
                            value=500,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_num_steps",
                            precision=0,
                        )
                    with gr.Row():
                        hyp_warmup_steps = gr.Number(
                            label="Warmup Steps",
                            value=0,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_warmup_steps",
                            precision=0,
                        )
                        hyp_batch_size = gr.Number(
                            label="Batch Size",
                            value=1,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_batch_size",
                            precision=0,
                        )
                    with gr.Row():
                        hyp_prior_loss_weight = gr.Number(
                            label="Prior Loss Weight",
                            value=1.0,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_prior_loss_weight",
                        )
                        hyp_weight_decay = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_weight_decay",
                        )
                    with gr.Row():
                        hyp_gradient_accumulation = gr.Number(
                            label="Gradient Accumulation Steps",
                            value=4,
                            visible=True,
                            interactive=True,
                            elem_id="hyp_gradient_accumulation",
                            precision=0,
                        )

        with gr.Row():
            add_job_button = gr.Button(value="Add Job", elem_id="add_job_button")
            clear_jobs_button = gr.Button(value="Clear Jobs", elem_id="clear_jobs_button")
            start_training_button = gr.Button(value="Start Training", elem_id="start_training_button")

        output_md = gr.Markdown(
            value="WARNING: Clicking `Start Training` will incur costs!", visible=True, interactive=False
        )
        jobs_df = gr.DataFrame(visible=False, interactive=False, value=pd.DataFrame())

        hyperparameters = [
            hyp_prompt,
            hyp_resolution,
            hyp_scheduler,
            hyp_gradient_checkpointing,
            hyp_xformers,
            hyp_prior_preservation,
            hyp_scale_lr,
            hyp_use_8bit_adam,
            hyp_train_text_encoder,
            hyp_fp16,
            hyp_center_crop,
            hyp_lr,
            hyp_num_steps,
            hyp_warmup_steps,
            hyp_batch_size,
            hyp_prior_loss_weight,
            hyp_weight_decay,
            hyp_gradient_accumulation,
        ]

        model_choice.change(
            app_utils.handle_model_choice_change,
            inputs=model_choice,
            outputs=[jobs_df, param_choice, autotrain_backend],
        )

        def _add_job(components):
            if not components[training_data]:
                raise gr.Error("Please upload training images first.")
            if len(str(components[hyp_prompt].strip())) == 0:
                raise gr.Error("Prompt cannot be empty.")
            if components[param_choice] == "AutoTrain" and components[autotrain_backend] != "AutoTrain":
                raise gr.Error("AutoTrain param choice is only available with AutoTrain backend.")

            _training_params = {}
            for _hyperparameter in hyperparameters:
                _training_params[_hyperparameter.elem_id] = components[_hyperparameter]

            _training_params_df = app_utils.fetch_training_params_df(
                components[param_choice],
                components[jobs_df],
                _training_params,
                components[model_choice],
                components[autotrain_backend],
            )
            return gr.DataFrame.update(value=_training_params_df, visible=True, interactive=False)

        add_job_button.click(
            _add_job,
            inputs=set(
                [
                    training_data,
                    param_choice,
                    autotrain_backend,
                    jobs_df,
                    model_choice,
                    autotrain_backend,
                ]
                + hyperparameters
            ),
            outputs=jobs_df,
        )

        clear_jobs_button.click(
            app_utils.clear_jobs,
            inputs=jobs_df,
            outputs=jobs_df,
        )

        start_training_button.click(
            start_training,
            inputs=[
                jobs_df,
                model_choice,
                training_data,
                project_name,
                autotrain_username,
                user_token,
                hyp_prompt,
            ],
            outputs=output_md,
        )

        demo.load(
            app_utils._update_project_name,
            outputs=project_name,
        ).then(
            partial(app_utils._update_hub_model_choices, task="dreambooth"),
            outputs=model_choice,
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch()
