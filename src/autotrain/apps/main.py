import gradio as gr

from autotrain.apps import utils as app_utils
from autotrain.apps.dreambooth import main as dreambooth
from autotrain.apps.llm import main as llm
from autotrain.apps.tabular import main as tabular
from autotrain.apps.text_classification import main as text_classification


llm = llm()
text_classification = text_classification()
tabular = tabular()
dreambooth = dreambooth()


def main():
    with gr.Blocks(theme=app_utils.THEME) as demo:
        gr.Markdown("## 🤗 AutoTrain Advanced")
        gr.Markdown(
            "AutoTrain Advanced is a no-code solution that allows you to train machine learning models in just a few clicks."
        )
        with gr.Accordion(label="Issues/Feature Requests", open=False):
            gr.Markdown(
                "If you have any issues or feature requests, please submit them to the [AutoTrain GitHub](https://github.com/huggingface/autotrain-advanced)."
            )
        with gr.Accordion(label="Pricing", open=False):
            gr.Markdown("You will be charged per minute of training time. The prices are as follows:")
            gr.Markdown("- A10g Large: $0.0525/minute")
            gr.Markdown("- A10g Small: $0.0175/minute")
            gr.Markdown("- A100 Large: $0.06883/minute")
            gr.Markdown("- T4 Medium: $0.015/minute")
            gr.Markdown("- T4 Small: $0.1/minute")
            gr.Markdown("- CPU: $0.0005/minute")
        with gr.Tabs():
            with gr.Tab(label="LLM"):
                llm.render()
            with gr.Tab(label="Text Classification"):
                text_classification.render()
            with gr.Tab(label="Tabular"):
                tabular.render()
            with gr.Tab(label="DreamBooth"):
                dreambooth.render()
    return demo
