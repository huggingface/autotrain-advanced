import gradio as gr

from autotrain.apps import utils as app_utils
from autotrain.apps.llm import main as llm
from autotrain.apps.text_classification import main as text_classification


def main():
    with gr.Blocks(theme=app_utils.THEME) as demo:
        gr.Markdown("## 🤗 AutoTrain Advanced")
        with gr.Tabs():
            with gr.Tab(label="LLM"):
                llm()
            with gr.Tab(label="Text Classification"):
                text_classification()
    return demo
