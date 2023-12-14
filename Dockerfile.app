FROM huggingface/autotrain-advanced:latest
CMD uvicorn autotrain.app:app --host 0.0.0.0 --port 7860 --reload --workers 4
