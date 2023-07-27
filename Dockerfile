FROM huggingface/autotrain-base-image:latest

RUN pip install git+https://github.com/huggingface/peft.git
COPY --chown=1000:1000 . /app/

RUN pip install -e .