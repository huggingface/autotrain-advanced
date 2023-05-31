FROM python:3.8.9

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN pip install pip==22.3.1

WORKDIR /app
RUN mkdir -p /app/.cache
ENV HF_HOME="/app/.cache"
RUN chown -R 1000:1000 /app
USER 1000

ENV PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && sh Miniconda3-latest-Linux-x86_64.sh -b -p /app/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH /app/miniconda/bin:$PATH

RUN conda create -p /app/env -y python=3.8


SHELL ["conda", "run","--no-capture-output", "-p","/app/env", "/bin/bash", "-c"]

COPY --chown=1000:1000 . /app/

RUN pip install -e .