FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    HF_HUB_ENABLE_HF_TRANSFER=1

ENV PATH="/app/ngc-cli:${PATH}:/app/.local/bin"
ARG PATH="/app/ngc-cli:${PATH}:/app/.local/bin"

RUN mkdir -p /tmp/model && \
    chown -R 1000:1000 /tmp/model && \
    mkdir -p /tmp/data && \
    chown -R 1000:1000 /tmp/data

RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    gcc \
    git \
    locales \
    net-tools \
    wget \
    libpq-dev \
    libsndfile1-dev \
    git \
    git-lfs \
    libgl1 \
    unzip \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean


RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    git lfs install

WORKDIR /app
RUN mkdir -p /app/.cache
ENV HF_HOME="/app/.cache"
RUN chown -R 1000:1000 /app
USER 1000
ENV HOME=/app

ENV PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    SYSTEM=spaces


COPY --chown=1000:1000 . /app/
RUN pip install -e . && \
    python -m nltk.downloader punkt && \
    autotrain setup && \
    pip install flash-attn && \
    pip install deepspeed
