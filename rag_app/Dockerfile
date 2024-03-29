FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ Europe/London 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    zsh stow subversion fasd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1
# RUN pip install uv  # <-- tried uv (way faster) but found annoying runtime issues
# RUN uv venv --system-site-packages
# RUN source .venv/bin/activate
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 uv pip install 'llama-cpp-python[server]'
# RUN uv pip install jupyter gradio langchain langchain-community langchain-core transformers>=4.39.2 accelerate bitsandbytes sentence_transformers
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install 'llama-cpp-python'
RUN pip install jupyter gradio langchain langchain-community langchain-core transformers>=4.39.2 accelerate bitsandbytes sentence_transformers python_dotenv


COPY llm_lib.py  .
COPY rag_client.py . 