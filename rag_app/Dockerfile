# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ Europe/London 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip
# Flash attention and pytorch
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# llamacpp
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN pip install llama-cpp-python
# huggingface
RUN pip install transformers accelerate bitsandbytes sentence_transformers 
# generic
RUN pip install gradio langchain langchain-text-splitters langchain-community langchain-core python_dotenv unstructured[all]


WORKDIR /app
COPY *.py  .

RUN ln -s /usr/bin/python3.10 /usr/bin/python