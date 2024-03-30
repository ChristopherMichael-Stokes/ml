FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
# [2021-09-09] TD: zsh, stow, subversion, fasd are for setting up my personal environment.
# [2021-12-07] TD: openmpi-bin for MPI (multi-node training)
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

# Python packages
ENV PIP_NO_CACHE_DIR=1
# RUN pip install ninja
# RUN MAX_JOBS=8 pip install flash-attn==2.5.6
RUN pip install git+https://github.com/HazyResearch/flash-attention@v2.5.6#subdirectory=csrc/fused_dense_lib
RUN pip install jupyter pandas matplotlib scikit-learn plotly catboost gradio
RUN pip install transformers>=4.39.2 accelerate bitsandbytes datasets
RUN pip install tensorflow keras_nlp  
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install tensorflow_datasets


RUN chsh -s /bin/zsh
RUN yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN sed -i 's/plugins=(git)/plugins=(git vi-mode)/' /root/.zshrc # ohmyzsh plugins
RUN printf "export EDITOR='vim'\nbindkey -v" >> /root/.zshrc # set vim keybinds for zsh, use vim as default editor

# RUN ln -sf /bin/zsh /bin/bash