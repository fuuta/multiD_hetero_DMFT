FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install -y \
    build-essential \
    wget curl git zip gcc make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    python3-pip libfreetype6-dev libffi-dev \
    liblzma-dev openssh-server msttcorefonts \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PROJECT_DIR
ARG UV_VERSION

####################
### python setup ###
####################

ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

##############
### others ###
##############

WORKDIR /app

# cache cuda, cudnn, and other libraries to speed up to prepare environment after the first build
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

