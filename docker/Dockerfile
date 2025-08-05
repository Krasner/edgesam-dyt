FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    ca-certificates \
    pybind11-dev \
    python3-pybind11 \
    git

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app
RUN uv sync --locked
ENV PATH="/app/.venv/bin:$PATH"

RUN git clone https://github.com/open-mmlab/mmcv.git && \
    git clone https://github.com/open-mmlab/mmdetection.git && \
    git clone https://github.com/open-mmlab/mmengine.git


