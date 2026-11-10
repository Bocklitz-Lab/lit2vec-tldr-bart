# ==== BASE IMAGE (CUDA for GPU training) ====
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ==== SYSTEM SETUP ====
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential python3 python3-pip python3-venv jq \
    && rm -rf /var/lib/apt/lists/*

# Use python3 by default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip

# ==== PYTHON DEPENDENCIES ====
# One pinned install. The extra index provides the CUDA 12.1 build of torch.
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ==== NLTK DATA ====
RUN python -m nltk.downloader punkt

# ==== COPY PROJECT FILES ====
# Now copy the rest of the repo
COPY . .

# ==== ENV VARS ====
# prevent huggingface token caching errors in containers
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers
ENV WANDB_CACHE_DIR=/workspace/.cache/wandb

# ==== DEFAULT COMMAND ====
CMD ["bash"]
