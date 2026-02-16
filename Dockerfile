FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
WORKDIR /

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Cache directories for models
RUN mkdir -p /cache/models /root/.cache/torch

# Install Python dependencies (torch is pre-installed in base image)
COPY builder/requirements.txt /builder/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /builder/requirements.txt && \
    python3 -m pip install --no-cache-dir --force-reinstall lightning && \
    python3 -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Copy VAD model to expected location
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# Download ASR + diarization models
COPY builder /builder
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh

# Copy source code
COPY src .

CMD [ "python3", "-u", "/rp_handler.py" ]
