FROM runpod/base:0.6.2-cuda12.4.1

SHELL ["/bin/bash", "-c"]
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg wget git libcudnn8 libcudnn8-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# In your Dockerfile
# Create cache directory
RUN mkdir -p /cache/models

# Create torch cache directory for VAD model
RUN mkdir -p /root/.cache/torch

# Copy only requirements file first to leverage Docker cache
COPY builder/requirements.txt /builder/requirements.txt

# Install Python dependencies (Worker Template)
RUN python3 -m pip install --upgrade pip hf_transfer && \
    python3 -m pip install -r /builder/requirements.txt

# Copy the local VAD model to the expected location
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# Copy the rest of the builder files
COPY builder /builder

# Download Faster Whisper Models
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh

# Copy source code
COPY src .

CMD [ "python3", "-u", "/rp_handler.py" ]