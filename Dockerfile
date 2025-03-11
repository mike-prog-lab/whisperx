FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directory
RUN mkdir -p /cache/models
RUN mkdir -p /models
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN mkdir -p /root/.cache/whisperx/assets
RUN mkdir -p /root/.cache/huggingface

# Copy all builder files
COPY builder /builder

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install -r /builder/requirements.txt

# Download Faster Whisper Models and VAD model
RUN chmod +x /builder/download_models.sh
RUN --mount=type=cache,target=/cache/models \
    /builder/download_models.sh

# Pre-download Silero VAD model using torch.hub
RUN python -c "import torch; vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False, trust_repo=True); print('Successfully pre-downloaded Silero VAD model')"

# Copy source code
COPY src .

CMD [ "python", "-u", "/rp_handler.py" ]