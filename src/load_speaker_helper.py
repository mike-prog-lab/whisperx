import os
import torch
import librosa
import numpy as np
import tempfile
import requests
from pyannote.audio import Inference

# Global cache for speaker embeddings
_SPEAKER_EMBEDDING_CACHE = {}

def load_known_speakers_from_urls(speaker_samples):
    """
    Download speaker sample files from provided URLs, compute embeddings, and return a dict.
    Uses an in-memory cache to avoid redundant computation.
    
    :param speaker_samples: List of dicts, each with 'name' and 'url' keys.
    :return: Dict mapping speaker names to embedding vectors.
    """
    global _SPEAKER_EMBEDDING_CACHE
    known_embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the pyannote embedding model
    model = Inference("pyannote/embedding", device=device)
    
    for sample in speaker_samples:
        name = sample.get("name")
        url = sample.get("url")
        if not name or not url:
            print(f"Skipping sample with missing name or url: {sample}")
            continue

        # Check cache first
        if name in _SPEAKER_EMBEDDING_CACHE:
            known_embeddings[name] = _SPEAKER_EMBEDDING_CACHE[name]
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()
            # Determine file suffix from URL; default to .wav if not found.
            suffix = os.path.splitext(url)[1]
            if not suffix:
                suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(response.content)
                tmp.flush()
                # Load audio using librosa (ffmpeg installed in container enables support for MP3, M4A, etc.)
                waveform, sr = librosa.load(tmp.name, sr=16000, mono=True)
                emb = model(torch.tensor(waveform).unsqueeze(0))
                emb_np = emb.detach().cpu().numpy().flatten()
                # Cache the computed embedding
                _SPEAKER_EMBEDDING_CACHE[name] = emb_np
                known_embeddings[name] = emb_np
        except Exception as e:
            print(f"Failed to load speaker sample {name} from {url}: {e}")
    
    return known_embeddings