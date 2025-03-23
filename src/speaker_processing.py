import os
import torch
import librosa
import numpy as np
import tempfile
import requests
from pyannote.audio import Inference
from scipy.spatial.distance import cosine
import logging
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
# Set up logging (you can adjust handlers as needed)
logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Only add handlers if none exist (to avoid duplicates)
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Global cache for computed speaker embeddings.
_SPEAKER_EMBEDDING_CACHE = {}

def load_known_speakers_from_samples(speaker_samples,  huggingface_access_token=None):
    # Use the passed token, environment variable, or fallback
    token_to_use = huggingface_access_token or os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # First try with minimal logging to use cached model
        model = Inference("pyannote/embedding", use_auth_token=token_to_use, device=device)
        logger.debug("Successfully loaded pyannote embedding model")
    except Exception as e:
        logger.error(f"Failed to load pyannote embedding model: {e}", exc_info=True)
        return {}
    
    """
    For each sample in speaker_samples (list of dicts with 'url' and optional 'name' and 'file_path'),
    download the file if necessary, then compute and return a dict mapping sample names to embeddings.
    If no 'name' is provided, the file name (without extension) is used.
    Uses an in-memory cache to avoid redundant computation.
    """
    global _SPEAKER_EMBEDDING_CACHE
    known_embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)
    except Exception as e:
        logger.error(f"Failed to load pyannote embedding model: {e}", exc_info=True)
        return {}

    for sample in speaker_samples:
        # Determine sample name: use provided name; if not, extract from URL.
        name = sample.get("name")
        url = sample.get("url")
        if not name:
            if url:
                name = os.path.splitext(os.path.basename(url))[0]
                logger.debug(f"No name provided; using '{name}' from URL.")
            else:
                logger.error(f"Skipping sample with missing name and URL: {sample}")
                continue

        # Check cache first.
        if name in _SPEAKER_EMBEDDING_CACHE:
            logger.debug(f"Using cached embedding for speaker '{name}'.")
            known_embeddings[name] = _SPEAKER_EMBEDDING_CACHE[name]
            continue

        # Determine source file: if sample has a local file_path, use that; otherwise, download.
        if sample.get("file_path"):
            filepath = sample["file_path"]
            logger.debug(f"Loading speaker sample '{name}' from local file: {filepath}")
        elif url:
            try:
                logger.debug(f"Downloading speaker sample '{name}' from URL: {url}")
                response = requests.get(url)
                response.raise_for_status()
                suffix = os.path.splitext(url)[1]
                if not suffix:
                    suffix = ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(response.content)
                    tmp.flush()
                    filepath = tmp.name
                    logger.debug(f"Downloaded sample '{name}' to temporary file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to download speaker sample '{name}' from {url}: {e}", exc_info=True)
                continue
        else:
            logger.error(f"Skipping sample '{name}': no file_path or URL provided.")
            try:
                waveform, _ = librosa.load(filepath, sr=16000, mono=True)
            except Exception as e:
                logger.error(f"Failed to load audio file {filepath}: {e}", exc_info=True)
                continue

        # Process the file: load audio and compute embedding.
        try:
            waveform, sr = librosa.load(filepath, sr=16000, mono=True)
            emb = model(torch.tensor(waveform).unsqueeze(0))
            emb_np = emb.detach().cpu().numpy().flatten()
            _SPEAKER_EMBEDDING_CACHE[name] = emb_np
            known_embeddings[name] = emb_np
            logger.debug(f"Computed embedding for '{name}' (norm={np.linalg.norm(emb_np):.2f}).")
        except Exception as e:
            logger.error(f"Failed to process speaker sample '{name}' from file {filepath}: {e}", exc_info=True)
        
        # If we downloaded to a temporary file, you may choose to delete it:
        if not sample.get("file_path") and url and 'filepath' in locals():
            try:
                os.remove(filepath)
                logger.debug(f"Removed temporary file for '{name}': {filepath}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {filepath}: {e}")

    return known_embeddings

def identify_speaker(segment_embedding, known_embeddings, threshold=0.75):
    """
    Compare a segment embedding against known speaker embeddings.
    Returns the best matching speaker and similarity score.
    If no match exceeds the threshold, returns "Unknown" and the best similarity.
    """
    if not isinstance(segment_embedding, np.ndarray):
        logger.error("Invalid segment_embedding: Expected a numpy array.")
        return "Unknown", -1

    if not isinstance(known_embeddings, dict):
        logger.error("Invalid known_embeddings: Expected a dictionary.")
        return "Unknown", -1

    best_match = "Unknown"
    best_similarity = -1
    for speaker, known_emb in known_embeddings.items():
        if not isinstance(known_emb, np.ndarray):
            logger.warning(f"Skipping invalid embedding for speaker '{speaker}'.")
            continue
        score = 1 - cosine(segment_embedding, known_emb)
        if score > best_similarity:
            best_similarity = score
            best_match = speaker
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return "Unknown", best_similarity

def process_diarized_output(output, audio_filepath, known_embeddings, huggingface_access_token=None):
    """
    For each diarized segment in the output, extract its audio from the given audio file,
    compute its embedding, and update the segment's speaker label using known_embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)
    
    segments = output.get("segments", [])
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        duration = end - start
        try:
            waveform, _ = librosa.load(audio_filepath, sr=16000, mono=True, offset=start, duration=duration)
        except Exception as e:
            logger.error(f"Failed to load audio segment from {audio_filepath} for {start}-{end}: {e}", exc_info=True)
            continue
        if len(waveform) == 0:
            logger.warning(f"Empty waveform for segment {start}-{end} in file {audio_filepath}.")
            continue
        seg_emb = embedding_model(torch.tensor(waveform).unsqueeze(0))
        seg_emb = seg_emb.detach().cpu().numpy().flatten()
        speaker, similarity = identify_speaker(seg_emb, known_embeddings)
        segment["speaker"] = speaker
        segment["similarity"] = similarity
        logger.debug(f"Segment {start}-{end}: identified speaker '{speaker}' with similarity {similarity:.2f}.")
    return output