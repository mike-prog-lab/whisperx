import logging
import os
import sys
import tempfile
from collections import defaultdict

import librosa
import numpy as np
import requests
import torch
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature
from scipy.spatial.distance import cdist, cosine

try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

# ── Logging ───────────────────────────────────────────────────────────

logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
for _h in [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("container_log.txt", mode="a"),
]:
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(_fmt)
    logger.addHandler(_h)

# ── Device ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global models (loaded once at import time) ───────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

EMBED_MODEL = Inference("pyannote/embedding", device=DEVICE)

ECAPA_MODEL = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": str(DEVICE)},
)

# ── Global cache ─────────────────────────────────────────────────────

_SPEAKER_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


# ── Helpers ──────────────────────────────────────────────────────────


def _to_numpy_flat(emb) -> np.ndarray:
    """Convert pyannote embedding output to a flat 1-D numpy array."""
    if isinstance(emb, np.ndarray):
        return emb.flatten()
    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy().flatten()
    if isinstance(emb, SlidingWindowFeature):
        return emb.data.flatten()
    data = getattr(emb, "data", None)
    if isinstance(data, np.ndarray):
        return data.flatten()
    raise TypeError(f"Unsupported embedding type: {type(emb)}")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embed_waveform(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Return an L2-normalized embedding for a mono waveform @ 16 kHz."""
    wf = torch.tensor(wav, dtype=torch.float32)
    if wf.ndim == 1:
        wf = wf.unsqueeze(0)
    feat = EMBED_MODEL({"waveform": wf, "sample_rate": sr})
    if hasattr(feat, "data"):
        arr = feat.data.mean(axis=0)
    else:
        arr = _to_numpy_flat(feat)
    return _l2_normalize(arr.astype(np.float32))


# ── Public API (imported by rp_handler.py) ───────────────────────────


def load_known_speakers_from_samples(
    speaker_samples: list[dict],
    huggingface_access_token: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute embeddings for speaker audio samples.

    speaker_samples: list of dicts with 'url' and optional 'name'/'file_path'.
    Returns mapping: speaker_name -> L2-normalized embedding vector.
    """
    known_embeddings: dict[str, np.ndarray] = {}

    for sample in speaker_samples:
        name = sample.get("name")
        url = sample.get("url")

        if not name:
            if url:
                name = os.path.splitext(os.path.basename(url))[0]
            else:
                logger.error(f"Skipping sample with no name or URL: {sample}")
                continue

        # Check cache
        if name in _SPEAKER_EMBEDDING_CACHE:
            logger.debug(f"Using cached embedding for '{name}'.")
            known_embeddings[name] = _SPEAKER_EMBEDDING_CACHE[name]
            continue

        # Resolve file path
        filepath = sample.get("file_path")
        tmp_path = None

        if not filepath and url:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                suffix = os.path.splitext(url)[1] or ".wav"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(response.content)
                tmp.flush()
                tmp.close()
                filepath = tmp.name
                tmp_path = tmp.name
            except Exception as e:
                logger.error(f"Failed to download '{name}' from {url}: {e}")
                continue
        elif not filepath:
            logger.error(f"Skipping '{name}': no file_path or URL.")
            continue

        # Compute embedding
        try:
            waveform, sr = librosa.load(filepath, sr=16000, mono=True)
            emb = _embed_waveform(waveform, sr)
            _SPEAKER_EMBEDDING_CACHE[name] = emb
            known_embeddings[name] = emb
            logger.debug(f"Computed embedding for '{name}'.")
        except Exception as e:
            logger.error(f"Failed to process '{name}': {e}", exc_info=True)

        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return known_embeddings


def identify_speakers_on_segments(
    segments: list[dict],
    audio_path: str,
    enrolled: dict[str, np.ndarray],
    threshold: float = 0.1,
) -> list[dict]:
    """
    For each diarized segment, compute an embedding and find the closest
    enrolled speaker. Adds 'speaker_id' and 'similarity' to each segment dict.
    """
    names = list(enrolled.keys())
    mat = np.stack([enrolled[n] for n in names])  # (N, dim)

    for seg in segments:
        try:
            wav, sr = librosa.load(
                audio_path, sr=16000, mono=True,
                offset=seg["start"],
                duration=seg["end"] - seg["start"],
            )
            if wav.size == 0:
                seg["speaker_id"] = "Unknown"
                seg["similarity"] = 0.0
                continue

            emb = _embed_waveform(wav, sr)
            sims = 1 - cdist(emb[None, :], mat, metric="cosine")[0]
            best = sims.argmax()

            if sims[best] >= threshold:
                seg["speaker_id"] = names[best]
                seg["similarity"] = float(sims[best])
            else:
                seg["speaker_id"] = "Unknown"
                seg["similarity"] = float(sims.max())
        except Exception as e:
            logger.error(f"Segment [{seg.get('start')}-{seg.get('end')}]: {e}")
            seg["speaker_id"] = "Unknown"
            seg["similarity"] = 0.0

    return segments


def relabel_speakers_by_avg_similarity(segments: list[dict]) -> list[dict]:
    """
    For each original diarized speaker label, assign the most likely speaker_id
    based on the highest average similarity across all segments with that label.
    """
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker")
        sid = seg.get("speaker_id")
        sim = seg.get("similarity")
        if spk and sid and sim is not None:
            grouped[spk].append((sid, sim))

    relabel_map: dict[str, str] = {}
    for orig_spk, samples in grouped.items():
        scores: dict[str, list[float]] = defaultdict(list)
        for sid, sim in samples:
            scores[sid].append(sim)
        avg = {sid: sum(vals) / len(vals) for sid, vals in scores.items()}
        relabel_map[orig_spk] = max(avg, key=avg.get)

    for seg in segments:
        spk = seg.get("speaker")
        if spk in relabel_map:
            seg["speaker"] = relabel_map[spk]

    return segments
