# speaker_processing.py
import os, tempfile, logging, requests
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import librosa
import torch
from scipy.spatial.distance import cosine
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature

logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------
# 0) small helpers
# ---------------------------------------------------------------------
_SPK_CACHE: Dict[str, np.ndarray] = {}
_EMBED_MODEL: Inference | None = None


def _embed(wav: np.ndarray, sr: int = 16000,
           token: str | None = None) -> np.ndarray:
    """Return a 512-d ℓ2-normalised PyAnnote embedding."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _EMBED_MODEL = Inference("pyannote/embedding",
                                 use_auth_token=token,
                                 device=device)

    feat = _EMBED_MODEL(
        {"waveform": torch.tensor(wav).unsqueeze(0),
         "sample_rate": sr}
    )

    if isinstance(feat, SlidingWindowFeature):
        feat = feat.data.mean(axis=0)          # (dim,)
    elif isinstance(feat, torch.Tensor):
        feat = feat.squeeze(0).cpu().numpy()   # (dim,)

    feat = feat.astype(np.float32)
    return feat / np.linalg.norm(feat)          # ℓ2-norm = 1


# ---------------------------------------------------------------------
# 1) enrollment
# ---------------------------------------------------------------------
def load_known_speakers_from_samples(
    samples: List[Dict[str, Any]],
    huggingface_access_token: str | None = None
) -> Dict[str, np.ndarray]:

    embeddings: dict[str, np.ndarray] = {}

    for s in samples:
        name = s.get("name") or os.path.splitext(os.path.basename(s["url"]))[0]
        if name in _SPK_CACHE:                 # cached
            embeddings[name] = _SPK_CACHE[name]
            continue

        # --- read the sample (local path or download) ----------------
        path = s.get("file_path")
        if path is None:
            # download to temp
            resp = requests.get(s["url"], timeout=60)
            resp.raise_for_status()
            suffix = os.path.splitext(s["url"])[1] or ".wav"
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tf.write(resp.content);  tf.close()
            path = tf.name

        wav, _ = librosa.load(path, sr=16000, mono=True)
        emb = _embed(wav, token=huggingface_access_token)
        _SPK_CACHE[name] = emb
        embeddings[name] = emb
        logger.debug(f"[enrol] {name}: vector norm={np.linalg.norm(emb):.2f}")

        if "file_path" not in s:               # we downloaded → delete
            os.remove(path)

    return embeddings


# ---------------------------------------------------------------------
# 2) diarization post-processing
# ---------------------------------------------------------------------
def identify_speaker(vec: np.ndarray,
                     enrol: Dict[str, np.ndarray],
                     threshold: float = .5) -> tuple[str, float]:
    best, score = "Unknown", -1.0
    for name, ref in enrol.items():
        sim = 1 - cosine(vec, ref)             # cosine similarity
        if sim > score:
            best, score = name, sim
    return (best, score) if score >= threshold else ("Unknown", score)


def process_diarized_output(
    output: dict,
    audio_path: str,
    enrol: dict[str, np.ndarray],
    threshold: float = .5,
    huggingface_access_token: str | None = None
) -> dict:

    segments = output.get("segments", [])
    if not segments:
        return output

    # --- (a) embed every segment ------------------------------------
    for seg in segments:
        start, end = seg["start"], seg["end"]
        wav, _ = librosa.load(audio_path, sr=16000,
                              mono=True, offset=start, duration=end - start)
        if wav.size == 0:
            continue
        seg["embedding"] = _embed(wav, token=huggingface_access_token)

    # --- (b) centroid per diarization label -------------------------
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        clusters[seg["speaker"]].append(seg["embedding"])

    centroids = {lbl: np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0))
                 for lbl, v in clusters.items()}

    # --- (c) label propagation using cosine sim ---------------------
    for lbl, cent in centroids.items():
        name, sim = identify_speaker(cent, enrol, threshold)
        if name == "Unknown":
            continue
        for seg in segments:
            if seg["speaker"] == lbl:
                seg["speaker"] = name
                seg["similarity"] = float(sim)
                for w in seg.get("words", []):
                    w["speaker"] = name

    # remove heavy vectors so the dict is JSON-serialisable
    for seg in segments:
        seg.pop("embedding", None)

    return output