# speaker_processing.py
import os, tempfile, logging, requests
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import librosa
import torch
from scipy.spatial.distance import cosine, cdist
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature

logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------
# 0) small helpers
# ---------------------------------------------------------------------
_EMBED_SEG = Inference(
    "pyannote/embedding",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_auth_token=os.getenv("HF_TOKEN")
)

def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)

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
                     threshold: float = .4) -> tuple[str, float]:
    best, score = "Unknown", -1.0
    for name, ref in enrol.items():
        sim = 1 - cosine(vec, ref)             # cosine similarity
        if sim > score:
            best, score = name, sim
    return (best, score) if score >= threshold else ("Unknown", score)




def process_diarized_output(
        output: dict,
        audio_path: str,
        enrol_embs: dict[str, np.ndarray],
        threshold: float = .40):
    """
    • compute 512-D pyannote embedding for every diarized segment
    • average them per label, compare to enrolments, relabel if cos≥thr
    """
    segs = output.get("segments", [])
    if not segs or "speaker" not in segs[0]:          # diarization disabled
        return output

    # ---------- 1) embed each segment --------------------------------
    for seg in segs:
        st, ed = seg["start"], seg["end"]
        wav, _ = librosa.load(audio_path, sr=16_000, mono=True,
                              offset=st, duration=ed - st)
        vec = _EMBED_SEG(torch.tensor(wav).unsqueeze(0)).cpu().nu
        seg["embedding"] = _l2(vec)



    # --- (a) embed every segment ------------------------------------
    # --- (b) centroid per diarization label -------------------------
    # --- build one list of embeddings per diarization label -----------
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)

    for seg in segments:
        if "speaker" not in seg:                       # <-- guard
            continue                                   # <-- guard
        clusters[seg["speaker"]].append(seg["embedding"])
        
    # ---------- 2) centroid per diarization label --------------------
    from collections import defaultdict
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for seg in segs:
        clusters[seg["speaker"]].append(seg["embedding"])

    centroids = {k: _l2(np.mean(v, axis=0)) for k, v in clusters.items()}
    

 # ---------- 3) speaker verification ------------------------------
    names   = list(enrol_embs)
    matrix  = np.stack([enrol_embs[n] for n in names])     # (N,512)

    for label, c in centroids.items():
        sims   = 1 - scipy.spatial.distance.cdist(c[None, :], matrix, "cosine")[0]
        best_i = sims.argmax()
        if sims[best_i] < threshold:
            continue                                      # keep original label
        new_name = names[best_i]
        for seg in segs:
            if seg["speaker"] == label:
                seg["speaker"]    = new_name
                seg["similarity"] = float(sims[best_i])
                logger.info(f"[verify] segment {start:.2f}–{end:.2f}: best match {real_name} @ {best_sim:.2f}")

    # ---------- 4) clean up ------------------------------------------
    for seg in segs:
        seg.pop("embedding", None)
    
    return output