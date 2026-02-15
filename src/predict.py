import gc
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import torch
import whisperx
from pydub import AudioSegment
from scipy.spatial.distance import cosine

try:
    from cog import BasePredictor, Input, Path, BaseModel
except ImportError:
    from cog_stub import BasePredictor, Input, Path, BaseModel

# ── Logging ───────────────────────────────────────────────────────────

logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
for h in [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("container_log.txt", mode="a"),
]:
    h.setLevel(logging.DEBUG)
    h.setFormatter(_fmt)
    logger.addHandler(h)

# ── GPU config ────────────────────────────────────────────────────────

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

COMPUTE_TYPE = "float16"
DEVICE = "cuda"
WHISPER_ARCH = "./models/faster-whisper-large-v3"

# ── Anti-hallucination: repetitive pattern detector ───────────────────

_REPEAT_RE = re.compile(r"(.{1,10}?)\1{5,}")  # same 1-10 chars repeated 5+ times


def _is_hallucination(text: str) -> bool:
    """Return True if the segment text looks like a Whisper hallucination."""
    stripped = text.strip()
    if not stripped:
        return True
    if _REPEAT_RE.search(stripped):
        return True
    return False


def _filter_hallucinations(segments: list[dict]) -> list[dict]:
    """Remove segments that match hallucination patterns."""
    clean = []
    for seg in segments:
        txt = seg.get("text", "")
        if _is_hallucination(txt):
            logger.warning(f"Filtered hallucination: {txt[:80]!r}")
            continue
        clean.append(seg)
    return clean


# ── Output model ──────────────────────────────────────────────────────


class Output(BaseModel):
    segments: Any
    detected_language: str


# ── Predictor ─────────────────────────────────────────────────────────


class Predictor(BasePredictor):
    def setup(self):
        # Copy VAD model to torch cache if present in /models/vad
        vad_src = "./models/vad/whisperx-vad-segmentation.bin"
        vad_dst_dir = "../root/.cache/torch"
        os.makedirs(vad_dst_dir, exist_ok=True)
        if os.path.exists(vad_src):
            dst = os.path.join(vad_dst_dir, "whisperx-vad-segmentation.bin")
            if not os.path.exists(dst):
                shutil.copy(vad_src, vad_dst_dir)

        # Pre-load diarization pipeline into VRAM (community-1 = best pyannote 4.x model)
        logger.info("Loading diarization model into VRAM...")
        self.diarize_model = whisperx.DiarizationPipeline(
            model_name="pyannote/speaker-diarization-community-1",
            use_auth_token=os.environ.get("HF_TOKEN", "").strip(),
            device=DEVICE,
        )
        logger.info("Diarization model loaded.")

    def _diarize(self, audio, result, debug, min_speakers, max_speakers):
        """Diarize using the pre-loaded model in VRAM."""
        start_time = time.time_ns() / 1e6

        diarize_segments = self.diarize_model(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if debug:
            elapsed = time.time_ns() / 1e6 - start_time
            logger.info(f"Duration to diarize: {elapsed:.2f} ms")

        return result

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        language: str = Input(
            description="ISO code of the language spoken in the audio, None for auto-detection",
            default=None,
        ),
        language_detection_min_prob: float = Input(
            description="Min probability for language detection recursion",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="Max iterations for language detection",
            default=5,
        ),
        initial_prompt: str = Input(
            description="Optional prompt for the first window",
            default=None,
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription",
            default=64,
        ),
        temperature: float = Input(
            description="Temperature for sampling",
            default=0,
        ),
        vad_onset: float = Input(description="VAD onset", default=0.500),
        vad_offset: float = Input(description="VAD offset", default=0.363),
        align_output: bool = Input(
            description="Align whisper output for word-level timestamps",
            default=False,
        ),
        diarization: bool = Input(
            description="Assign speaker ID labels",
            default=False,
        ),
        huggingface_access_token: str = Input(
            description="HuggingFace token (read) for gated models",
            default=None,
        ),
        min_speakers: int = Input(
            description="Min speakers for diarization (blank if unknown)",
            default=None,
        ),
        max_speakers: int = Input(
            description="Max speakers for diarization (blank if unknown)",
            default=None,
        ),
        debug: bool = Input(
            description="Print compute times and memory info",
            default=False,
        ),
        speaker_verification: bool = Input(
            description="Enable speaker verification",
            default=False,
        ),
        speaker_samples: list = Input(
            description="Speaker sample dicts with 'url', optional 'name'/'file_path'",
            default=[],
        ),
    ) -> Output:
        with torch.inference_mode():
            # ── ASR options with anti-hallucination settings ──
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset,
            }

            # ── Language detection (optional) ──
            audio_duration = get_audio_duration(audio_file)

            if language is None and language_detection_min_prob > 0 and audio_duration > 30000:
                segments_duration_ms = 30000
                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms),
                )
                segments_starts = distribute_segments_equally(
                    audio_duration, segments_duration_ms, language_detection_max_tries
                )
                logger.info(
                    "Detecting language on segments at " + ", ".join(map(str, segments_starts))
                )
                detected = detect_language(
                    audio_file, segments_starts,
                    language_detection_min_prob, language_detection_max_tries,
                    asr_options, vad_options,
                )
                language = detected["language"]
                logger.info(
                    f"Detected {language} ({detected['probability']:.2f}) "
                    f"after {detected['iterations']} iterations"
                )

            # ── Load model ──
            t0 = time.time_ns() / 1e6
            model = whisperx.load_model(
                WHISPER_ARCH, DEVICE,
                compute_type=COMPUTE_TYPE,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
            )
            if debug:
                logger.info(f"Model load: {time.time_ns() / 1e6 - t0:.2f} ms")

            # ── Load audio ──
            t0 = time.time_ns() / 1e6
            audio = whisperx.load_audio(audio_file)
            if debug:
                logger.info(f"Audio load: {time.time_ns() / 1e6 - t0:.2f} ms")

            # ── Transcribe ──
            t0 = time.time_ns() / 1e6
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]
            if debug:
                logger.info(f"Transcribe: {time.time_ns() / 1e6 - t0:.2f} ms")

            # Free whisper VRAM
            gc.collect()
            torch.cuda.empty_cache()
            del model

            # ── Filter hallucinations ──
            result["segments"] = _filter_hallucinations(result["segments"])

            # ── Alignment (optional) ──
            if align_output:
                if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or \
                   detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                    result = align(audio, result, debug)
                else:
                    logger.warning(
                        f"Alignment not supported for language: {detected_language}"
                    )

            # ── Diarization (optional) ──
            if diarization:
                result = self._diarize(audio, result, debug, min_speakers, max_speakers)

            if debug:
                logger.info(
                    f"Peak GPU memory: "
                    f"{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )

        return Output(
            segments=result["segments"],
            detected_language=detected_language,
        )


# ── Helper functions ──────────────────────────────────────────────────


def get_audio_duration(file_path):
    return len(AudioSegment.from_file(file_path))


def detect_language(
    full_audio_file_path, segments_starts, language_detection_min_prob,
    language_detection_max_tries, asr_options, vad_options, iteration=1,
):
    model = whisperx.load_model(
        WHISPER_ARCH, DEVICE,
        compute_type=COMPUTE_TYPE,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    start_ms = segments_starts[iteration - 1]
    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)
    audio = whisperx.load_audio(audio_segment_file_path)

    from whisperx.audio import N_SAMPLES, log_mel_spectrogram

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    logger.info(f"Iteration {iteration} — detected: {language} ({language_probability:.2f})")
    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration,
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected

    next_detected = detect_language(
        full_audio_file_path, segments_starts,
        language_detection_min_prob, language_detection_max_tries,
        asr_options, vad_options, iteration + 1,
    )
    return next_detected if next_detected["probability"] > detected["probability"] else detected


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path
    audio = AudioSegment.from_file(input_file_path)
    extracted = audio[start_time_ms : start_time_ms + duration_ms]
    ext = input_file_path.suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = Path(tmp.name)
        extracted.export(tmp_path, format=ext.lstrip("."))
    return tmp_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available = total_duration - segments_duration
    spacing = available // (iterations - 1) if iterations > 1 else 0
    starts = [i * spacing for i in range(iterations)]
    if iterations > 1:
        starts[-1] = total_duration - segments_duration
    return starts


def align(audio, result, debug):
    t0 = time.time_ns() / 1e6
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE,
        return_char_alignments=False,
    )
    if debug:
        logger.info(f"Alignment: {time.time_ns() / 1e6 - t0:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    return result
