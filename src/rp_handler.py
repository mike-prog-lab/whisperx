import logging
import os
import shutil
import sys

import runpod
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, whoami
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from predict import Predictor
from rp_schema import INPUT_VALIDATIONS
from speaker_processing import (
    identify_speakers_on_segments,
    load_known_speakers_from_samples,
    relabel_speakers_by_avg_similarity,
)

# ── Logging ───────────────────────────────────────────────────────────

logger = logging.getLogger("rp_handler")
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

# ── HuggingFace authentication ────────────────────────────────────────

load_dotenv(find_dotenv())
hf_token = os.environ.get("HF_TOKEN", "").strip()

if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face authenticated as: {user['name']}")
    except Exception:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No HF_TOKEN found — gated models will not be available.")

# ── Model setup ───────────────────────────────────────────────────────

MODEL = Predictor()
MODEL.setup()


# ── Helpers ───────────────────────────────────────────────────────────


def cleanup_job_files(job_id, jobs_directory="/jobs"):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing {job_path}: {e}", exc_info=True)


# ── RunPod serverless handler ─────────────────────────────────────────


def run(job):
    job_id = job["id"]
    job_input = job["input"]

    # Validate schema
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # 1) Download audio
    try:
        audio_file_path = download_files_from_urls(
            job_id, [job_input["audio_file"]]
        )[0]
        logger.debug(f"Audio downloaded → {audio_file_path}")
    except Exception as e:
        logger.error("Audio download failed", exc_info=True)
        return {"error": f"audio download: {e}"}

    # 2) Speaker profiles (optional)
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles, huggingface_access_token=hf_token
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)

    # 3) Run WhisperX pipeline
    predict_input = {
        "audio_file": audio_file_path,
        "language": job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt": job_input.get("initial_prompt"),
        "batch_size": job_input.get("batch_size", 64),
        "temperature": job_input.get("temperature", 0),
        "vad_onset": job_input.get("vad_onset", 0.50),
        "vad_offset": job_input.get("vad_offset", 0.363),
        "align_output": job_input.get("align_output", False),
        "diarization": job_input.get("diarization", False),
        "huggingface_access_token": job_input.get("huggingface_access_token"),
        "min_speakers": job_input.get("min_speakers"),
        "max_speakers": job_input.get("max_speakers"),
        "debug": job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments": result.segments,
        "detected_language": result.detected_language,
    }

    # 4) Speaker verification (optional)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1,
            )
            output_dict["segments"] = relabel_speakers_by_avg_similarity(
                segments_with_speakers
            )
            logger.info("Speaker identification completed.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings; skipping speaker identification.")

    # 5) Cleanup
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict


runpod.serverless.start({"handler": run})
