# rp_handler.py
######SETTING HF_TOKENT#############
from speaker_profiles import load_embeddings, relabel  # top of file
import os
import logging
from huggingface_hub import login, whoami
import torch
import numpy as np
from speaker_processing import process_diarized_output, load_known_speakers_from_samples
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from speechbrain.pretrained import EncoderClassifier # type: ignore

def spk_embed(wave_16k_mono: np.ndarray) -> np.ndarray:
    wav = torch.tensor(wave_16k_mono).unsqueeze(0).to(device)
    return ecapa.encode_batch(wav).squeeze(0).cpu().numpy()

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Grab the HF_TOKEN from environment
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if not hf_token.startswith("hf_"):
    print(f"Token malformed or missing 'hf_' prefix. Forcing correction...")
    hf_token = "h" + hf_token  # Force adding the 'h' (temporary fix)

#print(f" Final HF_TOKEN used: #{hf_token}")
if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")  # Show only start of token for security
        login(token=hf_token, add_to_git_credential=False)  # Safe for container runs
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error(" Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")
##############

import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
import os
import copy
import logging
import sys
# Create a custom logger
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)  # capture everything at DEBUG or above

# Create console handler and set level to DEBUG
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

# Create file handler to write logs to 'container_log.txt'
file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)




MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
error_log = []
def run(job):
    job_id     = job["id"]
    job_input  = job["input"]

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # ------------- 1) download primary audio ------------------------
    try:
        audio_file_path = download_files_from_urls(job_id,
                                                   [job_input["audio_file"]])[0]
        logger.debug(f"Audio downloaded → {audio_file_path}")
    except Exception as e:
        logger.error("Audio download failed", exc_info=True)
        return {"error": f"audio download: {e}"}

    # ------------- 2) download speaker profiles (optional) ----------
    speaker_profiles = job_input.get("speaker_samples", [])   # ← list of dicts
    if speaker_profiles:
        urls = [s.get("url") for s in speaker_profiles if s.get("url")]
        if urls:
            local_paths = download_files_from_urls(job_id, urls)
            for s, path in zip(speaker_profiles, local_paths):
                s["file_path"] = path                       # mutate in-place
                logger.debug(f"Profile {s.get('name')} → {path}")
    # ----------------------------------------------------------------

    # ------------- 3) call WhisperX / VAD / diarization -------------
    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 64),
        "temperature"              : job_input.get("temperature", 0),
        "vad_onset"                : job_input.get("vad_onset", 0.50),
        "vad_offset"               : job_input.get("vad_offset", 0.363),
        "align_output"             : job_input.get("align_output", False),
        "diarization"              : job_input.get("diarization", False),
        "huggingface_access_token" : job_input.get("huggingface_access_token"),
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)             # <-- heavy job
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments"         : result.segments,
        "detected_language": result.detected_language
    }
    # ------------------------------------------------embedding-info----------------
    # 4) speaker verification (optional)
    embeddings = {} # ensure the name is always bound
    if job_input.get("speaker_verification", False):
        logger.info(f"Speaker-verification requested: True")
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_samples,
                huggingface_access_token=predict_input["huggingface_access_token"]
            )
            logger.info(f"  • Enrolled {len(embeddings)} profiles")
        except Exception as e:
            logger.error("Failed loading speaker profiles", exc_info=True)
            output_dict["warning"] = f"enrollment skipped: {e}"

        if embeddings:  # only attempt verification if we actually got something
            try:
                output_dict = process_diarized_output(
                    output_dict,
                    audio_file_path,
                    embeddings,
                    huggingface_access_token=predict_input["huggingface_access_token"]
                )
            except Exception as e:
                logger.error("Error during speaker verification", exc_info=True)
                output_dict["warning"] = f"verification skipped: {e}"
        else:
            logger.info("No embeddings to verify against; skipping verification step")
    
    
    # sv = bool(job_input.get("speaker_verification", False))
    # logger.info(f"Speaker-verification requested: {sv}")
    # if sv and speaker_profiles:
    #     try:
    #         logger.info(f"  • Enrolling {len(speaker_profiles)} profiles")



    # 5) cleanup
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    finally:
        if error_log:
            output["error_log"] = "\n".join(error__log)             # if you have any errors, attach them to the output
            
    return output_dict

runpod.serverless.start({"handler": run})