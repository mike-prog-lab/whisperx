# rp_handler.py
import os
import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file if present
token = os.getenv("HF_TOKEN")

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

def run(job):
    job_input = job['input']
    job_id = job['id']
    # Input validation
    validated_input = validate(job_input, INPUT_VALIDATIONS)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    # Download audio file (any audio/video type)
    
    try:
        audio_file_path = download_files_from_urls(job['id'], [job_input['audio_file']])[0]
    except KeyError:
        logger.error("Missing 'audio_file' key in job input.", exc_info=True)
        return {"error": "Missing 'audio_file' key in job input."}
    except Exception as e:
        logger.error(f"Error downloading audio file: {str(e)}", exc_info=True)
        return {"error": f"Failed to download audio file: {str(e)}"}
    logger.debug(f"Downloaded main audio file to: {audio_file_path}")

    # Download speaker sample files, if provided.
    logger.debug("Job input speaker_samples: " + str(job_input.get('speaker_samples', [])))
    speaker_samples = job_input.get('speaker_samples', [])
    if speaker_samples:
        # Extract URLs from each sample.
        sample_urls = [sample.get("url") for sample in speaker_samples if sample.get("url")]
        if sample_urls:
            try:
                downloaded_sample_paths = download_files_from_urls(job['id'], sample_urls)
                if len(downloaded_sample_paths) != len(sample_urls):
                    raise ValueError("Mismatch between sample URLs and downloaded file paths.")
                # Update each sample dictionary with a 'file_path' key.
                for i, sample in enumerate(speaker_samples):
                    sample["file_path"] = downloaded_sample_paths[i]
                logger.debug(f"Downloaded speaker samples: {downloaded_sample_paths}")
            except Exception as e:
                logger.error(f"Error downloading speaker samples: {str(e)}", exc_info=True)
                return {"error": f"Failed to download speaker samples: {str(e)}"}
        else:
            logger.debug("No valid speaker sample URLs provided.")

    
    # Prepare input for prediction
    predict_input = {
        'audio_file': audio_file_path,
        'language': job_input.get('language', None),
        'language_detection_min_prob': job_input.get('language_detection_min_prob', 0),
        'language_detection_max_tries': job_input.get('language_detection_max_tries', 5),
        'initial_prompt': job_input.get('initial_prompt', None),
        'batch_size': job_input.get('batch_size', 64),
        'temperature': job_input.get('temperature', 0),
        'vad_onset': job_input.get('vad_onset', 0.500),
        'vad_offset': job_input.get('vad_offset', 0.363),
        'align_output': job_input.get('align_output', False),
        'diarization': job_input.get('diarization', False),
        'huggingface_access_token': job_input.get('huggingface_access_token', None),
        'min_speakers': job_input.get('min_speakers', None),
        'max_speakers': job_input.get('max_speakers', None),
        'debug': job_input.get('debug', False),
        'speaker_verification': job_input.get('speaker_verification', False),
        'speaker_samples': job_input.get('speaker_samples', [])
    }
    
    try:
        try:
            # Run prediction (which includes transcription, diarization, etc.)
            result = MODEL.predict(**predict_input)
            logger.debug("Prediction completed successfully.")
            
            # Convert prediction output to dict for JSON serialization
            output_dict = {
                "segments": result.segments,
                "detected_language": result.detected_language
            }
            try:
                from speaker_processing import load_known_speakers_from_samples, process_diarized_output
            except ImportError as e:
                logger.error(f"Error importing speaker_processing module: {str(e)}", exc_info=True)
                return {"error": f"Speaker verification module not found: {str(e)}"}
            
            # If speaker verification is enabled, process the diarized output
            if predict_input.get('speaker_verification', False):
                try:
                    speaker_samples = predict_input.get('speaker_samples', [])
                    if speaker_samples:
                        known_embeddings = load_known_speakers_from_samples(speaker_samples)
                        output_dict = process_diarized_output(output_dict, audio_file_path, known_embeddings)
                except Exception as e:
                    logger.error(f"Error during speaker verification: {str(e)}", exc_info=True)
                    return {"error": f"Speaker verification failed: {str(e)}"}
            
            # Cleanup downloaded files and temporary job directory
            try:
                rp_cleanup.clean(['input_objects'])
                cleanup_job_files(job_id)
            except Exception as e:
                logger.warning(f"Error during cleanup: {str(e)}", exc_info=True)
            
            return output_dict
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return {"error": f"Prediction failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return {"error": str(e)}

runpod.serverless.start({"handler": run})