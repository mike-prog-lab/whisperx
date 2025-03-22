# speaker_id.py
import os
import torch
import librosa
import numpy as np
from pyannote.audio import Inference
from scipy.spatial.distance import cosine

def load_known_speakers(samples_dir="speaker_samples"):
    """
    Load known speaker audio files from samples_dir and return a dict of embeddings.
    Files can be MP3 or WAV. Speaker name is inferred from the filename (without extension).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Inference("pyannote/embedding", device=device)
    known_embeddings = {}
    
    for filename in os.listdir(samples_dir):
        if filename.lower().endswith((".mp3", ".wav")):
            speaker_name = os.path.splitext(filename)[0]
            filepath = os.path.join(samples_dir, filename)
            # Load audio using librosa (resample to 16000 Hz)
            waveform, sr = librosa.load(filepath, sr=16000, mono=True)
            # Compute embedding
            emb = model(torch.tensor(waveform).unsqueeze(0))
            # Flatten embedding and store
            known_embeddings[speaker_name] = emb.detach().cpu().numpy().flatten()
    return known_embeddings

def identify_speaker(segment_embedding, known_embeddings, threshold=0.75):
    """
    Compare a segment embedding against known speakers. Returns the best matching speaker and similarity score.
    If no match exceeds the threshold, returns "Unknown".
    """
    best_match = None
    best_score = -1
    for speaker, known_emb in known_embeddings.items():
        score = 1 - cosine(segment_embedding, known_emb)
        if score > best_score:
            best_score = score
            best_match = speaker
    if best_score >= threshold:
        return best_match, best_score
    else:
        return "Unknown", best_score
    
def process_diarized_output(output, audio_filepath, known_embeddings):
    """
    For each diarized segment in the output, extract its audio from the given audio file,
    compute its embedding, and update the segment's speaker label using known_embeddings.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = Inference("pyannote/embedding", device=device)
    
    segments = output.get("segments", [])
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        duration = end - start
        try:
            waveform, sr = librosa.load(audio_filepath, sr=16000, mono=True, offset=start, duration=duration)
        except Exception as e:
            print(f"Failed to load audio segment from {audio_filepath} for {start}-{end}: {e}")
            continue
        if len(waveform) == 0:
            continue
        seg_emb = embedding_model(torch.tensor(waveform).unsqueeze(0))
        seg_emb = seg_emb.detach().cpu().numpy().flatten()
        speaker, similarity = identify_speaker(seg_emb, known_embeddings)
        segment["speaker"] = speaker
        segment["similarity"] = similarity
    return output
