import librosa
import numpy as np

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def extract_features(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # --- Tempo / BPM ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # --- Key Detection using Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_index = chroma.mean(axis=1).argmax()
    detected_key = KEYS[key_index]

    # --- Spectral Features ---
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

    # --- RMS Energy (Loudness) ---
    rms = float(librosa.feature.rms(y=y).mean())

    # --- MFCCs (20 Coefficients) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = mfcc.mean(axis=1).tolist()

    # --- Duration ---
    duration = float(librosa.get_duration(y=y, sr=sr))

    return {
        "tempo": tempo,
        "key": detected_key,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "rms": rms,
        "duration": duration,
        "mfcc": mfcc_means,
    }
