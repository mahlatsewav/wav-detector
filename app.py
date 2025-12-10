import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from feature_extraction import extract_features

# ----------------------
# UI CONFIG
# ----------------------
st.set_page_config(
    page_title="Audio Analyzer",
    layout="wide",
    page_icon="🎵",
)

st.title("Audio Feature Extraction & Analysis Tool")
st.caption("Upload an audio file to analyze BPM, key, loudness, and more.")

# ----------------------
# FILE UPLOAD
# ----------------------
uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "ogg", "flac", "m4a"])

if uploaded_file:
    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp_audio.wav")

    # Extract features
    with st.spinner("Extracting features..."):
        results = extract_features("temp_audio.wav")

    # ----------------------
    # FEATURE CARDS
    # ----------------------
    st.subheader("Key Audio Stats")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BPM", f"{results['tempo']:.1f}")
    col2.metric("Key", results["key"])
    col3.metric("Duration (s)", f"{results['duration']:.2f}")
    col4.metric("RMS Energy", f"{results['rms']:.4f}")

    # ----------------------
    # DETAILED FEATURE PRINTOUT
    # ----------------------
    with st.expander("🔍 Full Feature Data", expanded=False):
        st.json(results)

    # ----------------------
    # WAVEFORM VISUALIZATION
    # ----------------------
    st.subheader("📊 Waveform & Spectrogram")

    y, sr = librosa.load("temp_audio.wav")

    colA, colB = st.columns(2)

    with colA:
        st.write("**Waveform**")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

    with colB:
        st.write("**Spectrogram**")
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log", ax=ax2)
        fig2.colorbar(img, ax=ax2)
        ax2.set_title("Spectrogram")
        st.pyplot(fig2)

    st.success("Feature extraction complete!")
