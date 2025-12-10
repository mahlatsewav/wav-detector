# Audio Analyzer / Wav Detector

A Python-based audio analysis tool built with [Streamlit](https://streamlit.io/) and [Librosa](https://librosa.org/). This application allows users to upload audio files to extract various musical and acoustic features and visualize the audio data.

## Features

- **Audio Feature Extraction**: Automatically extracts key audio characteristics:
  - **BPM (Tempo)**: Beats per minute.
  - **Key**: Musical key detection (e.g., C, F#).
  - **RMS Energy**: Loudness/Energy level.
  - **Spectral Features**: Centroid, Bandwidth, and Rolloff.
  - **MFCCs**: Mel-frequency cepstral coefficients (first 20 means).
- **Visualization**:
  - **Waveform**: Time-domain representation of the audio signal.
  - **Spectrogram**: Frequency-domain visualization.
- **Interactive UI**: Upload and analyze audio files directly in the browser.

## Project Structure

```
wav-detector/
├── app.py                  # Main Streamlit application entry point
├── feature_extraction.py   # Core logic for audio feature extraction using Librosa
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
├── models/                 # Directory containing trained models (e.g., genre_model.pkl)
├── data/                   # Directory for data storage
└── assets/                 # Project assets
```

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <repository-url>
    cd wav-detector
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2.  **Use the interface**:
    -   The app will open in your default web browser (usually at `http://localhost:8501`).
    -   Click **"Upload audio"** to select an audio file (supported formats: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`).
    -   Wait for the analysis to complete.
    -   View the extracted features (BPM, Key, etc.) and visualizations.

## Technologies Used

-   **Streamlit**: For the web interface.
-   **Librosa**: For audio processing and feature extraction.
-   **NumPy**: For numerical operations.
-   **Matplotlib**: For plotting waveforms and spectrograms.
