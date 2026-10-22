import os
import sys
import time
import logging
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

# Bridge legacy Keras 2 formats seamlessly
import tf_keras as keras
from tf_keras.models import model_from_json

# Set page config
st.set_page_config(
    page_title="Music Genre Classifier | DL Mini Project",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); }
    .verdict-box {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(79, 172, 254, 0.2));
        border: 2px solid #00F2FE;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .verdict-title { font-size: 14px; text-transform: uppercase; color: #94A3B8; letter-spacing: 1px; }
    .verdict-genre { font-size: 42px; font-weight: 800; color: #00F2FE; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

GENRES = ["classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae"]
GENRE_EMOJIS = {
    "classical": "🎻", "country": "🤠", "disco": "🪩", "hiphop": "🎤",
    "jazz": "🎷", "metal": "⚡", "pop": "✨", "reggae": "🌴"
}

@st.cache_resource
def load_trained_model():
    model_path = "./weights/model.json"
    weights_path = "./weights/model_weights.h5"
    with open(model_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def extract_features(audio_path):
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

    return features, y, sr, mfcc, chroma

# Sidebar
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("**Deep Learning Mini Project — Group 25**")
st.sidebar.markdown("* Batch: **B23**")
st.sidebar.markdown("* Model: **Stacked 2-Layer LSTM (103K params)**")
st.sidebar.markdown("* Dataset: **GTZAN Genre Collection**")
st.sidebar.divider()

input_mode = st.sidebar.radio("Select Audio Source:", ["Preloaded Sample Audio", "Upload Custom Audio File"])

sample_map = {
    "Classical (Sample)": "audio/classical_music.mp3",
    "Jazz (Sample)": "audio/jazz_music.mp3",
    "Metal (Sample)": "audio/metal_music.mp3",
    "Pop (Sample)": "audio/pop_music.mp3",
    "Hip-Hop (Sample)": "audio/hip-hop_music.mp3",
    "Country (Sample)": "audio/country_music.mp3",
    "Disco (Sample)": "audio/disco_music.mp3",
    "Reggae (Sample)": "audio/reggae_music.mp3",
}

audio_path = None
if input_mode == "Preloaded Sample Audio":
    selected_sample = st.sidebar.selectbox("Choose Sample Track:", list(sample_map.keys()))
    audio_path = sample_map[selected_sample]
else:
    uploaded_file = st.sidebar.file_uploader("Upload Audio File (.mp3, .wav)", type=["mp3", "wav", "ogg"])
    if uploaded_file is not None:
        temp_dir = "./temp_upload"
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# Header Title
st.title("🎵 Music Genre Classification using LSTMs")
st.caption("Deep Learning Mini Project &bull; Batch B23 &bull; Arzoo Vaswani (125) &bull; Riya Maurya (133) &bull; Junaid Shaikh (137)")

if audio_path and os.path.exists(audio_path):
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("1. Audio Player & Waveform")
        st.audio(audio_path)

        with st.spinner("Extracting 33D Audio Features with Librosa..."):
            features, y, sr, mfcc, chroma = extract_features(audio_path)

        # Plot Waveform
        fig, ax = plt.subplots(figsize=(6, 2.2), facecolor="#0e131f")
        ax.set_facecolor("#0e131f")
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color="#00F2FE", lw=0.6)
        ax.set_title("Audio Amplitude Waveform (22,050 Hz)", color="#94A3B8", fontsize=10)
        ax.tick_params(colors="#64748B", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1E293B")
        st.pyplot(fig)

        # Feature Summary
        st.markdown("**Extracted Feature Matrix Shape:** `(1, 128, 33)`")
        st.markdown("- **MFCCs:** 13 Dimensions (Timbral texture)")
        st.markdown("- **Spectral Centroid:** 1 Dimension (Sound brightness)")
        st.markdown("- **Chromagram:** 12 Dimensions (Harmonic pitch distribution)")
        st.markdown("- **Spectral Contrast:** 7 Dimensions (Peak-to-valley energy difference)")

    with col2:
        st.subheader("2. Model Prediction & Probabilities")
        
        # Load Model
        model = load_trained_model()

        # Run inference
        t0 = time.time()
        preds = model.predict(features, verbose=0)[0]
        inference_time_ms = round((time.time() - t0) * 1000, 1)

        top_idx = int(np.argmax(preds))
        top_genre = GENRES[top_idx]
        top_conf = round(float(preds[top_idx]) * 100, 1)

        # Verdict Hero Banner
        emoji = GENRE_EMOJIS.get(top_genre, "🎵")
        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-title">Predicted Genre</div>
            <div class="verdict-genre">{emoji} {top_genre}</div>
            <div style="color: #00f5d4; font-weight: 600; font-size: 18px; margin-top: 6px;">
                Confidence: {top_conf}% Match
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Execution Stats
        m1, m2, m3 = st.columns(3)
        m1.metric("Inference Latency", f"{inference_time_ms} ms")
        m2.metric("Sequence Frames", "128")
        m3.metric("Trainable Params", "103,816")

        st.markdown("#### Softmax Class Probabilities:")
        prob_dict = {f"{GENRE_EMOJIS.get(g, '')} {g.capitalize()}": float(preds[i]) for i, g in enumerate(GENRES)}
        st.bar_chart(prob_dict)

else:
    st.info("👈 Please select or upload an audio track from the sidebar to test genre classification.")
