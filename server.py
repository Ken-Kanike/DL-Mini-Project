import os
import sys
import time
import tempfile
import logging
import numpy as np
import librosa
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tf_keras as keras
from tf_keras.models import model_from_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON_PATH = os.path.join(BASE_DIR, "LSTM-Music-Genre-Classification-master", "weights", "model.json")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "LSTM-Music-Genre-Classification-master", "weights", "model_weights.h5")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

GENRES = ["classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae"]

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)
sock = Sock(app)

# Real-time WebSocket Health & Telemetry Channel
@sock.route("/ws/health")
def ws_health(ws):
    logging.info("WebSocket client connected to /ws/health.")
    try:
        init_payload = {
            "type": "connected",
            "online": True,
            "model_ready": model is not None,
            "model_architecture": "Stacked 2-Layer LSTM",
            "total_parameters": 103816,
            "timestamp": time.time(),
            "genres": GENRES
        }
        ws.send(json.dumps(init_payload))
        while True:
            msg = ws.receive(timeout=15)
            ack = {
                "type": "heartbeat_ack",
                "online": True,
                "model_ready": model is not None,
                "timestamp": time.time()
            }
            ws.send(json.dumps(ack))
    except Exception:
        pass
    finally:
        logging.info("WebSocket client disconnected from /ws/health.")

# Global model instance
model = None

def load_lstm_model():
    global model
    if model is not None:
        return model
    
    logging.info(f"Loading LSTM model architecture from {MODEL_JSON_PATH}...")
    if not os.path.exists(MODEL_JSON_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError("Model JSON or weights file not found in weights directory!")
        
    with open(MODEL_JSON_PATH, "r") as f:
        model = model_from_json(f.read())
    
    logging.info(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
    model.load_weights(MODEL_WEIGHTS_PATH)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    logging.info("LSTM Model successfully loaded into memory! Trainable parameters: 103,816")
    return model

def extract_audio_features(file_path):
    """
    Extract 33 acoustic features matching GTZAN GenreFeatureData pipeline:
    - 13 MFCCs
    - 1 Spectral Centroid
    - 12 Chromagram
    - 7 Spectral Contrast
    Builds temporal sequence windows of length 128 across the entire audio.
    Returns:
    - batch matrix shape (N, 128, 33)
    - mean spectral centroid in Hz
    - audio duration in seconds
    - number of windows
    """
    timeseries_length = 128
    y, sr = librosa.load(file_path, sr=22050)
    duration = float(len(y) / sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    total_frames = mfcc.shape[1]
    mean_centroid = float(np.mean(spectral_center))

    # Combine all 33 features into a single matrix of shape (total_frames, 33)
    all_features = np.vstack([mfcc, spectral_center, chroma, spectral_contrast]).T

    windows = []
    if total_frames <= timeseries_length:
        pad = np.zeros((timeseries_length, 33), dtype=np.float64)
        pad[:total_frames, :] = all_features
        windows.append(pad)
    else:
        # Slide 128-frame windows with stride 64 across the audio
        stride = 64
        for start in range(0, total_frames - timeseries_length + 1, stride):
            window = all_features[start:start + timeseries_length, :]
            if window.shape[0] == timeseries_length:
                windows.append(window)

        if not windows:
            windows.append(all_features[:timeseries_length, :])

    batch = np.array(windows, dtype=np.float64)
    return batch, mean_centroid, duration, len(windows)

# Serve index.html
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# Serve other static assets
@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(BASE_DIR, path)

@app.route("/api/status", methods=["GET"])
def api_status():
    is_ready = model is not None
    return jsonify({
        "status": "ready" if is_ready else "initializing",
        "model_architecture": "Stacked 2-Layer LSTM",
        "total_parameters": 103816,
        "input_shape": [1, 128, 33],
        "feature_dimensions": 33,
        "genres": GENRES
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    t_start = time.time()
    temp_path = None
    try:
        loaded_model = load_lstm_model()
        
        # Check if an uploaded audio file was sent
        if "file" in request.files:
            uploaded_file = request.files["file"]
            suffix = os.path.splitext(uploaded_file.filename)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded_file.save(tmp.name)
                temp_path = tmp.name
            audio_source_path = temp_path
            source_type = "upload"
            source_name = uploaded_file.filename

        # Check if sample genre name was sent
        else:
            json_data = request.get_json(silent=True) or {}
            genre_query = request.form.get("genre") or request.args.get("genre") or json_data.get("genre")
            if not genre_query:
                return jsonify({"error": "No audio file or genre parameter provided"}), 400
            
            genre_clean = str(genre_query).strip().lower()
            
            # Map standard genre names to filenames
            filename_map = {
                "classical": "classical_music.mp3",
                "country": "country_music.mp3",
                "disco": "disco_music.mp3",
                "hiphop": "hip-hop_music.mp3",
                "hip-hop": "hip-hop_music.mp3",
                "jazz": "jazz_music.mp3",
                "metal": "metal_music.mp3",
                "pop": "pop_music.mp3",
                "reggae": "reggae_music.mp3"
            }
            
            fname = filename_map.get(genre_clean, f"{genre_clean}_music.mp3")
            audio_source_path = os.path.join(AUDIO_DIR, fname)
            
            # Fallback check
            if not os.path.exists(audio_source_path):
                audio_source_path = os.path.join(BASE_DIR, "LSTM-Music-Genre-Classification-master", "audio", fname)
                
            if not os.path.exists(audio_source_path):
                return jsonify({"error": f"Sample audio file not found for genre: {genre_query}"}), 404
                
            source_type = "sample"
            source_name = fname

        # Step 1: Feature Extraction
        t_extract_start = time.time()
        features_batch, mean_centroid, duration, num_windows = extract_audio_features(audio_source_path)
        feature_extraction_time_ms = round((time.time() - t_extract_start) * 1000, 1)

        # Step 2: Real LSTM Model Inference
        t_infer_start = time.time()
        all_preds = loaded_model.predict(features_batch, verbose=0)
        # Sequence pooling: mean softmax across all temporal windows
        preds = np.mean(all_preds, axis=0)
        inference_time_ms = round((time.time() - t_infer_start) * 1000, 1)

        total_latency_ms = round((time.time() - t_start) * 1000, 1)

        # Step 3: Probabilities and Argmax
        top_idx = int(np.argmax(preds))
        top_genre = GENRES[top_idx]
        top_confidence = round(float(preds[top_idx]) * 100, 2)

        probabilities = {}
        for idx, g in enumerate(GENRES):
            probabilities[g] = round(float(preds[idx]) * 100, 2)

        return jsonify({
            "success": True,
            "is_real_inference": True,
            "predicted_genre": top_genre.capitalize(),
            "confidence": top_confidence,
            "probabilities": probabilities,
            "spectral_centroid": f"{round(mean_centroid):,} Hz",
            "duration_seconds": round(duration, 2),
            "latency": f"{total_latency_ms} ms",
            "inference_time_ms": inference_time_ms,
            "feature_extraction_time_ms": feature_extraction_time_ms,
            "num_windows": num_windows,
            "tensor_shape": [num_windows, 128, 33],
            "source_type": source_type,
            "source_name": source_name,
            "model_architecture": "Stacked 2-Layer LSTM (103,816 parameters)"
        })

    except Exception as e:
        logging.exception("Error during audio classification")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_lstm_model()
    default_port = 7860 if "SPACE_ID" in os.environ else 5000
    port = int(os.environ.get("PORT", default_port))
    print("\n=======================================================")
    print(f">> LSTM Neural Engine Backend Running on port {port}")
    print("=======================================================\n", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
