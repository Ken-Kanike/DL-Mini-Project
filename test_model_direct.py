import os
import sys
import numpy as np

# Suppress TF messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tf_keras as keras
from tf_keras.models import model_from_json
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(BASE_DIR, "LSTM-Music-Genre-Classification-master", "weights", "model.json")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "LSTM-Music-Genre-Classification-master", "weights", "model_weights.h5")

GENRES = ["classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae"]

def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\junai\Desktop\music.mp3"
    
    print("\n" + "=" * 60)
    print("  DIRECT TENSORFLOW LSTM WEIGHT INFERENCE")
    print("=" * 60)
    print(f"Target Audio: {audio_path}")
    print(f"Loading Model Architecture: {MODEL_JSON}")
    print(f"Loading Weights File:       {MODEL_WEIGHTS}")
    
    # 1. Load Model Architecture
    with open(MODEL_JSON, "r") as f:
        model = model_from_json(f.read())
        
    # 2. Load Real Pretrained Weights
    model.load_weights(MODEL_WEIGHTS)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print(f"Model Name:                 {model.name}")
    print(f"Total Trainable Parameters: {model.count_params():,}")
    print("Layers:")
    for layer in model.layers:
        print(f"  - {layer.name} ({layer.__class__.__name__}): output shape {layer.output_shape}")
        
    print("\nExtracting Librosa 33D Acoustic Features...")
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
    
    all_features = np.vstack([mfcc, spectral_center, chroma, spectral_contrast]).T
    total_frames = all_features.shape[0]
    duration = len(y) / sr
    
    print(f"Audio Duration:             {duration:.2f} seconds")
    print(f"Total Extracted Frames:     {total_frames}")
    
    # 128-frame window slicing
    timeseries_length = 128
    stride = 64
    windows = []
    if total_frames <= timeseries_length:
        pad = np.zeros((timeseries_length, 33), dtype=np.float64)
        pad[:total_frames, :] = all_features
        windows.append(pad)
    else:
        for start in range(0, total_frames - timeseries_length + 1, stride):
            w = all_features[start:start + timeseries_length, :]
            if w.shape[0] == timeseries_length:
                windows.append(w)
                
    batch = np.array(windows, dtype=np.float64)
    print(f"Input Tensor Shape:         {batch.shape} (Batch of {len(windows)} windows)")
    
    print("\nRunning model.predict(batch) via TensorFlow engine...")
    raw_preds = model.predict(batch, verbose=0)
    
    # Mean across windows
    mean_preds = np.mean(raw_preds, axis=0)
    top_idx = int(np.argmax(mean_preds))
    top_genre = GENRES[top_idx]
    
    print("\n" + "-" * 60)
    print("  RAW TENSORFLOW SOFTMAX PROBABILITIES FROM MODEL WEIGHTS:")
    print("-" * 60)
    for g, prob in zip(GENRES, mean_preds):
        bar_len = int(prob * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"  {g.upper():<10} | [{bar}] {prob*100:6.2f}%")
        
    print("-" * 60)
    print(f">> ARGMAX PREDICTED CLASS: {top_genre.upper()} ({mean_preds[top_idx]*100:.2f}%)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
