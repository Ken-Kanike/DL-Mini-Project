# Music Genre Classification Using LSTM-Based Deep Learning on Audio Features

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-3B82F6?style=for-the-badge&logo=googlechrome&logoColor=white)](https://ken-kanike.github.io/DL-Mini-Project/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Ken--Kanike-black?style=for-the-badge&logo=github)](https://github.com/Ken-Kanike/DL-Mini-Project)
[![Model](https://img.shields.io/badge/Architecture-2--Layer_LSTM-10B981?style=for-the-badge)](https://github.com/Ken-Kanike/DL-Mini-Project)

> 🌐 **Live Interactive Web Application:** [https://ken-kanike.github.io/DL-Mini-Project/](https://ken-kanike.github.io/DL-Mini-Project/)  
> Test preloaded GTZAN audio samples or upload custom tracks directly in the browser with real-time waveform visualization and LSTM classification.

---

## Deep Learning Mini Project — Group 25

---

## Project Information

| Domain | Application Name | DL Architectures Used | Project Topic |
| :--- | :--- | :--- | :--- |
| Audio Processing, Music Information Retrieval (MIR) | Automated Music Genre Classification | Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN) | Music Genre Classification Using LSTM-Based Deep Learning on Audio Features |

---

## Team Details

| Group No. | Batch | Roll No. | Name of Student |
| :---: | :---: | :---: | :--- |
| **25** | B23 | 125 | Arzoo Vaswani |
| **25** | B23 | 133 | Riya Maurya |
| **25** | B23 | 137 | Junaid Shaikh |

---

## 1. Problem Statement
Categorizing music into genres manually is subjective, laborious, and impractical given the exponential volume of digital audio uploaded daily to streaming platforms. Traditional machine learning techniques rely heavily on handcrafted static global statistics (such as average energy or zero-crossing rate), which fail to capture the complex sequential, rhythmic, and harmonic temporal progressions in music. 

The objective of this project is to build an automated, sequence-aware deep learning classification system that takes raw musical audio, extracts high-dimensional acoustic and spectral temporal features across time slices, and utilizes stacked Long Short-Term Memory (LSTM) Recurrent Neural Networks to classify tracks into their respective genres with high accuracy.

---

## 2. Dataset Requirements & Specifications
* **Dataset Used**: GTZAN Genre Collection (benchmark dataset in Music Information Retrieval).
* **Target Classes**: 8 distinct genres:
  * `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`
* **Audio Characteristics**:
  * Original format: 30-second clips, 22,050 Hz sampling rate, 16-bit mono audio.
* **Data Splits**:
  * **Training Set**: 420 audio tracks (divided into mini-batches of 35 sequences).
  * **Validation (Dev) Set**: 120 audio tracks.
  * **Test Set**: 60 audio tracks.
* **Input Tensor Shape**: `(N, 128, 33)`
  * `N`: Number of audio samples in the batch.
  * `128`: Sequence time steps (extracted over audio frames).
  * `33`: Acoustic feature vector dimension per time step.

---

## 3. Implementation Methodology & Feature Extraction
Rather than feeding raw high-frequency waveforms directly to an RNN (which is computationally prohibitive), domain-specific audio feature representations are extracted using **Librosa**:

1. **Mel-Frequency Cepstral Coefficients (MFCCs) [13 dimensions]**:
   * Captures the spectral envelope and timbral texture mimicking human auditory perception.
2. **Spectral Centroid [1 dimension]**:
   * Represents the "brightness" or center of mass of the sound frequency spectrum.
3. **Chromagram (STFT) [12 dimensions]**:
   * Projects spectral energy onto the 12 semi-tone pitch classes ($C, C\sharp, D, \dots, B$), representing harmonic and melodic content.
4. **Spectral Contrast [7 dimensions]**:
   * Measures the energy difference between spectral peaks and valleys across 7 sub-bands, distinguishing harmonic clarity from broadband noise.
* **Combined Feature Vector**: $13 + 1 + 12 + 7 = 33$ acoustic features per time slice across a temporal sequence of 128 steps.

---

## 4. Architecture Design

The deep learning model is built using a stacked 2-layer Recurrent Neural Network (LSTM):

```
                   Input Audio Sequence (128 time steps, 33 features)
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    LSTM Layer 1: 128 Hidden Units     │
                      │    (return_sequences=True,            │
                      │     dropout=0.05, rec_dropout=0.35)   │
                      └───────────────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    LSTM Layer 2: 32 Hidden Units      │
                      │    (return_sequences=False,           │
                      │     dropout=0.05, rec_dropout=0.35)   │
                      └───────────────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Dense Layer: 8 Units (Softmax)     │
                      │    (Outputs genre probability vector) │
                      └───────────────────────────────────────┘
```

### Parameter Breakdown:
* **LSTM Layer 1**: $4 \times [(33 + 128 + 1) \times 128] = 82,944$ parameters.
* **LSTM Layer 2**: $4 \times [(128 + 32 + 1) \times 32] = 20,608$ parameters.
* **Dense Layer**: $(32 + 1) \times 8 = 264$ parameters.
* **Total Trainable Parameters**: **103,816** (~405 KB).

---

## 5. Model Training Process & Experimental Results

### Training Hyperparameters
* **Loss Function**: Categorical Crossentropy ($\mathcal{L} = -\sum y_i \log(\hat{y}_i)$)
* **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
* **Learning Rate**: $0.001$
* **Batch Size**: 35
* **Epochs**: 400
* **Regularization**: Input dropout (0.05) and Recurrent dropout (0.35) inside recurrent memory cells to prevent overfitting.

### Benchmark Performance at Epoch 400

| Dataset Split | Loss | Accuracy |
| :--- | :---: | :---: |
| **Training Set** | `0.3486` | **87.38%** |
| **Validation Set** | `0.7345` | **76.67%** |
| **Test Set** | `0.9008` | **68.33%** |

---

## 6. Advantages & Limitations

### Advantages:
1. **Temporal Awareness**: Unlike static multi-layer perceptrons (MLPs) or classical SVMs, LSTMs retain long-term dependencies in musical rhythm and melody across time steps.
2. **Compact & Efficient Architecture**: With only ~103K parameters, inference executes in under 2 seconds on a standard CPU.
3. **Multi-Domain Audio Features**: Incorporating timbral (MFCC), tonal (Chroma), and structural (Contrast, Centroid) features offers rich discriminative representation across widely differing genres.

### Limitations:
1. **Class Confusion**: Genres with shared musical instrumentation and structures (e.g., Rock vs. Metal or Pop vs. Disco) can exhibit boundary ambiguity.
2. **Fixed Audio Window**: The model analyzes fixed 128-frame time windows (approx. 3–5 seconds of representative audio) rather than full multi-minute songs.

---

## 7. Expected Outcomes
* When provided with custom or unseen audio clips (`.mp3` or `.wav`), the system automatically extracts the 33-dimensional feature matrices, feeds them through the trained LSTM network, and outputs the predicted music genre along with class probabilities.

---

## 8. Project Execution & Live Demonstration Options

### Option 1: Interactive Streamlit Web UI (Recommended for Teacher Presentation)
Launch the full interactive web application with audio players, live waveform visualizer, feature inspection, and real-time LSTM classification:

```bash
cd LSTM-Music-Genre-Classification-master
streamlit run app.py
```
* Automatically opens in your browser at `http://localhost:8501`.
* Allows uploading custom `.mp3` / `.wav` files or testing preloaded genre tracks.

---

### Option 2: Live GitHub Pages Web App
The project includes a client-side Web Audio API & Neural Engine interface ready to test on any browser or phone without installing anything:
* **Live Demo URL**: [https://ken-kanike.github.io/DL-Mini-Project/](https://ken-kanike.github.io/DL-Mini-Project/)
* Or open `index.html` directly in any web browser!

---

### Option 3: Terminal CLI Prediction
To run inference directly from the command line:

```bash
cd LSTM-Music-Genre-Classification-master
```

Run prediction on sample audio:
```powershell
# Test Classical music
python -c "import sys, tf_keras, tf_keras.models; sys.modules['keras'] = tf_keras; sys.modules['keras.models'] = tf_keras.models; sys.argv = ['predict_example.py', 'audio/classical_music.mp3']; exec(open('predict_example.py').read())"

# Test Metal music
python -c "import sys, tf_keras, tf_keras.models; sys.modules['keras'] = tf_keras; sys.modules['keras.models'] = tf_keras.models; sys.argv = ['predict_example.py', 'audio/metal_music.mp3']; exec(open('predict_example.py').read())"

# Test Pop music
python -c "import sys, tf_keras, tf_keras.models; sys.modules['keras'] = tf_keras; sys.modules['keras.models'] = tf_keras.models; sys.argv = ['predict_example.py', 'audio/pop_music.mp3']; exec(open('predict_example.py').read())"
```

**Expected Console Output:**
```
1/1 [==============================] - 1s 1s/step
Model predict: classical
```
