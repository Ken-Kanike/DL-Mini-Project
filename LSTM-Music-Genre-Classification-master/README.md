# Music Genre Classification with LSTMs

[![GitHub Profile](https://img.shields.io/badge/GitHub-Ken--Kanike-blue?logo=github)](https://github.com/Ken-Kanike)
[![Python](https://img.shields.io/badge/Python-3.x-brightgreen.svg)](https://www.python.org/)

Deep learning project for classifying audio files into music genres from the GTZAN music corpus using Recurrent Neural Networks (LSTM).

* Classify music files based on genre from the GTZAN music corpus
* GTZAN corpus is included for ease of use
* Implemented with multiple layers of LSTM Recurrent Neural Networks
* Implementations available in PyTorch, [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning), and Keras

---

## Getting Started

### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org)
* [Librosa](https://librosa.org/) &rarr; for audio feature extraction
* [Keras](https://keras.io)
* [PyTorch](https://pytorch.org)
* [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

### Test Trained LSTM Model
Pretrained model weights and architecture configurations can be found in `./weights/`.

To test the model on your custom audio file, run:
```bash
python predict_example.py path/to/custom/file.mp3
```

Or to test the model on the sample audio files provided:
```bash
python predict_example.py audio/classical_music.mp3
```

---

## Audio Features Extracted
* [MFCC (Mel-Frequency Cepstral Coefficients)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
* [Spectral Centroid](https://en.wikipedia.org/wiki/Spectral_centroid)
* [Chroma](http://labrosa.ee.columbia.edu/matlab/chroma-ansyn/)
* [Spectral Contrast](http://ieeexplore.ieee.org/document/1035731/)

---

## Model Performance

### Accuracy Benchmark

#### Training on GPU:
| Split | Loss | Accuracy |
| ----- | ---- | -------- |
| Training | `0.5801` | `0.7810` |
| Validation | `0.7345` | `0.7667` |
| Testing | `0.9008` | `0.6833` |

#### Training on CPU:
| Split | Loss | Accuracy |
| ----- | ---- | -------- |
| Training | `0.3486` | `0.8738` |
| Validation | `1.0284` | `0.7000` |
| Testing | `1.2097` | `0.6833` |

---

## Ideas for Improving Accuracy
* [GTZAN dataset considerations](https://arxiv.org/abs/1306.1461)
* Normalize MFCCs & other input features ([Recurrent BatchNorm](https://arxiv.org/pdf/1603.09025v4.pdf))
* Learning rate scheduling and decay
* Weight initialization tuning
* Better optimization hyperparameters (dropout tuning)

---

## References & Attribution
This project is referenced and adapted from the original open-source repository:
* Original Repository: [ruohoruotsi/LSTM-Music-Genre-Classification](https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification) by [@ruohoruotsi](https://github.com/ruohoruotsi)

