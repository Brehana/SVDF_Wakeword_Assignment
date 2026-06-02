# EdgeAI — SVDF Wake-Word Detection for OpenMV / Arduino Nicla Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brehana/SVDF_Wakeword_Assignment/blob/main/svdf_wakeword_completed.ipynb)

A complete pipeline for training and deploying a lightweight **SVDF (Singular Value Decomposition Filter)** wake-word detector on the **Arduino Nicla Vision** via **OpenMV**.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `edgeai_svdf_wakeword_assignment.ipynb` | Original assignment notebook (read-only reference) |
| `svdf_wakeword_completed.ipynb` | ✅ Completed implementation — start here |
| `nn_audio_teaching_notebook.ipynb` | Teaching reference: SVDF vs CNN vs LSTM for audio |

---

## 🚀 Quick Start (Google Colab)

Click the badge above, or open directly:  
[`svdf_wakeword_completed.ipynb` on Colab](https://colab.research.google.com/github/Brehana/SVDF_Wakeword_Assignment/blob/main/svdf_wakeword_completed.ipynb)

The notebook runs fully out-of-the-box with `USE_SYNTHETIC_DATA = True` — no dataset download required for a quick demo.

---

## 🎯 What This Project Does

1. **Trains a wake-word detector** for two wake words:
   - `"Hey Jarvis"`
   - `"OK Nambu"`
2. **Builds a strong `unknown` class** using Google Speech Commands data
3. **Extracts MFCC features** (49 frames × 40 bins) from 16 kHz mono audio
4. **Implements an SVDF architecture**: low-rank factorization via Dense (frequency projection) + Causal DepthwiseConv1D (temporal filter)
5. **Exports to TFLite float16** — required for Nicla Vision hardware (not int8)
6. **Generates an OpenMV `main.py`** deployment script for on-device inference

---

## 🏗️ SVDF Architecture

SVDF approximates the expensive full time-frequency weight matrix with a low-rank factorization:

```
W ≈ α × β
```

where:
- **α** = frequency projection weights (`Dense` layer)
- **β** = temporal filter weights (`DepthwiseConv1D` with causal padding)

This is equivalent to `FIFO buffer + learned filter over time` — memory-efficient for microcontrollers.

---

## 🔧 Audio Configuration

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Clip duration | 1,000 ms |
| Window size | 30 ms |
| Window stride | 20 ms |
| MFCC bins | 40 |
| Feature shape | (49, 40) |

---

## 📦 Hardware Deployment

### Target: Arduino Nicla Vision via OpenMV

1. Run all cells in `svdf_wakeword_completed.ipynb`
2. Copy `svdf_wakeword_float16.tflite` → SD card root
3. Copy generated `main.py` → SD card root
4. Insert SD card into Nicla Vision
5. Open **OpenMV IDE**, connect, click **Run**

> **Note:** The model uses **float16 quantization** (not int8). This is a hardware requirement — the Nicla Vision's OpenMV runtime does not support full int8 quantized ops for this architecture. Float16 weights still provide ~2× model size reduction with no accuracy loss.

### Detection Threshold

The deployment script uses a **0.85 confidence threshold** for wake-word detection to minimize false positives. Adjust in `main.py`:

```python
DETECTION_THRESHOLD = 0.85  # increase to reduce false positives
```

---

## 📊 Dataset Sources

| Class | Source |
|---|---|
| `hey_jarvis` | [Kaggle: Hey Jarvis wake word](https://www.kaggle.com/search?q=hey+jarvis+wake+word) |
| `ok_nambu` | Custom recordings (16 kHz mono WAV) |
| `unknown` | [Synthetic Speech Commands](https://www.kaggle.com/datasets/jbuchner/synthetic-speech-commands-dataset) — downloaded automatically |
| `silence` | [Environmental Sound Classification](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50) |

> Recommended class ratio: 1× wake word : 10× unknown : 3× silence

---

## 📋 Requirements

```
tensorflow &gt;= 2.15
librosa
soundfile
numpy
pandas
matplotlib
scikit-learn
seaborn
kagglehub
```

Install with:
```bash
pip install tensorflow librosa soundfile pandas matplotlib scikit-learn seaborn kagglehub
```

---

## 📝 Assignment Rubric

| Criteria | Points |
|---|---:|
| Dataset loading & preprocessing | 20 |
| Feature extraction (MFCC, shape 49×40) | 15 |
| SVDF model architecture | 20 |
| Training & evaluation (confusion matrix) | 20 |
| TFLite float16 export & verification | 15 |
| OpenMV deployment script | 10 |
| **Total** | **100** |
