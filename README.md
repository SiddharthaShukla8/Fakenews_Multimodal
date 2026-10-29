# 🛡️ Multimodal Fake News Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BERT-yellow.svg)](https://huggingface.co/bert-base-uncased)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Siddhartha%20Shukla-orange.svg)](https://github.com/SiddharthaShukla8)

> **An end-to-end deep learning system that detects deceptive and misleading online news by jointly learning contextual linguistic patterns from text and visual representations from images.**

---

## 📌 Executive Summary

With the exponential surge in multimedia misinformation on social media platforms, single-modality detectors (text-only or image-only) struggle when text contradicts images or when either modality lacks sufficient standalone signals.

This repository implements a **state-of-the-art dual-stream multimodal neural architecture** designed and developed by **Siddhartha Shukla**. It ingests both news textual claims/headlines and their accompanying visual graphics, projecting them into a joint semantic space to classify content into **Real** or **Fake**.

### 🌟 Key Highlights & Performance
* 🏆 **Test Accuracy**: **78.43%**
* 🎯 **Test F1-Score**: **0.7846** (Precision: `0.7854`, Recall: `0.7843`)
* ⚡ **Peak Validation F1**: **0.8172** achieved at Epoch 8
* 🧠 **Neural Architecture**: `BERT-base-uncased` (768d text embeddings) + `ResNet50` (2048d visual feature vectors)
* 🔄 **Intermediate Projection & Late Fusion**: Fully connected projection blocks + Dropout regularization + Softmax classification
* 💻 **Web Application**: Interactive Flask web interface with real-time probabilistic confidence breakdown

---

## 🏗️ System Architecture

The pipeline processes news headlines and images concurrently through dedicated feature extraction branches before combining them through an intermediate multimodal fusion network:

```
                            ┌────────────────────────┐
   News Headline (Text) ───►│   BERT-base-uncased    │───► [768d] ───► Text FC Layers [256 -> 128d]
                            └────────────────────────┘                                    │
                                                                                          ▼
                                                                                   [Concatenation] ──► [256d]
                                                                                          ▲
                            ┌────────────────────────┐                                    │
   News Image (RGB)     ───►│    ResNet-50 (v2)      │───► [2048d] ──► Image FC Layers [512 -> 128d]
                            └────────────────────────┘
                                                                                          │
                                                                                          ▼
                                                                               Fusion Network [256 -> 128 -> 64d]
                                                                                          │
                                                                                          ▼
                                                                             Softmax Classification Head
                                                                                 (Real vs Fake News)
```

---

## 💻 Tech Stack & Engineering Tools

| Domain | Technology / Library | Role & Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | **PyTorch (>= 2.0)** | Model architecture definition, autograd, forward/backward execution, mixed device dispatch |
| **Natural Language Processing** | **Hugging Face Transformers** | `BertTokenizer` & `BertModel` (`bert-base-uncased`) for 768-dim contextual text embeddings |
| **Computer Vision** | **Torchvision (>= 0.15)** | `ResNet50` pre-trained on ImageNet-1K V2 for robust visual feature extraction |
| **Image Processing** | **Pillow (PIL)** | Reading, validating RGB conversions, image sanitization, and transformations |
| **Data Engineering** | **Pandas, NumPy, kagglehub** | Dataset ingestion, stratified subset sampling, cleaning, tabular persistence |
| **Evaluation Metrics** | **Scikit-Learn** | Stratified train/val/test splits, weighted loss computation, F1, precision, recall, confusion matrix |
| **Web Service & UI** | **Flask, Jinja2, HTML5/CSS3/JS** | Lightweight web server, asynchronous JSON API (`/predict`), dynamic frontend results card |
| **Model Serving / Export** | **ONNX / PyTorch JIT** | Production serialization for cross-platform low-latency inference |
| **Acceleration** | **Apple MPS / NVIDIA CUDA / CPU** | Automatic device selection with Metal Performance Shaders for Mac and CUDA for GPUs |

---

## 📂 Repository Directory Layout

```bash
Fakenews_Multimodal/
├── app.py                      # Production Flask web application & REST endpoints
├── train_model.py              # Full training loop, validation, checkpointing, and evaluation
├── inference.py                # Command-line & programmatic prediction utility
├── dataset_setup.py            # Custom PyTorch Dataset class, data loaders, & class weights
├── preprocess_fakeddit.py      # Automated dataset download, image verification, & cleaning
├── evaluate_model.py           # Formal evaluation benchmark on the held-out test set
├── evaluate_custom.py          # Batch evaluation script for external datasets
├── export_model.py             # Export checkpoint weights to TorchScript / ONNX formats
├── requirements.txt            # Locked Python dependency manifest
├── static/
│   ├── css/style.css           # Modern gradient styling and responsive design
│   └── js/main.js              # Fetch API requests, client-side preview, dynamic feedback
├── templates/
│   └── index.html              # Clean single-page application dashboard
└── processed_data/
    ├── metadata.json           # Ingestion metrics, splits summary, and class distributions
    └── preprocessed_fakeddit.csv # Preprocessed dataset index
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup

> **Note**: Python 3.11 or 3.12 is strongly recommended for seamless PyTorch and Transformers compatibility.

```bash
# Clone the repository
git clone https://github.com/SiddharthaShukla8/Fakenews_Multimodal.git
cd Fakenews_Multimodal

# Initialize a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Dataset Preprocessing

The preprocessing pipeline automatically downloads the benchmark dataset, cleans annotations, retrieves images, and validates integrity:

```bash
python preprocess_fakeddit.py
```

* **Dataset source**: [Fakeddit Multimodal Dataset](https://github.com/entitize/Fakeddit)
* **Sampling**: Stratified balancing across classes
* **Output**: Formatted CSV and cleaned images directory in `processed_data/`

---

### 3. Model Training & Fine-Tuning

You can train the model from scratch or fine-tune upper layers for maximum accuracy:

```bash
# Feature extraction mode (frozen BERT and ResNet backbones)
python train_model.py --epochs 10 --batch_size 16

# Fine-tuning mode (recommended: unfreezes last BERT layer & ResNet layer4)
python train_model.py --fine_tune --epochs 10 --batch_size 16 --lr 1e-4
```

---

### 4. Running the Web Application

Launch the local web server to test predictions interactively:

```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5001
```

---

### 5. CLI Single-Instance Inference

Run predictions directly from your terminal:

```bash
python inference.py \
  --text "NASA confirms discovery of water ice plumes erupting from Europa's ocean." \
  --image sample_news.jpg
```

**Terminal Output Example:**
```yaml
--------------------------------------------------
Prediction Result:
--------------------------------------------------
Verdict: Real News
Confidence: 89.42%
Class Probabilities:
  - Real: 89.42%
  - Fake: 10.58%
--------------------------------------------------
```

---

## 📊 Evaluation & Detailed Metrics

The model is evaluated using comprehensive classification metrics on an unseen test partition:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **78.43%** |
| **Precision** | **78.54%** |
| **Recall** | **78.43%** |
| **F1-Score** | **0.7846** |
| **Best Validation F1** | **0.8172 (Epoch 8)** |

---

## 🔌 RESTful API Reference

### Predict News Authenticity
`POST /predict`

**Request (`multipart/form-data`):**
* `text` (string, required): News headline or text body.
* `image` (file, required): Image file (`.jpg`, `.jpeg`, `.png`, `.webp`).

**Response (`application/json`):**
```json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 85.23,
  "fake_probability": 85.23,
  "real_probability": 14.77
}
```

### Healthcheck
`GET /health`
* Returns status of server and model availability.

---

## 👨‍💻 Author

**Siddhartha Shukla**
* GitHub: [@SiddharthaShukla8](https://github.com/SiddharthaShukla8)
* Repository: [Fakenews_Multimodal](https://github.com/SiddharthaShukla8/Fakenews_Multimodal)

---

## 📜 License

This project is open-source under the MIT License. Please consult the original [Fakeddit Research Paper](https://github.com/entitize/Fakeddit) for dataset licensing and usage guidelines.
