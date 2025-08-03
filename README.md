# 🧠 Deepfake Image Detection using Deep Learning

This project aims to detect deepfake images using advanced deep learning architectures. Deepfakes pose a serious threat in the era of misinformation, and this model attempts to counter such challenges with robust and interpretable AI.

## 📌 Project Overview

The system is designed to classify images as **real** or **fake** using convolutional neural networks. We trained and evaluated multiple state-of-the-art models on benchmark datasets to build a high-performing detection pipeline. The models are enhanced with attention mechanisms and explainability techniques for better reliability.

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **OpenCV**
- **Grad-CAM (for explainability)**

## 🧪 Models Trained

- **ResNet**
- **DenseNet**
- **SFFNet**
- **AMTENet**
- **MCNet**

### ✅ Enhancements

- **CBAM (Convolutional Block Attention Module)** integrated for improved feature focus
- **Grad-CAM** used for visualizing attention areas in the classification

## 🔍 Datasets

- **140K Deepfake Dataset**
- **HFM (High-Fidelity Manipulation) Dataset**

These datasets include a diverse set of fake and real images, manipulated using state-of-the-art generation techniques.

## 💡 Research Contribution

We also explored **One-Class Classification (OCC)** methods:
- Built models using **Autoencoder** and **Support Vector Data Description (SVDD)**
- Fine-tuned on 20% of fake data for anomaly detection, though classification models outperformed in accuracy and precision.

## 📈 Results & Evaluation

Each model was evaluated using:
- Accuracy
- Precision / Recall / F1-score
- AUC-ROC
- GradCAM visualizations for model explainability

## 🔔 Future Work

- Integrate real-time video detection pipeline
- Extend detection to Deepfake audio/video using multimodal inputs
- Deploy the best model via Flask/FastAPI for live detection

## 📎 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/sanketdhonde/deepfake-image-detection.git
   cd deepfake-image-detection

## Download all the Requirements

-pip install -r requirements.txt
