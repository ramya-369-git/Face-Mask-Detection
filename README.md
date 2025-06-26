# 😷 Face Mask Detection using CNN and OpenCV (Live Camera)

This project implements a real-time **face mask detector** using a **Convolutional Neural Network (CNN)** and **OpenCV**. The model predicts whether a person is **wearing a face mask** or **not** from a live webcam feed.

---

## 📌 Project Overview

With the ongoing importance of public health safety, face mask detection systems have become vital in crowded or indoor environments. This project builds a deep learning model using CNN to classify face images into two categories:

- **Mask**
- **No Mask**

OpenCV is used for real-time face detection and webcam access, while the trained CNN model performs classification.

---

## 🧾 Dataset

- **Name**: Face Mask Detection Dataset
- **Structure**:
  ```
  dataset/
  ├── with_mask/
  ├── without_mask/
  ```

---

## 🧠 Model Architecture (CNN)

- **Input**: 224x224 RGB face images
- **Layers**:
  - Convolutional + MaxPooling layers (ReLU)
  - Flatten + Dense layers
  - Dropout for regularization
  - Output layer: 1 neuron (sigmoid activation)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Accuracy**: ~95% on validation set

The model is saved as `mask_detector_model.keras`.

---

## 📸 Live Detection with OpenCV

The notebook/code uses:
- **Haar Cascade Classifier** for face detection
- **OpenCV** to capture webcam feed
- The CNN model to classify detected faces

```python
# Pseudocode
1. Load trained CNN model
2. Start webcam stream using OpenCV
3. Detect faces using Haar cascades
4. For each face:
   - Preprocess the face image
   - Predict mask/no mask using CNN
   - Display label and bounding box on screen
```

---

## 🖥️ How to Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/mask_pred.git
   cd mask_pred
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the live detection script**:
   ```bash
   python detect_mask_live.py
   ```

   Or run the Jupyter notebook:
   ```bash
   jupyter notebook code.ipynb
   ```

---

## ✅ Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## 📈 Example Output

- Green box = **With Mask**
- Red box = **No Mask**

Live webcam feed displays real-time predictions.

---

## 🛡️ Use Cases

- Office and building entry monitoring
- Public transport systems
- Retail stores and malls
---
