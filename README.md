# 🌿 Text-Based Crop Disease Detection using NLP

This project implements a **text-based crop disease detection system** using **Natural Language Processing (NLP)** techniques.  
Instead of relying on leaf images, the system predicts crop diseases from **farmer-written symptom descriptions**, making it suitable for **rural and low-resource environments**.

---

## 📌 Project Overview

- **Input:** Textual description of crop symptoms  
  (e.g., *"Tomato leaf has yellow patches and olive-green mold underneath"*)
- **Output:** Predicted crop disease with confidence score
- **Model Type:** Sentence Embeddings + Machine Learning Classifier
- **Deployment:** Streamlit Web Application

---

## 🚀 Features

- Text-only disease detection (no images required)
- Robust to spelling mistakes and informal language
- Fast inference using lightweight NLP models
- Offline-capable after initial setup
- Simple and farmer-friendly interface

---

## 🧠 Technologies Used

- **Python**
- **Sentence Transformers (MiniLM / BERT-based embeddings)**
- **Scikit-learn**
- **Streamlit**
- **NumPy, Pandas**
- **Joblib**

---

## 📂 Project Structure

```text
├── app.py                  # Streamlit web application
├── predictor.py            # Prediction logic
├── train.py / train_fast_nlp.py   # Model training script
├── disease_text_dataset.csv # Text-based symptom dataset
├── fast_model/             # Saved model files
│   ├── model.pkl
│   ├── label_encoder.pkl
│   └── embedder_info.pkl
├── README.md               # Project documentation
