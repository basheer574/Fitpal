# FitPal: Personalized Nutrition and Fitness Chatbot

[![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-v3.0.0-black.svg)](https://flask.palletsprojects.com/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Project Overview
FitPal is an advanced generative AI-powered chatbot designed to offer personalized fitness and nutrition advice by integrating NLP techniques and machine learning algorithms (BERT combined with classifiers such as XGBoost, Random Forest, SVM, etc.).

---

## 🚀 Features
- Interactive conversational chatbot using **Flask**.
- Highly accurate text classification using **BERT embeddings**.
- Dynamic recommendation system for diet and workouts.
- Real-time user profile updates for personalized recommendations.
- Ensured data privacy and ethical compliance (Federated Learning, Differential Privacy).

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **PyTorch**
- **Flask**
- **Transformers (BERT Tokenization)**
- **XGBoost, Random Forest, SVM, LightGBM, KNN, Naïve Bayes**

---

## 📦 Dataset Information
The chatbot utilizes two primary datasets fetched from Kaggle:

- **[Gym Exercises Dataset] https://www.kaggle.com/datasets/niharika41298/gym-exercise-data

---

## 💻 Installation and Setup
Follow these steps to replicate the environment:

### Clone the repository:
```bash
git clone https://github.com/yourusername/FitPal.git
cd FitPal

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt

python app.py

### Model Training and Evaluation
python train.py bert_xgboost.py

📂 Project Structure

FitPal/
├── app.py
├── model.py
├── fitpal.db
├── requirements.txt
├── data/
├── models/
├── static/
├── LICENSE
└── README.md

