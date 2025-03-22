from flask import Flask, request, jsonify, send_from_directory
import torch
import pickle
import numpy as np
from fitness_utils import generate_pdf
import os
import pandas as pd
import random
from transformers import BertTokenizer, BertForSequenceClassification
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database import Base, User, Interaction
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ============== 🔹 Load Models and Data 🔹 ==============
# Load fitness dataset
fitness_data = pd.read_csv("dataset.csv")

# Ensure "description" column exists
if "description" not in fitness_data.columns:
    fitness_data["description"] = "No description available."

# Load BERT tokenizer & model (for embeddings only)
MODEL_PATH = "fine_tuned_bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load trained xgboost model & label encoder
with open("xgb_model.pkl", "rb") as xgb_file:
    xgb_model = pickle.load(xgb_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Flask app initialization
app = Flask(__name__)

# Ensure the static directory exists
os.makedirs("static", exist_ok=True)

# Database setup
engine = create_engine("sqlite:///fitpal.db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

# Move model to the appropriate device
device = torch.device("cpu")
model.to(device)


# ============== 🔹 Utility Functions 🔹 ==============
def get_bert_embeddings(text_list):
    """
    Extracts CLS token embeddings from BERT.
    """
    tokenized = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    tokenized = {key: val.to(device) for key, val in tokenized.items()}

    with torch.no_grad():
        outputs = model.bert(**tokenized)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


# ============== 🔹 Flask Routes 🔹 ==============
@app.route("/")
def home():
    return jsonify({"message": "FitPal API is running!"})

