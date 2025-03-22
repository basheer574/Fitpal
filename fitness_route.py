from flask import Blueprint, request, jsonify
from database import session, User, Interaction
from fitness_utils import generate_exercise_plan, generate_pdf
import pickle
import numpy as np
import pandas as pd
import torch
import re
import os
from transformers import BertTokenizer, BertForSequenceClassification

fitness_bp = Blueprint("fitness", __name__)

# ✅ Load Fitness Dataset
fitness_data = pd.read_csv("dataset.csv")

# ✅ Load BERT tokenizer & model
MODEL_PATH = "fine_tuned_bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# ✅ Load trained XGB model & label encoder
try:
    with open("xgb_model.pkl", "rb") as xgb_model:
        xgb_model = pickle.load(xgb_model)

    with open("label_encoder.pkl", "rb") as le_file:
        label_encoder = pickle.load(le_file)

    print("✅ XGB Model & Label Encoder Loaded Successfully!")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    xgb_model, label_encoder = None, None  # Prevent crashes if model loading fails


# ============== 🔹 Category Classification 🔹 ==============
def classify_fitness_query(user_input):
    """
    Classifies the user query into categories using keywords.
    Uses keyword detection in both English and Arabic.
    """
    user_input = user_input.lower()

    category_keywords = {
        "weight_loss": [
            "lose weight", "fat loss", "burn fat", "slim down", "weight reduction",
            "انحف", "فقدان الوزن", "حرق الدهون", "اخسر الوزن"
        ],
        "muscle_gain": [
            "gain muscle", "build muscle", "strength training", "increase muscle",
            "بناء عضلات", "زيادة العضلات", "تمارين القوة", "تضخيم العضلات"
        ],
        "endurance_training": [
            "increase stamina", "endurance workout", "cardio training", "improve stamina",
            "تمارين التحمل", "تمارين القلب", "زيادة التحمل", "تمارين الجري"
        ],
        "strength_training": [
            "heavy lifting", "powerlifting", "strength training plan",
            "تدريب القوة", "رفع الأثقال", "تمارين القوة البدنية"
        ],
        "flexibility_mobility": [
            "improve flexibility", "mobility exercises", "yoga routine", "stretching exercises",
            "تمارين التمدد", "اليوغا", "تحسين المرونة", "تمارين الحركة"
        ],
        "athletic_performance": [
            "improve speed", "athletic drills", "sports training",
            "تمارين السرعة", "تمارين رياضية", "التدريب الرياضي"
        ]
    }

    for category, keywords in category_keywords.items():
        if any(re.search(rf"\b{kw}\b", user_input) for kw in keywords):
            return category  # ✅ Return detected category

    return "general_fitness"  # ✅ Default category


# ============== 🔹 Prediction Route 🔹 ==============

@fitness_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"📥 Received data: {data}")  # Debug input

        user_input = data.get("user_input", "").strip()
        user_id = data.get("user_id")
        category = data.get("category")  # ✅ Get the category from the request

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # ✅ Use the category if provided, else classify the input
        if not category:
            category = classify_fitness_query(user_input)

        print(f"✅ Predicted Category: {category}")

        # ✅ Step 2: Generate Fitness Plan
        exercise_plan = generate_exercise_plan([category])
        print(f"📑 DEBUG: Exercise Plan: {exercise_plan}")  # Ensure it's a list

        # ✅ Step 3: Generate Category-Specific PDF
        pdf_path = generate_pdf(user_id, exercise_plan, category)
        print(f"📄 DEBUG: PDF Path: {pdf_path}")

        # ✅ Step 4: Store Response in Database
        new_interaction = Interaction(
            user_id=user_id,
            input_text=user_input,
            response_text="Here is your recommended fitness plan.",
            category=category,  # ✅ Store category in DB
            pdf_url=pdf_path,
        )
        session.add(new_interaction)
        session.commit()

        return jsonify({
            "message": "Here is your recommended fitness plan.",
            "category": category,
            "pdf_url": pdf_path if pdf_path else None
        })

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500




# ============== 🔹 Chat History Route 🔹 ==============
@fitness_bp.route("/history/<int:user_id>", methods=["GET"])
def get_fitness_chat_history(user_id):
    """
    Fetch fitness-related chat history for a user.
    """
    try:
        interactions = session.query(Interaction).filter_by(user_id=user_id).all()
        
        if not interactions:
            return jsonify([])  # Return an empty list instead of 404

        history = []
        for interaction in interactions:
            # ✅ Ensure category is present
            category = interaction.category if interaction.category else "general_fitness"

            # ✅ Fix PDF URL
            pdf_url = None
            if interaction.pdf_url:
                pdf_filename = os.path.basename(interaction.pdf_url)  # Extract file name
                pdf_url = f"http://10.0.2.2:5000/static/{pdf_filename}"  # Correct static file path

            # ✅ User's input message
            history.append({
                "user_id": interaction.user_id,
                "text": interaction.input_text,
                "sender": "user",
                "category": category,
                "created_at": interaction.created_at.timestamp(),
                "pdf_url": None  # User messages don't have PDFs
            })

            # ✅ Bot's response message with category-specific PDF
            response_entry = {
                "user_id": interaction.user_id,
                "text": interaction.response_text,
                "sender": "bot",
                "category": category,
                "created_at": interaction.created_at.timestamp(),
                "pdf_url": pdf_url  # Ensure full URL is returned
            }

            history.append(response_entry)

        return jsonify(history)

    except Exception as e:
        print(f"❌ ERROR in get_fitness_chat_history: {str(e)}")  # Debugging error log
        return jsonify({"error": str(e)}), 500