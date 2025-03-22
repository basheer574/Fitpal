import os
import pandas as pd
import random
import torch
import pickle
from flask import jsonify
from database import session, Interaction, User
from transformers import BertTokenizer, BertForSequenceClassification
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ============ 🔹 Load Models and Data 🔹 ============
# Load fitness dataset
fitness_data = pd.read_csv("dataset.csv")

# Ensure "description" column exists
if "description" not in fitness_data.columns:
    fitness_data["description"] = "No description available."

# ✅ Load fine-tuned BERT tokenizer & model
MODEL_PATH = "fine_tuned_bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cpu")  # Set to CPU to avoid GPU OOM errors
model.to(device)

# ✅ Load trained XGB model & label encoder
with open("xgb_model.pkl", "rb") as xgb_model:
    xgb_model = pickle.load(xgb_model)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# ============ 🔹 Utility Functions 🔹 ============
def get_user_or_error(user_id):
    """
    Fetch user from database or return an error response.
    """
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        return None, jsonify({"error": "User not found"}), 404
    return user, None


def get_bert_embeddings(text_list):
    """
    Extracts CLS token embeddings from fine-tuned BERT.
    """
    tokenized = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    tokenized = {key: val.to(device) for key, val in tokenized.items()}

    with torch.no_grad():
        outputs = model.bert(**tokenized)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def classify_fitness_query(user_input):
    """
    Predicts the category of a user input using BERT embeddings and XGB.
    """
    embedding = get_bert_embeddings([user_input])
    predicted_label = xgb_model.predict(embedding)
    predicted_class = label_encoder.inverse_transform(predicted_label)[0]

    return predicted_class


# ============ 🔹 Generate Exercise Plan 🔹 ============
def generate_exercise_plan(categories):
    """
    Generate a structured 3-day exercise plan based on the new categories.
    """
    plan = []
    used_exercises = set()

    for day in range(1, 4):  # Generate exercises for 3 days
        filtered_exercises = fitness_data[fitness_data['category'].isin(categories)]

        # If no matching exercises found, return random ones
        if filtered_exercises.empty:
            daily_exercises = fitness_data.sample(n=6).to_dict(orient="records")
        else:
            available_exercises = filtered_exercises[~filtered_exercises['title'].isin(used_exercises)]
            
            if len(available_exercises) < 6:
                extra_needed = 6 - len(available_exercises)
                extra_exercises = fitness_data[~fitness_data['title'].isin(used_exercises)].sample(
                    n=min(extra_needed, len(fitness_data)), random_state=day
                ).to_dict(orient="records")
                daily_exercises = available_exercises.to_dict(orient="records") + extra_exercises
            else:
                daily_exercises = available_exercises.sample(n=6, random_state=day).to_dict(orient="records")

        daily_exercises = list({ex["title"]: ex for ex in daily_exercises}.values())[:6]
        used_exercises.update(ex["title"] for ex in daily_exercises)

        plan.append({
            "day": f"Day {day}",
            "exercises": daily_exercises
        })

    return plan  # ✅ Returns a structured **list**


# ============ 🔹 Generate PDF 🔹 ============
def generate_pdf(user_id, exercise_plan, category):
    """
    Generate a PDF file containing a structured 3-day fitness plan, ensuring each category has its own PDF.
    """
    user = session.query(User).filter_by(id=user_id).first()
    username = user.name if user else "Unknown"

    # ✅ Ensure category-specific PDF filename
    pdf_filename = f"static/user_{user_id}_{category}_plan.pdf"

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Load exercise descriptions from dataset
    exercise_data = pd.read_csv("dataset.csv")
    exercise_dict = exercise_data.drop_duplicates(subset=["title"]).set_index("title")[["description", "image_file"]].to_dict(orient="index")

    # ✅ Add Title
    elements.append(Paragraph(f"{username}'s 3-Day {category.replace('_', ' ').title()} Plan", styles["Title"]))
    elements.append(Spacer(1, 12))

    # ✅ Add Exercises with Images
    for day in exercise_plan:
        elements.append(Paragraph(day["day"], styles["Heading2"]))
        elements.append(Spacer(1, 8))

        for idx, exercise in enumerate(day["exercises"], start=1):
            title = exercise["title"]
            clean_title = title.split(": ", 1)[-1]  # Extract actual title
            details = exercise_dict.get(clean_title, {"description": "No description available.", "image_file": ""})
            description = details.get("description", "No description available.")
            image_filename = details.get("image_file", "")

            elements.append(Paragraph(f"Exercise {idx}: {clean_title}", styles["Heading3"]))
            elements.append(Paragraph(f"Description: {description}", styles["BodyText"]))

            # ✅ Check if the image exists before adding it
            if image_filename and os.path.exists(f"fitness_images/{image_filename}"):
                image_path = f"fitness_images/{image_filename}"
                elements.append(Image(image_path, width=200, height=150))
            else:
                elements.append(Paragraph("❌ Image not available", styles["BodyText"]))

            elements.append(Spacer(1, 8))

        elements.append(Spacer(1, 12))

    doc.build(elements)
    
    print(f"✅ PDF generated: {pdf_filename}")  # Debugging
    
    return pdf_filename  # ✅ Ensure correct file path is returned

