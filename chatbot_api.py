from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import random
from langdetect import detect
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database import Base, Interaction, User

# =================== 🔹 Initialize Flask App 🔹 ===================
app = Flask(__name__)

# =================== 🔹 Load Fine-Tuned BERT Model 🔹 ===================
print("📌 Loading BERT chatbot model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = BertForSequenceClassification.from_pretrained("bert_chatbot_model")
tokenizer = BertTokenizer.from_pretrained("bert_chatbot_model")
bert_model.to(device)

# Load Label Encoder
with open("bert_label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

print("✅ Chatbot Model Loaded Successfully!")

# =================== 🔹 Load Database 🔹 ===================
engine = create_engine("sqlite:///fitpal.db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


# =================== 🔹 Helper Functions 🔹 ===================
def get_bert_prediction(text):
    """
    Predict response category using fine-tuned BERT.
    """
    encoding = tokenizer(
        text, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    predicted_label = outputs.logits.argmax(dim=1).item()
    predicted_class = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_class


def generate_response(user_input, category):
    """
    Generate a response based on detected language and category.
    """
    try:
        # Detect input language (fallback to English if uncertain)
        lang = detect(user_input)
    except:
        lang = "en"  # Default to English if language detection fails

    # Hardcoded greeting responses
    greetings = {
        "en": ["I'm fine, thanks! How about you?", "Doing great! How can I help?", "I'm good! What's on your mind?"],
        "ar": ["أنا بخير، ماذا عنك؟", "أنا بخير، شكراً لسؤالك!", "جيد جدًا! كيف حالك؟"]
    }

    # Common greeting keywords in Arabic & English
    greeting_keywords = {
        "en": ["hello", "hi", "how are you", "good morning", "good evening"],
        "ar": ["مرحبا", "اهلا", "كيف حالك", "صباح الخير", "مساء الخير"]
    }

    # Check if user input contains a greeting
    for keyword in greeting_keywords.get(lang, []):
        if keyword in user_input.lower():
            return random.choice(greetings[lang])  # Return a random greeting response

    # Default response templates based on category
    responses = {
        "assistant": {
            "en": ["I'm here to assist you! How can I help today?", "Hello! What do you need assistance with?", "I'm happy to help! Ask me anything."],
            "ar": ["أنا هنا لمساعدتك، كيف يمكنني المساعدة اليوم؟", "مرحبًا! كيف يمكنني مساعدتك؟", "أنا سعيد بمساعدتك، اسألني عن أي شيء!"]
        },
        "prompter": {
            "en": ["Interesting thought! Tell me more.", "That’s a great question! What do you think?", "I'm curious to hear more about this!"],
            "ar": ["فكرة مثيرة للاهتمام! أخبرني المزيد.", "هذا سؤال رائع! ماذا تعتقد؟", "أنا مهتم بمعرفة المزيد عن هذا!"]
        },
        "default": {
            "en": ["I'm not sure how to respond to that.", "Could you clarify what you mean?", "I didn't understand. Can you rephrase?"],
            "ar": ["لست متأكدًا من كيفية الرد على ذلك.", "هل يمكنك توضيح ما تعنيه؟", "لم أفهم، هل يمكنك إعادة الصياغة؟"]
        }
    }

    # Select appropriate response
    response_category = responses.get(category, responses["default"])
    response_list = response_category.get(lang, response_category["en"])  # Default to English if language is unsupported
    return random.choice(response_list)



# =================== 🔹 API Endpoints 🔹 ===================
@app.route("/")
def home():
    return jsonify({"message": "Chatbot API is running!"})


@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Handle user input, classify text using BERT, and return a chatbot response.
    """
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        user_id = data.get("user_id")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # Predict the category
        category = get_bert_prediction(user_input)

        # Generate a response
        response = generate_response(user_input, category)

        # Store conversation history if user_id exists
        if user_id:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                new_interaction = Interaction(user_id=user_id, input_text=user_input, response_text=response)
                session.add(new_interaction)
                session.commit()

        return jsonify({
            "message": response,
            "category": category
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import Blueprint, request, jsonify
from chatbot_utils import classify_text, generate_response
from database import session, User, Interaction

chatbot_blueprint = Blueprint("chatbot", __name__)

@chatbot_blueprint.route("/", methods=["POST"])
def chatbot():
    """
    Handle user input, classify text, and return chatbot response.
    """
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        user_id = data.get("user_id")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # Classify text
        category = classify_text(user_input)

        # Generate response
        response = generate_response(user_input, category)

        # Store conversation history
        if user_id:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                new_interaction = Interaction(user_id=user_id, input_text=user_input, response_text=response)
                session.add(new_interaction)
                session.commit()

        return jsonify({"message": response, "category": category})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
