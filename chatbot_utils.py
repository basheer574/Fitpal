import torch
import pickle
import numpy as np
import gc
import random
from transformers import BertTokenizer, BertModel
from langdetect import detect

# Load BERT & Decision Tree Model
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
with open("decision_tree_model.pkl", "rb") as dt_file:
    dt_model = pickle.load(dt_file)
with open("language_label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def get_bert_embeddings(text_list):
    all_embeddings = []
    for i in range(0, len(text_list), 16):
        batch_texts = text_list[i : i + 16]
        tokenized = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=64)
        tokenized = {key: val.to(device) for key, val in tokenized.items()}

        with torch.no_grad():
            outputs = bert_model(**tokenized)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(batch_embeddings)

        del tokenized, outputs
        torch.cuda.empty_cache()
        gc.collect()

    return np.vstack(all_embeddings)

def classify_text(text):
    embedding = get_bert_embeddings([text])
    predicted_label = dt_model.predict(embedding)
    return label_encoder.inverse_transform(predicted_label)[0]

def generate_response(user_input, category):
    try:
        lang = detect(user_input)
    except:
        lang = "en"

    greetings = {
        "en": ["I'm fine, thanks! How about you?", "Doing great! How can I help?", "I'm good! What's on your mind?"],
        "ar": ["أنا بخير، ماذا عنك؟", "أنا بخير، شكراً لسؤالك!", "جيد جدًا! كيف حالك؟"]
    }
    greeting_keywords = {"en": ["hello", "hi", "how are you"], "ar": ["مرحبا", "اهلا", "كيف حالك"]}

    for keyword in greeting_keywords.get(lang, []):
        if keyword in user_input.lower():
            return random.choice(greetings[lang])

    responses = {
        "assistant": {"en": ["I'm here to assist you!", "How can I help today?"], "ar": ["كيف يمكنني مساعدتك؟"]},
        "prompter": {"en": ["Interesting! Tell me more."], "ar": ["أخبرني المزيد."]},
        "default": {"en": ["I didn't understand."], "ar": ["لم أفهم."]}
    }

    return random.choice(responses.get(category, responses["default"]).get(lang, responses["default"]["en"]))
