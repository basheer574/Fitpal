import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# =======================
# 1. Load and Preprocess Dataset
# =======================
print("📌 Loading dataset...")
classification_data = pd.read_csv("dataset.csv")

texts = classification_data["title"].tolist()
labels = classification_data["category"].tolist()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save label encoder
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# Stratified split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# =======================
# 2. Extract BERT Embeddings
# =======================
print("📌 Extracting BERT embeddings...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_bert_embeddings(text_list):
    tokenized = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    tokenized = {key: val.to(device) for key, val in tokenized.items()}
    with torch.no_grad():
        outputs = model.bert(**tokenized)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

X_train = get_bert_embeddings(train_texts)
X_val = get_bert_embeddings(val_texts)

# =======================
# 3. Train LightGBM Classifier
# =======================
print("📌 Training LightGBM model...")
lgb_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=10)
lgb_model.fit(X_train, train_labels)

# Save model
with open("lgb_model.pkl", "wb") as lgb_file:
    pickle.dump(lgb_model, lgb_file)

# Evaluate
y_pred = lgb_model.predict(X_val)
accuracy = accuracy_score(val_labels, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average="macro")
classification_rep = classification_report(val_labels, y_pred, target_names=label_encoder.classes_)

print(f"\n✅ LightGBM Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print("\n📌 Classification Report:\n", classification_rep)

with open("lgb_metrics.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep)

print("✅ Metrics saved to lgb_metrics.txt")
