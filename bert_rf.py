import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# =======================
# 1. Load and Preprocess Dataset
# =======================
print("📌 Loading dataset...")
# Load the cleaned dataset
classification_data = pd.read_csv("dataset.csv")  

# Extract input text (titles)
texts = classification_data["title"].tolist()

# Extract labels (Single-label)
labels = classification_data["category"].tolist()

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the label encoder for inference
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# Split dataset into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

print(f"✅ Dataset loaded: {len(train_texts)} training samples, {len(val_texts)} validation samples.")

# =======================
# 2. Tokenize Inputs for BERT Fine-Tuning
# =======================
print("📌 Tokenizing inputs...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# =======================
# 3. Create Custom Dataset Class
# =======================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# =======================
# 4. Fine-Tune BERT Model
# =======================
print("📌 Fine-tuning BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save fine-tuned BERT model
model.save_pretrained("fine_tuned_bert_model")
tokenizer.save_pretrained("fine_tuned_bert_model")

print("✅ Fine-tuned BERT model saved successfully.")

# =======================
# 5. Extract BERT Embeddings from Fine-Tuned Model
# =======================
print("📌 Extracting embeddings from fine-tuned BERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_bert_embeddings(text_list):
    tokenized = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    tokenized = {key: val.to(device) for key, val in tokenized.items()}
    
    with torch.no_grad():
        outputs = model.bert(**tokenized)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token representation

# Convert the train and validation sets to embeddings
train_embeddings = get_bert_embeddings(train_texts)
val_embeddings = get_bert_embeddings(val_texts)

# Save embeddings for future use
np.save("train_embeddings.npy", train_embeddings)
np.save("val_embeddings.npy", val_embeddings)
np.save("train_labels.npy", train_labels)
np.save("val_labels.npy", val_labels)

print("✅ Fine-tuned BERT embeddings extracted and saved successfully.")

# =======================
# 6. Train Random Forest on Fine-Tuned BERT Embeddings
# =======================
print("📌 Training Random Forest classifier...")
X_train = np.load("train_embeddings.npy")
X_val = np.load("val_embeddings.npy")
y_train = np.load("train_labels.npy")
y_val = np.load("val_labels.npy")

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained Random Forest model
with open("rf_model.pkl", "wb") as rf_file:
    pickle.dump(rf_model, rf_file)

# Make predictions
y_pred = rf_model.predict(X_val)

# =======================
# 7. Evaluate Model
# =======================
print("📌 Evaluating model...")
accuracy = accuracy_score(y_val, y_pred)

# Compute precision, recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro")

# Compute classification report
# Get unique labels in y_val
unique_labels = np.unique(y_val)

# Compute classification report only for labels in y_val
classification_rep = classification_report(
    y_val, y_pred, labels=unique_labels, target_names=[label_encoder.classes_[i] for i in unique_labels]
)

# Print results
print(f"\n📌 Random Forest Accuracy: {accuracy:.4f}")
print(f"📌 Precision: {precision:.4f}")
print(f"📌 Recall: {recall:.4f}")
print(f"📌 F1-score: {f1:.4f}")
print("\n📌 Classification Report:\n", classification_rep)

# Save results to a text file
with open("rf_metrics.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep)

print("✅ Metrics saved to rf_metrics.txt")
