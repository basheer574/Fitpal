import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# =======================
# 1. Load and Preprocess Dataset
# =======================
print("📌 Loading dataset...")
classification_data = pd.read_csv("dataset.csv")  

# Extract input text (titles)
texts = classification_data["title"].tolist()

# Extract labels (Single-label)
labels = classification_data["category"].tolist()

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save label encoder for future use
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# =======================
# 2. Fix Test Size for Stratification
# =======================
num_classes = len(np.unique(encoded_labels))
min_test_size = max(num_classes, int(0.2 * len(encoded_labels)))  # Ensure test set is large enough

# Ensure test size is within valid limits
test_size = min(min_test_size, len(encoded_labels) - num_classes)

# Split dataset into train and validation **ensuring stratification to keep all classes**
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
)

print(f"✅ Dataset loaded: {len(train_texts)} training samples, {len(val_texts)} validation samples.")

# =======================
# 3. Tokenize Inputs for BERT
# =======================
print("📌 Tokenizing inputs...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# =======================
# 4. Create Custom Dataset Class
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
# 5. Fine-Tune BERT Model
# =======================
print("📌 Fine-tuning BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save fine-tuned BERT model
model.save_pretrained("fine_tuned_bert_model")
tokenizer.save_pretrained("fine_tuned_bert_model")

print("✅ Fine-tuned BERT model saved successfully.")

# =======================
# 6. Extract BERT Embeddings
# =======================
print("📌 Extracting embeddings from fine-tuned BERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_bert_embeddings(text_list):
    tokenized = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    tokenized = {key: val.to(device) for key, val in tokenized.items()}
    
    with torch.no_grad():
        outputs = model.bert(**tokenized)

    return torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()


train_embeddings = get_bert_embeddings(train_texts)
val_embeddings = get_bert_embeddings(val_texts)

np.save("train_embeddings.npy", train_embeddings)
np.save("val_embeddings.npy", val_embeddings)
np.save("train_labels.npy", train_labels)
np.save("val_labels.npy", val_labels)

print("✅ BERT embeddings extracted.")

# =======================
# 7. Train XGBoost with Fixed Class Labels
# =======================
print("📌 Training XGBoost...")

X_train = np.load("train_embeddings.npy")
X_val = np.load("val_embeddings.npy")
y_train = np.load("train_labels.npy")
y_val = np.load("val_labels.npy")

# Ensure labels are complete and sequential
unique_classes = np.unique(encoded_labels)  # Get full list of classes from the original dataset
class_mapping = {old: new for new, old in enumerate(unique_classes)}  # Ensure continuity
y_train_fixed = np.array([class_mapping[label] for label in y_train])
y_val_fixed = np.array([class_mapping[label] for label in y_val])

# Save class mapping for later use
with open("class_mapping.pkl", "wb") as cm_file:
    pickle.dump(class_mapping, cm_file)

print("✅ Labels re-mapped to continuous range (0,1,2,...).")

# Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=500,         # Increase trees (default=100)
    learning_rate=0.05,       # Reduce learning rate (default=0.3)
    max_depth=10,             # Increase tree depth
    subsample=0.8,            # Use 80% of data per tree
    colsample_bytree=0.8,     # Use 80% of features per tree
    reg_lambda=1.5,           # L2 Regularization
    reg_alpha=0.5,            # L1 Regularization
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train_fixed)

with open("xgb_model.pkl", "wb") as xgb_file:
    pickle.dump(xgb_model, xgb_file)

# =======================
# 8. Evaluate XGBoost Model
# =======================
print("📌 Evaluating model...")

# Predict labels
y_pred_fixed = xgb_model.predict(X_val)

# Convert predictions back to original class labels
reverse_mapping = {v: k for k, v in class_mapping.items()}
y_pred_original = np.array([reverse_mapping[label] for label in y_pred_fixed])

# Compute metrics
accuracy = accuracy_score(y_val, y_pred_original)
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_original, average="macro")

# Classification Report (Use only labels present in val_labels)
unique_labels = np.unique(y_val)

classification_rep = classification_report(
    y_val, y_pred_original, labels=unique_labels,
    target_names=[label_encoder.classes_[i] for i in unique_labels]
)

# Print results
print(f"\n✅ XGBoost Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")

print("\n📌 Classification Report:\n", classification_rep)

# Save results to file
with open("xgb_metrics.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep)

print("✅ Metrics saved to xgb_metrics.txt")
