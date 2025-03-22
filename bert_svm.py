import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE  # Import SMOTE
from tqdm import tqdm

# ============================
# 1️⃣ Load and Preprocess Dataset
# ============================
print("📌 Loading dataset...")
classification_data = pd.read_csv("dataset.csv")

# Extract text (exercise names)
texts = classification_data["title"].astype(str).tolist()

# Extract new labels (categories)
labels = classification_data["category"].astype(str).tolist()

# Encode labels numerically
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the label encoder for API use
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# Split dataset into training and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42  # Fix seed
)

print(f"✅ Dataset split: {len(train_texts)} train, {len(val_texts)} validation samples.")

# ============================
# 2️⃣ Tokenization for BERT
# ============================
print("📌 Tokenizing inputs...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Convert to Torch tensors
train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels))
val_dataset = TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels))

# ============================
# 3️⃣ Define Training Arguments
# ============================
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
torch.manual_seed(42)  # Fix seed for reproducibility
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# ============================
# 4️⃣ Fine-Tune BERT
# ============================
print("📌 Fine-tuning BERT...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"🔄 Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_dataloader):.4f}")

# Save fine-tuned model
model.save_pretrained("fine_tuned_bert_model")

# ============================
# 5️⃣ Extract BERT Embeddings
# ============================
print("📌 Extracting BERT embeddings...")

def get_bert_embeddings(text_list):
    tokenized = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    tokenized = {key: val.to(device) for key, val in tokenized.items()}

    with torch.no_grad():
        outputs = model.bert(**tokenized)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

train_embeddings = get_bert_embeddings(train_texts)
val_embeddings = get_bert_embeddings(val_texts)

# ============================
# 6️⃣ Apply SMOTE to Balance Classes
# ============================
print("📌 Applying SMOTE to balance dataset...")

# Find the smallest class size
min_class_size = min(pd.Series(train_labels).value_counts())

# Adjust k_neighbors dynamically based on the smallest class size
smote_k = min(1, min_class_size - 1)  # Ensure SMOTE doesn't fail

smote = SMOTE(sampling_strategy="auto", k_neighbors=smote_k, random_state=42)

# Only apply SMOTE if enough samples exist
if min_class_size > 1:
    train_embeddings_resampled, train_labels_resampled = smote.fit_resample(train_embeddings, train_labels)
    print(f"✅ Resampled dataset: {len(train_embeddings_resampled)} samples")
else:
    train_embeddings_resampled, train_labels_resampled = train_embeddings, train_labels
    print("⚠️ SMOTE skipped due to insufficient samples.")

# Save new embeddings
np.save("train_embeddings.npy", train_embeddings_resampled)
np.save("train_labels.npy", train_labels_resampled)

# ============================
# 7️⃣ Train SVM Model
# ============================
print("📌 Training SVM classifier...")

svm_model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
svm_model.fit(train_embeddings_resampled, train_labels_resampled)

# Save trained SVM model
with open("svm_model.pkl", "wb") as svm_file:
    pickle.dump(svm_model, svm_file)

print("✅ SVM model retrained and saved!")

# ============================
# 8️⃣ Evaluate the Model
# ============================
print("📌 Evaluating model...")

y_pred = svm_model.predict(val_embeddings)

accuracy = accuracy_score(val_labels, y_pred)
f1score = f1_score(val_labels, y_pred, average="macro")
recallscore = recall_score(val_labels, y_pred, average="macro")
precisionscore = precision_score(val_labels, y_pred, average="macro")

classification_rep = classification_report(
    val_labels, y_pred, labels=np.unique(val_labels), target_names=[label_encoder.classes_[i] for i in np.unique(val_labels)]
)

# Print Metrics
print(f"\n✅ SVM Accuracy: {accuracy:.4f}")
print(f"🔹 Precision: {precisionscore:.4f}")
print(f"🔹 Recall: {recallscore:.4f}")
print(f"🔹 F1-score: {f1score:.4f}")
print("\n📌 Classification Report:\n", classification_rep)

# ============================
# 9️⃣ Save Metrics
# ============================
with open("svm_metrics.txt", "w") as file:
    file.write(f"🔹 Accuracy: {accuracy:.4f}\n")
    file.write(f"🔹 Precision: {precisionscore:.4f}\n")
    file.write(f"🔹 Recall: {recallscore:.4f}\n")
    file.write(f"🔹 F1-score: {f1score:.4f}\n\n")
    file.write("📌 Classification Report:\n")
    file.write(classification_rep)

print("✅ Metrics saved to svm_metrics.txt")
print(classification_data["category"].value_counts())
