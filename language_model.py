import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle

# ======================= 🔹 Load Dataset 🔹 =======================
print("📌 Loading dataset...")
df = pd.read_csv("cleaned_language_data.csv")  # The dataset with text, role, parent_id

# Drop rows with missing values
df = df.dropna(subset=["text", "role"])

# Encode role labels (e.g., "assistant" → 0, "prompter" → 1)
label_encoder = LabelEncoder()
df["role"] = label_encoder.fit_transform(df["role"])

# Save Label Encoder for later use
with open("bert_label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# ======================= 🔹 Train-Test Split 🔹 =======================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["role"].tolist(), test_size=0.2, random_state=42
)

print(f"✅ Dataset loaded: {len(train_texts)} training samples, {len(val_texts)} validation samples.")

# ======================= 🔹 Load BERT Tokenizer 🔹 =======================
print("📌 Initializing BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # English + Arabic Support

# Define PyTorch Dataset Class
class ChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Create DataLoader
train_dataset = ChatDataset(train_texts, train_labels, tokenizer)
val_dataset = ChatDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("✅ Data prepared for BERT training.")

from transformers import AdamW
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim

# ======================= 🔹 Load BERT Model 🔹 =======================
print("📌 Initializing BERT model for training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
bert_model.to(device)

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(bert_model.parameters(), lr=5e-5)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

from tqdm import tqdm

# ======================= 🔹 Training Loop 🔹 =======================
epochs = 3  # Adjust based on dataset size
print("📌 Starting BERT training...")

for epoch in range(epochs):
    bert_model.train()
    total_loss = 0
    correct_predictions = 0

    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    accuracy = correct_predictions / len(train_texts)
    print(f"✅ Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")

    scheduler.step()

print("✅ BERT training complete.")

# ======================= 🔹 Save Fine-Tuned Model 🔹 =======================
bert_model.save_pretrained("bert_chatbot_model")
tokenizer.save_pretrained("bert_chatbot_model")

print("✅ Fine-tuned BERT model saved.")
