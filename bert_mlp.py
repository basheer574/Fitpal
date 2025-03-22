import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
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
    return torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

X_train = get_bert_embeddings(train_texts)
X_val = get_bert_embeddings(val_texts)

# =======================
# 3. Define MLP Model
# =======================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

mlp_model = MLPClassifier(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# =======================
# 4. Train MLP Model
# =======================
print("📌 Training MLP...")
num_epochs = 10
for epoch in range(num_epochs):
    mlp_model.train()
    optimizer.zero_grad()
    inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
    labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    outputs = mlp_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save trained model
torch.save(mlp_model.state_dict(), "mlp_model.pth")

# =======================
# 5. Evaluate MLP Model
# =======================
mlp_model.eval()
with torch.no_grad():
    inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
    outputs = mlp_model(inputs)
    y_pred = torch.argmax(outputs, axis=1).cpu().numpy()

accuracy = accuracy_score(val_labels, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels, y_pred, average="macro")
classification_rep = classification_report(val_labels, y_pred, target_names=label_encoder.classes_)

print(f"\n✅ MLP Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print("\n📌 Classification Report:\n", classification_rep)

# Save metrics
with open("mlp_metrics.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep)

print("✅ Metrics saved to mlp_metrics.txt")
