import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import lightgbm as lgb
from tqdm import tqdm

# =======================
# 1. Load & Preprocess Data
# =======================
print("📌 Loading dataset...")
dataset_path = "cleaned_oversampled_dataset.csv"
data = pd.read_csv(dataset_path)

# Ensure true_labels are strings and handle NaN values
data['true_labels'] = data['true_labels'].fillna("").astype(str)

# Extract inputs (titles) and true labels
texts = data['title'].astype(str).tolist()
true_labels = data['true_labels'].apply(lambda x: x.split('|')).tolist()  # Convert to list of labels

# ✅ **Ensure labels are at least twice in dataset**
label_counts = pd.Series([label for sublist in true_labels for label in sublist]).value_counts()
valid_labels = label_counts[label_counts >= 2].index  # Keep only labels appearing at least **twice**
filtered_data = [(t, [label for label in labels if label in valid_labels]) for t, labels in zip(texts, true_labels)]

# Remove empty labels after filtering
filtered_data = [(t, labels) for t, labels in filtered_data if labels]

if len(filtered_data) < 50:
    raise ValueError("🚨 Dataset is too small after filtering! Try using more data.")

texts, true_labels = zip(*filtered_data)  # Extract back into separate lists

# Convert multi-label true labels to binary format using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
binarized_labels = mlb.fit_transform(true_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, binarized_labels, test_size=0.2, random_state=42)

print(f"✅ Dataset split: {len(X_train)} train, {len(X_test)} validation samples.")

# =======================
# 2. Convert Texts to SBERT Embeddings
# =======================
print("📌 Generating SBERT embeddings...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient SBERT model

# Convert text to embeddings
X_train_embeddings = np.array([sbert_model.encode(text, convert_to_numpy=True) for text in tqdm(X_train)])
X_test_embeddings = np.array([sbert_model.encode(text, convert_to_numpy=True) for text in tqdm(X_test)])

# Save embeddings for future use
np.save("X_train_sbert.npy", X_train_embeddings)
np.save("X_test_sbert.npy", X_test_embeddings)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ SBERT embeddings generated and saved!")

# =======================
# 3. Train LightGBM Classifier
# =======================
print("📌 Training LightGBM model for multi-label classification...")

multi_output_model = MultiOutputClassifier(
    lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42)
)
multi_output_model.fit(X_train_embeddings, y_train)

# Save model
with open("sbert_lgbm_model.pkl", "wb") as f:
    pickle.dump(multi_output_model, f)

print("✅ LightGBM model trained and saved!")

# =======================
# 4. Make Predictions & Evaluate Model
# =======================
print("📌 Evaluating model...")
y_pred = multi_output_model.predict(X_test_embeddings)

# Decode predictions back to label names
decoded_predictions = mlb.inverse_transform(y_pred)

# Evaluate the model
classification_rep = classification_report(y_test, y_pred, target_names=mlb.classes_)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

print("Classification Report:\n", classification_rep)
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")

# Save metrics to a text file
metrics_file = "sbert_lgbm_metrics.txt"
with open(metrics_file, "w") as file:
    file.write("Classification Report:\n")
    file.write(classification_rep)
    file.write(f"\nAccuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")

print(f"✅ Metrics saved to {metrics_file}")

# =======================
# 5. Make Predictions on New Data
# =======================
def predict_exercise(title):
    print(f"\n🔎 Predicting for: {title}")
    
    # Convert input text to SBERT embedding
    embedding = sbert_model.encode([title], convert_to_numpy=True)
    
    # Load the trained LightGBM model
    with open("sbert_lgbm_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    
    # Predict label
    predicted_labels = loaded_model.predict(embedding)
    
    # Convert back to original multi-label format
    predicted_classes = mlb.inverse_transform(predicted_labels)[0]
    return predicted_classes

# Example Prediction
exercise_title = "Barbell Biceps Curl"
predicted_classes = predict_exercise(exercise_title)
print(f"\n✅ Predicted Classes for '{exercise_title}': {predicted_classes}")
