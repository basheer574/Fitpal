import pandas as pd

# ======================= 🔹 Load and Process Dataset 🔹 =======================
print("📌 Loading dataset...")
df = pd.read_csv("language_dataset.csv")  # Load original dataset before cleaning

# Ensure only required columns exist
df = df[["text", "parent_id", "role"]].dropna()

# ======================= 🔹 Generate Response Text 🔹 =======================
print("📌 Generating response_text column...")

# Create a dictionary to map message_id -> text
response_map = df.set_index("parent_id")["text"].to_dict()

# Generate responses by looking up parent_id
df["response_text"] = df["parent_id"].map(response_map)

# If no parent message exists, set a default response
df["response_text"].fillna("I'm not sure how to respond to that.", inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_language_data.csv", index=False)
print("✅ Response text generated and dataset saved.")
