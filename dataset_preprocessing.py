import pandas as pd
import random

# Load dataset
df = pd.read_csv("language_dataset.csv")

# Drop unnecessary columns (Keep only 'text' and 'role' if needed)
df = df[['text', 'role']].dropna()

# Shuffle the dataset for randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the preprocessed dataset
df.to_csv("cleaned_language_data.csv", index=False, encoding="utf-8")

print(f"✅ Preprocessed dataset saved as 'cleaned_chatbot_data.csv'")
