import pandas as pd
import numpy as np
from collections import Counter

# -----------------------------
# 1. Load the Dataset
# -----------------------------
df = pd.read_csv("final_dataset.csv")

# Ensure that the 'true_labels' column is filled and treated as a string.
df['true_labels'] = df['true_labels'].fillna("")

# Convert the true_labels column into a list of labels.
# For each row, if there are labels (non-empty string), split them using "|" as the delimiter.
df['labels_list'] = df['true_labels'].apply(lambda x: x.split('|') if x.strip() != "" else [])

# -----------------------------
# 2. Calculate Label Frequencies
# -----------------------------
# Flatten the list of labels for all samples to compute the frequency of each label.
all_labels = [label for labels in df['labels_list'] for label in labels if label]
label_counts = Counter(all_labels)
print("Original label distribution:")
print(label_counts)

# Find the maximum frequency (this is our target count for oversampling).
max_count = max(label_counts.values())

# -----------------------------
# 3. Compute Oversampling Factors
# -----------------------------
# Define a function that computes an oversampling factor for a sample.
# The factor is based on how underrepresented the sample's most rare label is.
def compute_oversample_factor(labels):
    # If no labels are provided, you can choose to return 1 (no oversampling) or handle separately.
    if not labels:
        return 1
    # For each label, compute how many times more it should appear to match the max_count.
    factors = [max_count / label_counts[label] for label in labels if label in label_counts]
    # Return the ceiling of the maximum factor, ensuring at least 1 repetition.
    return int(np.ceil(max(factors)))

# Apply the oversampling factor computation to each sample.
df['oversample_factor'] = df['labels_list'].apply(compute_oversample_factor)

# -----------------------------
# 4. Create the Oversampled Dataset
# -----------------------------
# Repeat each row in the DataFrame according to its oversample_factor.
oversampled_df = df.loc[df.index.repeat(df['oversample_factor'])].reset_index(drop=True)

print("Original dataset size:", len(df))
print("Oversampled dataset size:", len(oversampled_df))

# Optionally, check the label distribution after oversampling.
oversampled_labels = [label for labels in oversampled_df['labels_list'] for label in labels if label]
oversampled_label_counts = Counter(oversampled_labels)
print("Oversampled label distribution:")
print(oversampled_label_counts)

# -----------------------------
# 5. Save the Oversampled Dataset
# -----------------------------
# You can save the oversampled dataset to a new CSV file.
oversampled_df.to_csv("oversampled_dataset.csv", index=False)
print("Oversampled dataset saved to 'oversampled_dataset.csv'")
