import os
import re
import pandas as pd

# ======================== 📌 Step 1: Clean Arabic EveTAR-S Dataset ========================

def clean_arabic_text(text):
    """
    Cleans Arabic text by removing unwanted characters, links, and normalizing letters.
    """
    if not isinstance(text, str):
        return ""  # Ignore non-string data

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions (@username) and hashtags (#tag)
    text = re.sub(r"@\S+|#\S+", "", text)

    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"[^؀-ۿ\s]", "", text)  # Keeping only Arabic letters and spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_evetar_file(input_file, output_file):
    """
    Reads EveTAR-S.txt (Arabic dataset), cleans it, and saves it as a CSV.
    """
    arabic_data = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            cleaned_text = clean_arabic_text(line)
            if cleaned_text:  # Only save non-empty lines
                arabic_data.append(cleaned_text)

    # Save Arabic dataset as CSV
    df_arabic = pd.DataFrame(arabic_data, columns=["text"])
    df_arabic.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ Cleaned Arabic dataset saved to: {output_file}")


# ======================== 📌 Step 2: Clean OASST1 English Dataset (Already CSV) ========================

def clean_english_text(text):
    """
    Cleans English text by removing unwanted characters, links, and extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters (keeping letters, digits, and basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\"()\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_oasst1_csv(input_file, output_file):
    """
    Reads OASST1 CSV dataset, cleans the text, and saves it back as CSV.
    """
    df_english = pd.read_csv(input_file)

    # Ensure there's a column named "text" (Modify if needed)
    if "text" not in df_english.columns:
        raise ValueError("The OASST1 dataset must contain a 'text' column.")

    # Clean English text
    df_english["text"] = df_english["text"].astype(str).apply(clean_english_text)

    # Remove empty rows after cleaning
    df_english = df_english[df_english["text"].str.strip() != ""]

    # Save cleaned English dataset as CSV
    df_english.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ Cleaned English dataset saved to: {output_file}")


# ======================== 📌 Step 3: Merge Arabic & English Datasets ========================

def merge_datasets(arabic_file, english_file, output_file):
    """
    Merges Arabic and English cleaned datasets into one file.
    """
    df_arabic = pd.read_csv(arabic_file)
    df_english = pd.read_csv(english_file)

    # Concatenate both datasets
    df_merged = pd.concat([df_arabic, df_english], ignore_index=True)

    # Save the final dataset
    df_merged.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ Final merged dataset saved to: {output_file}")


# ======================== 📌 Run the Cleaning and Merging Process ========================

arabic_input_file = "EveTAR-S.txt"  # Change to the actual EveTAR-S file path
cleaned_arabic_file = "cleaned_evetar_s.csv"

oasst1_input_file = "oasst1_english.csv"  # Your existing OASST1 CSV file
cleaned_english_file = "cleaned_oasst1.csv"

final_dataset = "language_dataset.csv"

# Step 1: Clean Arabic EveTAR-S
process_evetar_file(arabic_input_file, cleaned_arabic_file)

# Step 2: Clean English OASST1 CSV
process_oasst1_csv(oasst1_input_file, cleaned_english_file)

# Step 3: Merge Arabic & English datasets
merge_datasets(cleaned_arabic_file, cleaned_english_file, final_dataset)
