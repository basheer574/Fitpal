import os
import pandas as pd

# ---------------------------
# User configuration section
# ---------------------------
csv_path = 'final_dataset.csv'       # Replace with your CSV file path
image_directory = 'fitness_images'   # Replace with your image directory path
new_name_column = 'image_file'            # Column in CSV that contains new image names

# ---------------------------
# Load new names from CSV
# ---------------------------
df = pd.read_csv(csv_path)

if new_name_column not in df.columns:
    raise ValueError(f"Column '{new_name_column}' not found in the CSV file.")

# Create a list of new image names from the CSV
new_names = df[new_name_column].tolist()

# ---------------------------
# Get current image files
# ---------------------------
# List all files in the directory (adjust filtering for specific image extensions if needed)
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
image_files.sort()  # Ensure consistent ordering

# Process only as many files as available (if CSV has more names, only process up to the number of images)
count = min(len(new_names), len(image_files))
print(f"Processing {count} files...")

# ---------------------------
# Rename files with unique names
# ---------------------------
for i in range(count):
    old_file = image_files[i]
    new_name = new_names[i]
    
    # If the new name doesn't include an extension, preserve the original file's extension.
    if not os.path.splitext(new_name)[1]:
        _, ext = os.path.splitext(old_file)
        new_name += ext

    old_path = os.path.join(image_directory, old_file)
    new_path = os.path.join(image_directory, new_name)
    
    # Check if a file with the new name already exists. If yes, add a counter suffix.
    if os.path.exists(new_path):
        base, ext = os.path.splitext(new_name)
        counter = 1
        # Loop until a unique filename is found.
        while os.path.exists(os.path.join(image_directory, f"{base} ({counter}){ext}")):
            counter += 1
        new_name = f"{base} ({counter}){ext}"
        new_path = os.path.join(image_directory, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)
    print(f"Renamed: '{old_file}' -> '{new_name}'")

print("Renaming process completed!")
