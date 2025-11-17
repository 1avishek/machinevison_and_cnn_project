import os
import pandas as pd

# Directories
ROOT_DIR = "data"
XRAY_DIR = os.path.join(ROOT_DIR, "Femur", "Group 5")
MASK_DIR = os.path.join(ROOT_DIR, "Annotation_Femur_120_139_Kwanruethai_Kathy")
CSV_DIR = os.path.join(ROOT_DIR, "csvs")

os.makedirs(CSV_DIR, exist_ok=True)

# 1) Build dataset.csv by scanning the folders
rows = []

# List all xray images
xrays = [f for f in os.listdir(XRAY_DIR) if f.lower().endswith(".png")]

# Sort numerically if filenames are like "0.png", "1.png", ...
def sort_key(name):
    base = os.path.splitext(name)[0]
    try:
        return int(base)
    except ValueError:
        return base

xrays = sorted(xrays, key=sort_key)

for fname in xrays:
    xray_rel = os.path.join("data", "Femur", "Group 5", fname)
    mask_rel = os.path.join("data", "Annotation_Femur_120_139_Kwanruethai_Kathy", fname)

    # If mask file with same name exists, use it; otherwise mark as None
    if os.path.exists(os.path.join(ROOT_DIR, "Annotation_Femur_120_139_Kwanruethai_Kathy", fname)):
        mask_path = mask_rel
    else:
        mask_path = None  # unlabeled sample

    rows.append({"xrays": xray_rel, "masks": mask_path})

df = pd.DataFrame(rows)
dataset_path = os.path.join(CSV_DIR, "dataset.csv")
df.to_csv(dataset_path, index=False)
print(f"✔ Rebuilt dataset.csv with {len(df)} rows at {dataset_path}")

# 2) Split into train / val / test based on masks

# test = rows with no mask
test_df = df[df["masks"].isna()]

# labeled = rows with mask
labeled_df = df[~df["masks"].isna()].reset_index(drop=True)

print(f"Labeled samples: {len(labeled_df)}")
print(f"Unlabeled (test) samples: {len(test_df)}")

# Shuffle labeled samples
labeled_df = labeled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 80% train, 20% val
n_train = int(0.8 * len(labeled_df))
train_df = labeled_df.iloc[:n_train]
val_df = labeled_df.iloc[n_train:]

# Save splits
train_df.to_csv(os.path.join(CSV_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(CSV_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(CSV_DIR, "test.csv"), index=False)

print(f"✔ train.csv: {len(train_df)} samples")
print(f"✔ val.csv:   {len(val_df)} samples")
print(f"✔ test.csv:  {len(test_df)} samples")
