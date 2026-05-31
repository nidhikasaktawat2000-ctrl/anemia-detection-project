"""Fixed patient-aware dataset splitting to prevent data leakage.

IMPORTANT: This prevents same patient appearing in both train and validation sets.
"""

import os
import shutil
import random
from collections import defaultdict

source_dir = "data"
dest_dir = "dataset"
classes = ["anemia", "non_anemia"]
split_ratio = 0.80  # 80% train, 20% val

def extract_patient_id(filename):
    """Extract patient ID from filename, handling augmentation suffixes."""
    base = os.path.splitext(filename)[0]
    if "_aug" in base:
        base = base.split("_aug")[0]
    # Remove trailing body part identifiers
    for suffix in ['_nail', '_eye', '_palm', '_fingernail', '_conjunctiva']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base

print("\n" + "="*70)
print("PATIENT-AWARE DATASET SPLITTING (FIX FOR DATA LEAKAGE)")
print("="*70)

for cls in classes:
    src_path = os.path.join(source_dir, cls)
    if not os.path.isdir(src_path):
        print(f"\n⚠️  Class folder not found: {src_path}")
        continue
    
    files = os.listdir(src_path)
    
    # Group by patient ID
    patient_files = defaultdict(list)
    for f in files:
        pid = extract_patient_id(f)
        patient_files[pid].append(f)
    
    # Split patients (not images!)
    patients = list(patient_files.keys())
    random.shuffle(patients)
    
    split_index = int(len(patients) * split_ratio)
    train_patients = set(patients[:split_index])
    val_patients = set(patients[split_index:])
    
    train_files = [f for pid in train_patients for f in patient_files[pid]]
    val_files = [f for pid in val_patients for f in patient_files[pid]]
    
    # Create folders
    os.makedirs(f"{dest_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{dest_dir}/val/{cls}", exist_ok=True)
    
    # Copy files
    for f in train_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(dest_dir, "train", cls, f)
        )
    
    for f in val_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(dest_dir, "val", cls, f)
        )
    
    print(f"\n[{cls.upper()}]")
    print(f"  Unique patients:")
    print(f"    Train: {len(train_patients)} patients → {len(train_files)} images")
    print(f"    Val:   {len(val_patients)} patients → {len(val_files)} images")
    print(f"  ✅ No patient overlap between train and val")

print("\n" + "="*70)
print("✅ Dataset split complete (patient-aware, no leakage)")
print("="*70 + "\n")
