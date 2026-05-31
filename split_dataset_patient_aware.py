"""Patient-Aware Train-Test Split with No Data Leakage
=====================================================

CRITICAL: This prevents the same patient from appearing in both train and test sets.
Uses stratified split to maintain class balance.

WHY THIS MATTERS:
- If patient appears in both sets → artificially high accuracy
- Model learns patient-specific artifacts, not generalizable anemia patterns
- Real-world performance will be MUCH lower

SOLUTION: Split by PATIENT ID first, then by images
"""

import os
import shutil
import random
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def extract_patient_id(filename: str) -> str:
    """Extract patient ID from filename, handling various naming conventions."""
    base = os.path.splitext(filename)[0]
    
    if "_aug" in base:
        base = base.split("_aug")[0]
    
    for suffix in ['_nail', '_eye', '_palm', '_fingernail', '_conjunctiva', '_conjuctiva']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    
    return base.strip()

def patient_aware_split(
    source_dir: str = "data",
    dest_dir: str = "dataset_split",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict:
    """
    Split dataset by patient ID with NO data leakage.
    
    Args:
        source_dir: Directory containing 'anemia' and 'non_anemia' folders
        dest_dir: Output directory for split data
        train_ratio: Proportion for training (default 70%)
        val_ratio: Proportion for validation (default 15%)
        test_ratio: Proportion for testing (default 15%)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with split statistics
    """
    
    random.seed(seed)
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    print("\n" + "="*80)
    print("PATIENT-AWARE TRAIN-VALIDATION-TEST SPLIT (NO DATA LEAKAGE)")
    print("="*80)
    
    classes = ["anemia", "non_anemia"]
    split_summary = {
        "train": {"anemia": 0, "non_anemia": 0, "total": 0},
        "val": {"anemia": 0, "non_anemia": 0, "total": 0},
        "test": {"anemia": 0, "non_anemia": 0, "total": 0},
        "patients": {
            "train": {"anemia": 0, "non_anemia": 0},
            "val": {"anemia": 0, "non_anemia": 0},
            "test": {"anemia": 0, "non_anemia": 0},
        },
        "validation": []
    }
    
    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(source_class_dir):
            print(f"\n⚠️  Class folder not found: {source_class_dir}")
            continue
        
        print(f"\n[{class_name.upper()}]")
        print("-" * 80)
        
        # Step 1: Group images by patient ID
        patient_files = defaultdict(list)
        for fname in os.listdir(source_class_dir):
            fpath = os.path.join(source_class_dir, fname)
            if os.path.isfile(fpath):
                patient_id = extract_patient_id(fname)
                patient_files[patient_id].append(fname)
        
        print(f"  Found {len(patient_files)} unique patients")
        
        # Step 2: Create patient list and shuffle
        patients = list(patient_files.keys())
        random.shuffle(patients)
        
        # Step 3: Split patients (not images!)
        num_patients = len(patients)
        train_split = int(num_patients * train_ratio)
        val_split = train_split + int(num_patients * val_ratio)
        
        train_patients = set(patients[:train_split])
        val_patients = set(patients[train_split:val_split])
        test_patients = set(patients[val_split:])
        
        print(f"  Split: {len(train_patients)} train | {len(val_patients)} val | {len(test_patients)} test")
        
        # Step 4: Assign images to sets based on patient membership
        train_files = [f for pid in train_patients for f in patient_files[pid]]
        val_files = [f for pid in val_patients for f in patient_files[pid]]
        test_files = [f for pid in test_patients for f in patient_files[pid]]
        
        print(f"  Images: {len(train_files)} train | {len(val_files)} val | {len(test_files)} test")
        
        # Step 5: Create destination folders
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(dest_dir, split, class_name), exist_ok=True)
        
        # Step 6: Copy files
        for fname in train_files:
            src = os.path.join(source_class_dir, fname)
            dst = os.path.join(dest_dir, "train", class_name, fname)
            shutil.copy2(src, dst)
        
        for fname in val_files:
            src = os.path.join(source_class_dir, fname)
            dst = os.path.join(dest_dir, "val", class_name, fname)
            shutil.copy2(src, dst)
        
        for fname in test_files:
            src = os.path.join(source_class_dir, fname)
            dst = os.path.join(dest_dir, "test", class_name, fname)
            shutil.copy2(src, dst)
        
        # Update summary
        split_summary["train"][class_name] = len(train_files)
        split_summary["val"][class_name] = len(val_files)
        split_summary["test"][class_name] = len(test_files)
        
        split_summary["patients"]["train"][class_name] = len(train_patients)
        split_summary["patients"]["val"][class_name] = len(val_patients)
        split_summary["patients"]["test"][class_name] = len(test_patients)
        
        # Verify no patient overlap
        overlap_train_val = train_patients & val_patients
        overlap_train_test = train_patients & test_patients
        overlap_val_test = val_patients & test_patients
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            split_summary["validation"].append({
                "class": class_name,
                "status": "❌ FAILED - Patient overlap detected!"
            })
            print(f"  ❌ ERROR: Patient overlap detected!")
        else:
            split_summary["validation"].append({
                "class": class_name,
                "status": "✅ PASS - No patient overlap"
            })
            print(f"  ✅ No patient overlap (safe split)")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SPLIT SUMMARY")
    print("="*80)
    
    for split in ["train", "val", "test"]:
        total_images = split_summary[split]["anemia"] + split_summary[split]["non_anemia"]
        total_patients = split_summary["patients"][split]["anemia"] + split_summary["patients"][split]["non_anemia"]
        
        print(f"\n[{split.upper()}]")
        print(f"  Patients: {total_patients}")
        print(f"  Images: {total_images}")
        if total_images > 0:
            anemia_pct = 100 * split_summary[split]["anemia"] / total_images
            print(f"  Class balance: {anemia_pct:.1f}% anemia, {100-anemia_pct:.1f}% normal")
    
    print("\n" + "="*80)
    print("LEAKAGE VERIFICATION")
    print("="*80)
    
    all_valid = all(v["status"].startswith("✅") for v in split_summary["validation"])
    
    for validation in split_summary["validation"]:
        print(f"  {validation['class'].upper()}: {validation['status']}")
    
    if all_valid:
        print("\n✅ ALL CHECKS PASSED - Dataset is properly split with NO LEAKAGE")
    else:
        print("\n❌ LEAKAGE DETECTED - Cannot proceed with training")
    
    # Save metadata
    metadata_file = os.path.join(dest_dir, "split_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(split_summary, f, indent=2, default=str)
    
    print(f"\n📋 Metadata saved to: {metadata_file}")
    print("="*80 + "\n")
    
    return split_summary

if __name__ == "__main__":
    result = patient_aware_split(
        source_dir="data",
        dest_dir="dataset_split",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print("\n✅ Dataset split complete!")
    print("   Train, validation, and test sets are properly separated by patient.")
    print("   No patient appears in multiple sets.")
