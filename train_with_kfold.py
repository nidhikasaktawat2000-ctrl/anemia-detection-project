"""K-Fold Cross-Validation Training Pipeline
============================================

Provides robust evaluation by training on multiple folds.
Better assessment of model generalization vs single train-test split.

Benefits:
1. Better estimate of true model performance
2. Detect overfitting across different data splits
3. Use all data for both training and validation
4. More stable metrics
"""

import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib
from pathlib import Path
import random
from PIL import Image
from typing import Dict, List, Tuple

def extract_patient_id(filepath: str) -> str:
    """Extract patient ID from file path."""
    filename = os.path.basename(filepath)
    base = os.path.splitext(filename)[0]
    if "_aug" in base:
        base = base.split("_aug")[0]
    for suffix in ['_nail', '_eye', '_palm', '_fingernail', '_conjunctiva']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base.strip()

def load_dataset_stratified(
    data_dir: str,
    image_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset and maintain stratification info.
    
    Returns:
        X: Feature matrix
        y: Labels
        patient_ids: Patient ID for each image (for group split)
    """
    from features import image_to_features
    
    features_list = []
    labels_list = []
    patient_ids = []
    
    for label_idx, class_name in enumerate(["anemia", "non_anemia"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue
            
            try:
                img = Image.open(fpath).convert("RGB")
                feat = image_to_features(img, image_size)
                features_list.append(feat)
                labels_list.append(label_idx)
                patient_ids.append(extract_patient_id(fname))
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y, patient_ids

def train_with_kfold(
    data_dir: str,
    output_dir: str = "model",
    n_splits: int = 5,
    seed: int = 42,
    stratified_by: str = "class"  # 'class' or 'patient'
) -> Dict:
    """
    Train model using K-Fold Cross-Validation.
    
    Args:
        data_dir: Directory with anemia/non_anemia folders
        output_dir: Where to save results
        n_splits: Number of folds (default 5)
        seed: Random seed
        stratified_by: 'class' (stratify by class) or 'patient' (group by patient)
    
    Returns:
        Dictionary with fold results and best model
    """
    
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION (K={n_splits}, stratified_by={stratified_by})")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    X, y, patient_ids = load_dataset_stratified(data_dir)
    print(f"  Total samples: {len(X)}")
    print(f"  Anemia: {(y == 0).sum()}, Non-anemia: {(y == 1).sum()}")
    
    # Initialize K-Fold
    if stratified_by == "patient":
        # Group by patient - patients don't repeat across folds
        from sklearn.model_selection import GroupKFold
        unique_patients = list(set(patient_ids))
        patient_groups = np.array([unique_patients.index(pid) for pid in patient_ids])
        kfold = GroupKFold(n_splits=n_splits)
        splits = kfold.split(X, y, groups=patient_groups)
    else:
        # Stratified by class - maintain class balance in each fold
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kfold.split(X, y)
    
    fold_results = []
    best_model = None
    best_accuracy = 0.0
    best_fold = -1
    
    print(f"\nStarting {n_splits}-fold cross-validation...\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFOLD {fold_idx + 1}/{n_splits}")
        print("-" * 80)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Train: {len(train_idx)} samples ({(y_train == 0).sum()} anemia, {(y_train == 1).sum()} normal)")
        print(f"  Val:   {len(val_idx)} samples ({(y_val == 0).sum()} anemia, {(y_val == 1).sum()} normal)")
        
        # Preprocessing
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        
        X_train_t = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_t = scaler.transform(imputer.transform(X_val))
        
        # Train model
        model = HistGradientBoostingClassifier(
            random_state=seed,
            class_weight="balanced",
            max_iter=300,
            max_depth=8,
            learning_rate=0.05,
            l2_regularization=0.1,
        )
        
        model.fit(X_train_t, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_t)
        val_pred = model.predict(X_val_t)
        
        val_proba = model.predict_proba(X_val_t)[:, 0]
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="weighted")
        val_auc = roc_auc_score(y_val, val_proba)
        
        cm = confusion_matrix(y_val, val_pred)
        
        fold_result = {
            "fold": fold_idx + 1,
            "train_acc": round(float(train_acc), 4),
            "val_acc": round(float(val_acc), 4),
            "val_f1": round(float(val_f1), 4),
            "val_auc": round(float(val_auc), 4),
            "overfit_gap": round(float(train_acc - val_acc), 4),
            "confusion_matrix": cm.tolist(),
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
        }
        
        fold_results.append(fold_result)
        
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Acc:   {val_acc:.4f}")
        print(f"  Val F1:    {val_f1:.4f}")
        print(f"  Val AUC:   {val_auc:.4f}")
        print(f"  Overfit:   {train_acc - val_acc:.4f}")
        
        # Track best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = {
                "model": model,
                "imputer": imputer,
                "scaler": scaler,
                "fold": fold_idx + 1,
            }
            best_fold = fold_idx + 1
    
    # Summary statistics
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    train_accs = [r["train_acc"] for r in fold_results]
    val_accs = [r["val_acc"] for r in fold_results]
    val_f1s = [r["val_f1"] for r in fold_results]
    val_aucs = [r["val_auc"] for r in fold_results]
    overfit_gaps = [r["overfit_gap"] for r in fold_results]
    
    print(f"\nTrain Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Val Accuracy:   {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Val F1 Score:   {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Val AUC:        {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"Overfit Gap:    {np.mean(overfit_gaps):.4f} ± {np.std(overfit_gaps):.4f}")
    
    # Assess overfitting
    mean_gap = np.mean(overfit_gaps)
    if mean_gap > 0.20:
        status = "🔴 SEVERE OVERFITTING"
    elif mean_gap > 0.10:
        status = "🟡 MODERATE OVERFITTING"
    elif mean_gap > 0.05:
        status = "🟠 MILD OVERFITTING"
    else:
        status = "🟢 WELL GENERALIZED"
    
    print(f"\nOverfitting Assessment: {status}")
    print(f"Best model from fold: {best_fold} (val_acc={best_accuracy:.4f})")
    
    # Save results
    report = {
        "method": f"K-Fold Cross-Validation (K={n_splits})",
        "stratified_by": stratified_by,
        "n_splits": n_splits,
        "summary": {
            "mean_train_acc": round(float(np.mean(train_accs)), 4),
            "std_train_acc": round(float(np.std(train_accs)), 4),
            "mean_val_acc": round(float(np.mean(val_accs)), 4),
            "std_val_acc": round(float(np.std(val_accs)), 4),
            "mean_val_f1": round(float(np.mean(val_f1s)), 4),
            "mean_val_auc": round(float(np.mean(val_aucs)), 4),
            "mean_overfit_gap": round(float(np.mean(overfit_gaps)), 4),
            "overfitting_status": status,
        },
        "fold_results": fold_results,
        "best_fold": best_fold,
        "best_val_accuracy": round(float(best_accuracy), 4),
    }
    
    report_path = os.path.join(output_dir, "kfold_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Report saved to: {report_path}")
    
    # Save best model
    if best_model:
        best_model_path = os.path.join(output_dir, "anemia_model_kfold_best.joblib")
        joblib.dump({
            "model": best_model["model"],
            "imputer": best_model["imputer"],
            "scaler": best_model["scaler"],
            "classes": ["anemia", "non_anemia"],
            "fold": best_model["fold"],
            "validation_accuracy": best_accuracy,
        }, best_model_path)
        print(f"💾 Best model saved to: {best_model_path}")
    
    print("="*80 + "\n")
    
    return report

if __name__ == "__main__":
    report = train_with_kfold(
        data_dir="data",
        output_dir="model",
        n_splits=5,
        seed=42,
        stratified_by="class"
    )
    
    print("\n✅ K-Fold cross-validation complete!")
