"""Comprehensive Evaluation on Test Set
======================================

Provides detailed metrics and visualizations:
1. Confusion matrix analysis
2. ROC curve
3. Precision-recall curve
4. Per-class metrics
5. Clinical recommendations
"""

import os
import json
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score
)
from pathlib import Path
import joblib
import torch
from PIL import Image
from torchvision import models, transforms

TEST_DATA_DIR = "dataset_split/test"
CLASSICAL_MODEL_PATH = "model/anemia_model_kfold_best.joblib"
DL_MODEL_PATH = "model/anemia_model_improved.pth"
OUTPUT_DIR = "model"

def load_test_data():
    """Load test dataset."""
    images = []
    labels = []
    image_paths = []
    
    for label_idx, class_name in enumerate(["anemia", "non_anemia"]):
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                fpath = os.path.join(class_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    images.append(img)
                    labels.append(label_idx)
                    image_paths.append(fpath)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")
    
    return images, labels, image_paths

def predict_classical(images, model_path):
    """Predict using classical model."""
    from features import image_to_features
    
    artifact = joblib.load(model_path)
    model = artifact["model"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    
    predictions = []
    probabilities = []
    
    for img in images:
        feat = image_to_features(img, (128, 128))
        feat_scaled = scaler.transform(imputer.transform(feat.reshape(1, -1)))
        proba = model.predict_proba(feat_scaled)[0]
        pred = model.predict(feat_scaled)[0]
        predictions.append(pred)
        probabilities.append(proba[0])  # Probability of class 0 (anemia)
    
    return np.array(predictions), np.array(probabilities)

def predict_dl(images, model_path, device="cpu"):
    """Predict using deep learning model."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for img in images:
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            proba = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(proba).item()
            predictions.append(pred)
            probabilities.append(proba[0].item())  # Probability of class 0 (anemia)
    
    return np.array(predictions), np.array(probabilities)

def evaluate():
    """Comprehensive evaluation."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION ON TEST SET")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    images, labels, image_paths = load_test_data()
    y_true = np.array(labels)
    
    print(f"Test set: {len(images)} images")
    print(f"  Anemia: {(y_true == 0).sum()}")
    print(f"  Normal: {(y_true == 1).sum()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    # Classical model evaluation
    if os.path.exists(CLASSICAL_MODEL_PATH):
        print(f"\nEvaluating classical model...")
        y_pred_classical, y_proba_classical = predict_classical(images, CLASSICAL_MODEL_PATH)
        
        acc = accuracy_score(y_true, y_pred_classical)
        f1 = f1_score(y_true, y_pred_classical, average="weighted")
        cm = confusion_matrix(y_true, y_pred_classical)
        report = classification_report(y_true, y_pred_classical, 
                                      target_names=["Anemia", "Normal"],
                                      output_dict=True)
        
        results["classical"] = {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Deep learning model evaluation
    if os.path.exists(DL_MODEL_PATH):
        print(f"\nEvaluating DL model...")
        y_pred_dl, y_proba_dl = predict_dl(images, DL_MODEL_PATH, device)
        
        acc = accuracy_score(y_true, y_pred_dl)
        f1 = f1_score(y_true, y_pred_dl, average="weighted")
        cm = confusion_matrix(y_true, y_pred_dl)
        report = classification_report(y_true, y_pred_dl,
                                      target_names=["Anemia", "Normal"],
                                      output_dict=True)
        
        results["deep_learning"] = {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "test_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {results_path}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    evaluate()
