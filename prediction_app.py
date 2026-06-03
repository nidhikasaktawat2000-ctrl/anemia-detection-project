from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import os
import zipfile
import tempfile
from torch.utils.data import Dataset, DataLoader
import random

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing pipeline expected by EfficientNet-B0
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load Model — EfficientNet-B0 ────────────────────────
def load_model(model_path=None):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1)
    )

    trained = False
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            print("[SUCCESS] Loaded trained B0 model")
            trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")
    else:
        print("[WARNING] No trained model found - using pretrained only")

    model.to(device)
    model.eval()
    return model, trained

MODEL_PATH = "model/anemia_model.pth"
model, MODEL_TRAINED = load_model(MODEL_PATH)

# ── Predict single image ─────────────────────────────────
def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob   = torch.sigmoid(output).item()

        return round(prob * 100, 1)
    except Exception as e:
        print(f"Predict error: {e}")
        return 0.0

from collections import Counter

# Simple custom dataset for uploaded images
class RealWorldAnemiaDataset(Dataset):
    """Dataset that loads images from a directory.
    Expected folder structure:
        root/
            Anemic/   (images labeled as Anemic)
            Normal/   (images labeled as Normal)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label_name in ["Anemic", "Normal"]:
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(label_dir, fname), 1 if label_name == "Anemic" else 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def fine_tune_on_realworld(dataset_path, epochs=3, lr=1e-4):
    """Fine‑tune the classifier head on a real‑world dataset.
    Returns the updated model.
    """
    model, _ = load_model(MODEL_PATH)
    model.to(device)
    model.train()
    # Freeze backbone, train only classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    dataset = RealWorldAnemiaDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[FINETUNE] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
    model.eval()
    return model

def get_age_category(age_val, unit="years"):
    try:
        age_val = float(age_val)
    except:
        return "Adult (31-45 yrs)" # default
    
    if unit == "days":
        age_val = age_val / 365.0
            
    if age_val <= 1.0:
        return "Newborn (0-1 yr)"
    elif age_val <= 5.0:
        return "Infant (1-5 yrs)"
    elif age_val <= 12.0:
        return "Child (6-12 yrs)"
    elif age_val <= 18.0:
        return "Adolescent (13-18 yrs)"
    elif age_val <= 30.0:
        return "Young Adult (19-30 yrs)"
    elif age_val <= 45.0:
        return "Adult (31-45 yrs)"
    else:
        return "Middle-Aged (46-60 yrs)"

def get_dataset_hb_threshold(age_category, gender):
    if age_category == "Newborn (0-1 yr)":
        return 13.5
    elif age_category == "Infant (1-5 yrs)":
        return 11.0
    elif age_category == "Child (6-12 yrs)":
        return 11.5
    else:
        return 13.0 if gender == "Male" else 12.0

# ── Voting ───────────────────────────────────────────────
def weighted_vote(predictions):
    score = 0.0
    anemic_count = 0
    total_weight = 0.0

    weights = {
        "Nail Bed Pallor": 1.0,
        "Conjunctival Pallor": 1.2,
        "Skin Tone Pallor": 0.8
    }

    for p in predictions:
        feature = p.get('feature', '')
        weight = weights.get(feature, 1.0)
        prob = p.get('anemia_probability', p.get('confidence', 0) / 100.0)
        
        if p['is_anemia']:
            anemic_count += 1
            
        score += prob * weight
        total_weight += weight
        
    avg_weighted_prob = score / total_weight if total_weight > 0 else 0
    
    anemic_preds = [p for p in predictions if p['is_anemia']]
    if any(p['confidence'] >= 90.0 for p in anemic_preds):
        final_label = 'Anemic'
        rule = 'critical_sign_override'
    elif avg_weighted_prob >= 0.5:
        final_label = 'Anemic'
        rule = 'weighted_majority'
    else:
        final_label = 'Normal'
        rule = 'weighted_majority'

    labels = [p['prediction'] for p in predictions]
    counts = Counter(labels)
    
    return {
        'final_prediction': final_label,
        'rule_used': rule,
        'is_anemia': final_label == 'Anemic',
        'votes': labels,
        'counts': dict(counts),
        'anemia_votes': anemic_count,
        'total_votes': len(predictions),
        'weighted_score': round(avg_weighted_prob * 100, 1)
    }

def baseline_vote(predictions):
    # Simple majority voting (original implementation)
    labels = [p['prediction'] for p in predictions]
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    # Standard majority voting without aggressive override
    if top_count >= 2:
        final_label = top_label
        rule = 'majority'
    else:
        # Tie or all different: choose highest confidence
        by_conf = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        final_label = by_conf[0]['prediction'] if by_conf else labels[0]
        rule = 'highest_confidence'
    return {
        'final_prediction': final_label,
        'rule_used': rule,
        'is_anemia': final_label == 'Anemic',
        'votes': labels,
        'counts': dict(counts),
        'anemia_votes': int(counts.get('Anemic', 0)),
        'total_votes': len(labels)
    }

# ── Hemoglobin estimate ──────────────────────────────────
def estimate_hb(confidences, predictions, final_is_anemia, age_category, gender, rule_used):
    avg_conf = float(np.mean(confidences))
    expected_hb = get_dataset_hb_threshold(age_category, gender)
    
    if final_is_anemia:
        if rule_used == "critical_sign_override":
            anemic_conf = max([p["confidence"] for p in predictions if p["is_anemia"]] + [50.0])
            hb = expected_hb - (anemic_conf / 100.0) * 4.0
        else:
            hb = expected_hb - (avg_conf / 100.0) * 3.0
    else:
        hb = expected_hb + (avg_conf / 100.0) * 2.0
        
    return round(max(5.0, min(hb, 20.0)), 1)

# ── Severity ─────────────────────────────────────────────
def get_severity(final_is_anemia, avg_conf):
    if not final_is_anemia:
        return "None"
    if avg_conf < 55:   return "Mild"
    if avg_conf < 80:   return "Moderate"
    return "Severe"

# ── Routes ───────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "anemia-detection.html")

@app.route("/predict", methods=["POST"])
def predict():
    required = ["nail", "eye", "palm"]

    for key in required:
        if key not in request.files:
            return jsonify({"error": f"Missing image: {key}"}), 400

    age = request.form.get("age", 25)
    age_unit = request.form.get("age_unit", "years")
    gender = request.form.get("gender", "Female")
    age_category = get_age_category(age, age_unit)

    try:
        predictions = []
        confidences = []

        feature_map = {
            "nail": "Nail Bed Pallor",
            "eye":  "Conjunctival Pallor",
            "palm": "Skin Tone Pallor"
        }
        source_map = {
            "nail": "Image 1 — Fingernail",
            "eye":  "Image 2 — Conjunctiva",
            "palm": "Image 3 — Palm / Finger"
        }
        
        expected_hb = get_dataset_hb_threshold(age_category, gender)
        # Raise base threshold for anemia to be highly conservative (reduce false positives)
        base_threshold = 0.88
        hb_diff = 12.0 - expected_hb
        anemia_threshold = base_threshold + (hb_diff * 0.05)  # Adjusted sensitivity
        # Ensure threshold stays within a realistic range (80%–95%)
        anemia_threshold = max(0.80, min(anemia_threshold, 0.95))
        threshold_percentage = anemia_threshold * 100.0

        for key in required:
            img_bytes = request.files[key].read()
            conf      = predict_image(img_bytes)
            confidences.append(conf)

            pred_label = "Anemic" if conf >= threshold_percentage else "Normal"
            predictions.append({
                "feature":    feature_map[key],
                "source":     source_map[key],
                "confidence": conf,
                "anemia_probability": conf / 100.0,
                "threshold_used": threshold_percentage,
                "prediction": pred_label,
                "is_anemia":  pred_label == "Anemic"
            })
        # Compute weighted vote and prepare response
        vote = weighted_vote(predictions)
        avg_conf = float(np.mean(confidences))
        hb = estimate_hb(confidences, predictions, vote["is_anemia"], age_category, gender, vote["rule_used"])
        severity = get_severity(vote["is_anemia"], avg_conf)

        return jsonify({
            "model_trained": MODEL_TRAINED,
            "threshold_used": predictions[0]["threshold_used"] if predictions else None,
            "patient_info": {
                "age": age,
                "unit": age_unit,
                "gender": gender,
                "age_category": age_category
            },
            "features": predictions,
            "predictions": [p["prediction"] for p in predictions],
            "majority_vote": vote,
            "final_result_text": f"Final Prediction (Weighted Voting): {vote['final_prediction']}",
            "metrics": {
                "haemoglobin": str(hb),
                "pallor_score": f"{int(avg_conf)}%",
                "severity": severity
            }
        })
@app.route("/predict_ab", methods=["POST"]) 
def predict_ab():
    """A/B test endpoint returning both weighted and baseline voting results."""
    required = ["nail", "eye", "palm"]
    for key in required:
        if key not in request.files:
            return jsonify({"error": f"Missing image: {key}"}), 400
    age = request.form.get("age", 25)
    age_unit = request.form.get("age_unit", "years")
    gender = request.form.get("gender", "Female")
    age_category = get_age_category(age, age_unit)
    try:
        predictions = []
        confidences = []
        feature_map = {
            "nail": "Nail Bed Pallor",
            "eye":  "Conjunctival Pallor",
            "palm": "Skin Tone Pallor"
        }
        source_map = {
            "nail": "Image 1 — Fingernail",
            "eye":  "Image 2 — Conjunctiva",
            "palm": "Image 3 — Palm / Finger"
        }
        expected_hb = get_dataset_hb_threshold(age_category, gender)
        base_threshold = 0.88
        hb_diff = 12.0 - expected_hb
        anemia_threshold = base_threshold + (hb_diff * 0.05)
        anemia_threshold = max(0.80, min(anemia_threshold, 0.95))
        threshold_percentage = anemia_threshold * 100.0
        for key in required:
            img_bytes = request.files[key].read()
            conf = predict_image(img_bytes)
            confidences.append(conf)
            pred_label = "Anemic" if conf >= threshold_percentage else "Normal"
            predictions.append({
                "feature": feature_map[key],
                "source":  source_map[key],
                "confidence": conf,
                "prediction": pred_label,
                "is_anemia": pred_label == "Anemic"
            })
        # Weighted vote
        weighted = weighted_vote(predictions)
        # Baseline majority vote
        baseline = baseline_vote(predictions)
        avg_conf = float(np.mean(confidences))
        hb_weighted = estimate_hb(confidences, predictions, weighted["is_anemia"], age_category, gender, weighted["rule_used"])
        hb_baseline = estimate_hb(confidences, predictions, baseline["is_anemia"], age_category, gender, baseline["rule_used"])
        severity_weighted = get_severity(weighted["is_anemia"], avg_conf)
        severity_baseline = get_severity(baseline["is_anemia"], avg_conf)
        return jsonify({
            "model_trained": MODEL_TRAINED,
            "patient_info": {
                "age": age,
                "unit": age_unit,
                "gender": gender,
                "age_category": age_category
            },
            "features": predictions,
            "weighted_vote": weighted,
            "baseline_vote": baseline,
            "hb": {
                "weighted": str(hb_weighted),
                "baseline": str(hb_baseline)
            },
            "severity": {
                "weighted": severity_weighted,
                "baseline": severity_baseline
            },
            "metrics": {
                "pallor_score": f"{int(avg_conf)}%"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# New endpoint to accept a zipped real‑world dataset and fine‑tune the model
@app.route("/train_dataset", methods=["POST"]) 
def train_dataset():
    """Accept a zip file containing 'Anemic' and 'Normal' folders.
    The zip is extracted to a temporary directory and the model is fine‑tuned.
    Returns a success message and the updated training flag.
    """
    if 'dataset_zip' not in request.files:
        return jsonify({"error": "Missing dataset_zip file"}), 400
    zip_file = request.files['dataset_zip']
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'uploaded.zip')
            zip_file.save(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            # Assume the zip contains a top‑level folder with the required structure
            extracted_root = tmpdir
            # Fine‑tune the model (this may take a few seconds depending on dataset size)
            global model
            model = fine_tune_on_realworld(extracted_root, epochs=2)
        return jsonify({"status": "model fine‑tuned with uploaded real‑world dataset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    print("=" * 50)
    print("[INFO] AnemAI Backend Running")
    print(f"   Model trained : {MODEL_TRAINED}")
    print(f"   Model path    : {MODEL_PATH}")
    print(f"   URL           : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
