from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import io

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

try:
    import pillow_avif  # noqa: F401 — registers AVIF with Pillow (folder / phone exports)
except ImportError:
    pass
import joblib
import numpy as np
import os
from collections import Counter
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

TARGET_IMAGE_SIZE = (224, 224)
MODEL_PATH = "model/anemia_classical_model.joblib"
DL_MODEL_PATH = "model/anemia_model.pth"


from features import image_to_features

ALLOWED_IMAGE_EXT = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".jpe",
        ".jfif",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
    }
)


def filename_allowed_photo(filename: str) -> bool:
    if not filename or not isinstance(filename, str):
        return True
    ext = os.path.splitext(filename.strip())[1].lower()
    if ext == "":
        return True
    return ext in ALLOWED_IMAGE_EXT


def load_model(model_path):
    if model_path and os.path.exists(model_path):
        try:
            artifact = joblib.load(model_path)
            print("Loaded trained classical model")
            return artifact, True
        except Exception as e:
            print(f"Could not load model: {e}")
    else:
        print("No trained model found - using pretrained only")
    return None, False

artifact, MODEL_TRAINED = load_model(MODEL_PATH)

def load_dl_model(model_path):
    if model_path and os.path.exists(model_path):
        try:
            device = torch.device("cpu")
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 1)
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("Loaded trained PyTorch deep learning model")
            return model, True
        except Exception as e:
            print(f"Could not load PyTorch model: {e}")
    return None, False

dl_model, DL_MODEL_TRAINED = load_dl_model(DL_MODEL_PATH)


def predict_image(image_bytes, age_category="Adult (31-45 yrs)", gender="Female"):
    if not (MODEL_TRAINED or DL_MODEL_TRAINED):
        raise RuntimeError("Models not trained. Run train.py and train_dl.py first.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(
            "Could not decode image (corrupt file, AVIF without pillow-avif-plugin, or unsupported type). "
            "Try JPG/PNG from folder, or run: pip install pillow-avif-plugin pillow-heif"
        ) from e
    image = image.convert("RGB")
    
    classes = ["anemia", "non_anemia"]
    
    # Demographic-aware thresholding handling using WHO hemoglobin standards
    # Female Hb < 12 = Anemic, Male Hb < 13 = Anemic
    expected_hb = 12.0 if gender.lower() == "female" else 13.0
    
    base_threshold = 0.55 
    hb_diff = 12.0 - expected_hb 
    anemia_threshold = base_threshold + (hb_diff * 0.05) 
    anemia_threshold = max(0.40, min(anemia_threshold, 0.70))
    
    xgb_prob = 0.5
    dl_prob = 0.5
    
    if DL_MODEL_TRAINED and dl_model is not None:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = dl_model(img_tensor)
            dl_prob = torch.sigmoid(output).item()
    
    if MODEL_TRAINED and artifact is not None:
        feat = image_to_features(image, tuple(artifact.get("image_size", (128, 128))))
        feat = feat.reshape(1, -1)

        imputer = artifact["imputer"]
        scaler = artifact["scaler"]
        model = artifact["model"]
        model_classes = artifact.get("classes", ["anemia", "non_anemia"])

        anemia_idx = next((i for i, cls in enumerate(model_classes) if cls.lower().startswith("anemia")), 0)

        x = scaler.transform(imputer.transform(feat))
        proba = model.predict_proba(x)[0]
        xgb_prob = float(proba[anemia_idx])
        
    if DL_MODEL_TRAINED and MODEL_TRAINED:
        anemia_probability = (dl_prob + xgb_prob) / 2.0
    elif DL_MODEL_TRAINED:
        anemia_probability = dl_prob
    else:
        anemia_probability = xgb_prob
        
    # --- Real-world Strict Calibration ---
    # The models are overly sensitive to smartphone images and overpredict Anemia.
    # We apply a penalty offset to force it to only flag true pallor.
    anemia_probability = max(0.0, anemia_probability - 0.40)
        
    pred_idx = 0 if anemia_probability >= anemia_threshold else 1
    
    # Calculate confidence relative to the chosen threshold
    if pred_idx == 0: # Assuming 0 is Anemia
        range_val = 1.0 - anemia_threshold
        confidence = 50.0 + ((anemia_probability - anemia_threshold) / range_val) * 50.0 if range_val > 0 else 100.0
    else:
        range_val = anemia_threshold
        confidence = 50.0 + ((anemia_threshold - anemia_probability) / range_val) * 50.0 if range_val > 0 else 100.0
        
    confidence = float(min(max(confidence, 0.0), 100.0))

    raw_label = "Anemic" if pred_idx == 0 else "Normal"
    predicted_label = raw_label
    
    return predicted_label, round(confidence, 1), round(anemia_probability * 100.0, 1), round(anemia_threshold * 100.0, 1)


def majority_vote(predictions, age_category):
    labels = [p["prediction"] for p in predictions]
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    
    # Strict Majority Voting - Removed aggressive single-image override
    if top_count >= 2:
        final_label = top_label
        rule = "majority"
    else:
        # All 3 different or tie: choose most confident, else first.
        by_conf = sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True)
        final_label = by_conf[0]["prediction"] if by_conf else labels[0]
        rule = "highest_confidence"

    return {
        "votes": labels,
        "counts": dict(counts),
        "final_prediction": final_label,
        "rule_used": rule,
        "is_anemia": final_label == "Anemic",
        "anemia_votes": int(counts.get("Anemic", 0)),
        "total_votes": len(labels),
    }

# ── Age Category & Thresholds ────────────────────────────
def get_age_category(age_val, unit):
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
    return 12.0 if gender.lower() == "female" else 13.0

# ── Hemoglobin estimate ──────────────────────────────────
def estimate_hb(confidences, predictions, final_is_anemia, age_category, gender, rule_used):
    avg_conf = float(np.mean(confidences))
    expected_hb = get_dataset_hb_threshold(age_category, gender)
    
    if final_is_anemia:
        if rule_used == "critical_sign_override":
            # Find the confidence of the specific image that triggered the anemia flag
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
        
    # Only calculate severity if they are actually Anemic
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

        for key in required:
            f = request.files[key]
            fname = f.filename or ""
            if fname and not filename_allowed_photo(fname):
                return jsonify(
                    {
                        "error": f"Only photo files are allowed for {key}. "
                        "Use JPG/JFIF, PNG, WEBP, AVIF, HEIC, GIF, BMP, or TIFF — not PDF/DOC/etc."
                    }
                ), 400
            f.seek(0)
            img_bytes = f.read()
            if not img_bytes:
                return jsonify({"error": f"Empty or unreadable upload for {key}. Re-select the image."}), 400
            try:
                pred_label, confidence, anemia_probability, threshold_used = predict_image(
                    img_bytes, age_category, gender
                )
            except ValueError as err:
                slot = source_map[key]
                hint = f' ({fname})' if fname else ""
                raise ValueError(f"{slot}{hint}: {err}") from err
            confidences.append(confidence)

            predictions.append({
                "feature":    feature_map[key],
                "source":     source_map[key],
                "confidence": confidence,
                "anemia_probability": anemia_probability,
                "threshold_used": threshold_used,
                "prediction": pred_label,
                "is_anemia":  pred_label == "Anemic"
            })

        vote = majority_vote(predictions, age_category)
        avg_conf = float(np.mean(confidences))
        hb = estimate_hb(confidences, predictions, vote["is_anemia"], age_category, gender, vote["rule_used"])
        severity = get_severity(vote["is_anemia"], avg_conf)

        return jsonify({
            "model_trained": MODEL_TRAINED or DL_MODEL_TRAINED,
            "dl_model_active": DL_MODEL_TRAINED,
            "threshold_used": predictions[0]["threshold_used"] if predictions else None,
            "patient_info": {
                "age": age,
                "unit": age_unit,
                "gender": gender,
                "age_category": age_category
            },
            "features":      predictions,
            "predictions":   [p["prediction"] for p in predictions],
            "majority_vote": vote,
            "final_result_text": f"Final Prediction (Majority Voting): {vote['final_prediction']}",
            "metrics": {
                "haemoglobin": str(hb),
                "pallor_score": f"{int(avg_conf)}%",
                "severity":    severity
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("AnemAI Backend Running")
    print(f"   Classical Model trained : {MODEL_TRAINED}")
    print(f"   DL Model trained        : {DL_MODEL_TRAINED}")
    print(f"   URL                     : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)