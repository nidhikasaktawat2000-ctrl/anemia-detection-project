import json
import os
import random
from typing import List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

SEED = 42
DATA_DIR = "data"
MODEL_PATH = os.path.join("model", "anemia_classical_model.joblib")
OUT_PATH = os.path.join("model", "unseen_test_report.json")
IMAGE_SIZE = (128, 128)
OUTLIER_Z = 3.0

random.seed(SEED)
np.random.seed(SEED)


def load_paths(data_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if len(classes) != 2:
        raise ValueError(f"Expected 2 class folders in {data_dir}, found {classes}")
    samples = []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for name in os.listdir(cls_dir):
            p = os.path.join(cls_dir, name)
            if os.path.isfile(p):
                samples.append((p, label))
    return samples, classes


from features import image_to_features


def build_matrix(samples: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for path, label in samples:
        try:
            with Image.open(path) as img:
                feats.append(image_to_features(img, IMAGE_SIZE))
            labels.append(label)
        except Exception:
            continue
    if not feats:
        raise ValueError("No valid images to evaluate.")
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


def remove_outliers(x: np.ndarray, y: np.ndarray, z_thr: float) -> Tuple[np.ndarray, np.ndarray, int]:
    row_signal = x.mean(axis=1)
    z = np.abs((row_signal - row_signal.mean()) / (row_signal.std() + 1e-6))
    keep = z <= z_thr
    return x[keep], y[keep], int((~keep).sum())


def main() -> None:
    samples, classes = load_paths(DATA_DIR)
    x, y = build_matrix(samples)
    x, y, outliers = remove_outliers(x, y, OUTLIER_Z)

    # Fresh split: 70% train, 15% val, 15% unseen test
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.30, random_state=SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train_t = scaler.fit_transform(imputer.fit_transform(x_train))
    x_val_t = scaler.transform(imputer.transform(x_val))
    x_test_t = scaler.transform(imputer.transform(x_test))

    if os.path.exists(MODEL_PATH):
        artifact = joblib.load(MODEL_PATH)
        params = artifact["model"].get_params()
        model = HistGradientBoostingClassifier(**params)
        classes = artifact.get("classes", classes)
        anemia_threshold = float(artifact.get("anemia_threshold", 0.5))
    else:
        model = HistGradientBoostingClassifier(
            random_state=SEED,
            class_weight="balanced",
            learning_rate=0.1,
            max_iter=200,
            max_depth=10,
            min_samples_leaf=20,
            l2_regularization=0.1,
        )
        anemia_threshold = 0.5

    model.fit(x_train_t, y_train)
    train_pred = model.predict(x_train_t)
    val_pred = model.predict(x_val_t)
    test_proba = model.predict_proba(x_test_t)
    anemia_idx = next((i for i, cls in enumerate(classes) if cls.lower().startswith("anemia")), 0)
    test_pred = np.where(test_proba[:, anemia_idx] >= anemia_threshold, anemia_idx, 1 - anemia_idx)

    train_acc = float(accuracy_score(y_train, train_pred))
    val_acc = float(accuracy_score(y_val, val_pred))
    test_acc = float(accuracy_score(y_test, test_pred))
    overfit_gap = train_acc - test_acc
    cm = confusion_matrix(y_test, test_pred, labels=list(range(len(classes))))

    result = {
        "classes": classes,
        "feature_type": artifact.get("feature_type", "unknown") if os.path.exists(MODEL_PATH) else "fallback",
        "anemia_threshold": round(anemia_threshold, 3),
        "samples_after_cleaning": int(len(y)),
        "outliers_removed": outliers,
        "split": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "accuracy": {
            "train": round(train_acc, 4),
            "val": round(val_acc, 4),
            "test_unseen": round(test_acc, 4),
            "overfit_gap_train_minus_test": round(overfit_gap, 4),
        },
        "assessment": (
            "moderate_overfitting" if overfit_gap > 0.15 else
            "mild_overfitting" if overfit_gap > 0.08 else
            "well_generalized"
        ),
        "confusion_matrix": cm.tolist(),
        "false_positives_non_anemia_as_anemia": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "false_negatives_anemia_as_non_anemia": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "test_report": classification_report(y_test, test_pred, target_names=classes, output_dict=True, zero_division=0),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Unseen evaluation complete.")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy  : {val_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Overfit gap   : {overfit_gap:.4f}")
    print(f"Saved         : {OUT_PATH}")


if __name__ == "__main__":
    main()
