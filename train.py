import json
import os
import random
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# ---------- FULL TRAINING (UPGRADED) CONFIG ----------
DATA_DIR = "data"
MODEL_DIR = "model"
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")
MODEL_PATH = os.path.join(MODEL_DIR, "anemia_classical_model.joblib")

SEED = 42
VAL_SPLIT = 0.2
DATA_FRACTION = 1.0
TARGET_IMAGE_SIZE = (128, 128)
OUTLIER_Z = 3.0
RF_SEARCH_ITERS = 25
MAX_RUNTIME_SEC = 900
CV_FOLDS = 5
ANEMIA_RECALL_FLOOR = 0.85

random.seed(SEED)
np.random.seed(SEED)


def log(step: str, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


def load_image_paths(data_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    log("LOADING", f"Scanning dataset folder: {data_dir}")
    classes = ["anemia", "non_anemia"]
    missing = [c for c in classes if not os.path.isdir(os.path.join(data_dir, c))]
    if missing:
        raise ValueError(f"Missing required class folders under '{data_dir}': {missing}")

    samples: List[Tuple[str, int]] = []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath):
                samples.append((fpath, label))
    return samples, classes


def safe_sample(samples: List[Tuple[str, int]], fraction: float) -> List[Tuple[str, int]]:
    log("PREPROCESS", f"Applying dataset sampling fraction: {fraction:.2f}")
    if not 0 < fraction <= 1:
        return samples

    sampled: List[Tuple[str, int]] = []
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    for label, group in by_class.items():
        keep = max(12, int(len(group) * fraction))
        sampled.extend(random.sample(group, min(keep, len(group))))
        log("PREPROCESS", f"Class {label}: original={len(group)} sampled={min(keep, len(group))}")

    random.shuffle(sampled)
    return sampled


from features import image_to_features


def build_dataset(samples: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, int]:
    log("PREPROCESS", "Extracting upgraded image features")
    feats: List[np.ndarray] = []
    labels: List[int] = []
    invalid_count = 0
    first_error = None

    for path, label in samples:
        try:
            with Image.open(path) as img:
                feats.append(image_to_features(img, TARGET_IMAGE_SIZE))
            labels.append(label)
        except Exception as exc:
            invalid_count += 1
            if first_error is None:
                first_error = f"{path} -> {exc}"

    if not feats:
        detail = f" First error: {first_error}" if first_error else ""
        raise ValueError(f"No valid images left after reading/preprocessing.{detail}")

    x = np.vstack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y, invalid_count


def remove_outliers(x: np.ndarray, y: np.ndarray, z_thr: float) -> Tuple[np.ndarray, np.ndarray, int]:
    log("PREPROCESS", "Detecting outliers with z-score on feature means")
    row_signal = x.mean(axis=1)
    std = row_signal.std() + 1e-6
    z = np.abs((row_signal - row_signal.mean()) / std)
    keep_mask = z <= z_thr
    removed = int((~keep_mask).sum())
    return x[keep_mask], y[keep_mask], removed


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    total = float(len(y))
    weights_by_class = {cls: total / (len(counts) * max(cnt, 1)) for cls, cnt in enumerate(counts)}
    return np.array([weights_by_class[int(label)] for label in y], dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict:
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "report": report,
        "false_positives_non_anemia_as_anemia": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "false_negatives_anemia_as_non_anemia": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
    }


def evaluate(name: str, model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, classes: List[str]) -> Dict:
    log("TRAIN", f"Training model: {name}")
    model.fit(x_train, y_train)
    tr_pred = model.predict(x_train)
    va_pred = model.predict(x_val)
    train_metrics = compute_metrics(y_train, tr_pred, classes)
    val_metrics = compute_metrics(y_val, va_pred, classes)
    return {
        "name": name,
        "model": model,
        "train_acc": train_metrics["accuracy"],
        "val_acc": val_metrics["accuracy"],
        "gap": float(train_metrics["accuracy"] - val_metrics["accuracy"]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def threshold_predictions(proba: np.ndarray, anemia_idx: int, threshold: float) -> np.ndarray:
    return np.where(proba[:, anemia_idx] >= threshold, anemia_idx, 1 - anemia_idx).astype(np.int64)


def select_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: List[str],
    recall_floor: float = ANEMIA_RECALL_FLOOR,
) -> Dict:
    anemia_idx = classes.index("anemia")
    best = None

    for threshold in np.linspace(0.35, 0.70, 15):
        pred = threshold_predictions(proba, anemia_idx, float(threshold))
        metrics = compute_metrics(y_true, pred, classes)
        anemia_metrics = metrics["report"]["anemia"]
        non_anemia_metrics = metrics["report"]["non_anemia"]

        if anemia_metrics["recall"] < recall_floor:
            score = -1.0
        else:
            score = (
                non_anemia_metrics["precision"] * 0.55
                + non_anemia_metrics["recall"] * 0.20
                + anemia_metrics["recall"] * 0.15
                + metrics["accuracy"] * 0.10
            )

        candidate = {
            "threshold": round(float(threshold), 3),
            "score": round(float(score), 6),
            "metrics": metrics,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    assert best is not None
    return best


def main() -> None:
    t0 = time.time()
    log("START", "UPGRADED TRAINING MODE started")
    log("START", f"Runtime soft budget: {MAX_RUNTIME_SEC}s")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    samples, classes = load_image_paths(DATA_DIR)
    total_scanned = len(samples)
    log("LOADING", f"Total discovered images: {total_scanned}")
    samples = safe_sample(samples, DATA_FRACTION)
    sampled_count = len(samples)
    log("LOADING", f"Images selected for training: {sampled_count}")

    x_raw, y_raw, invalid_count = build_dataset(samples)
    log("PREPROCESS", f"Invalid/corrupt images skipped: {invalid_count}")

    x_clean, y_clean, outlier_count = remove_outliers(x_raw, y_raw, OUTLIER_Z)
    log("PREPROCESS", f"Outliers removed: {outlier_count}")

    if len(np.unique(y_clean)) < 2 or len(y_clean) < 20:
        raise ValueError("Insufficient clean samples for stable train/validation split.")

    x_train, x_val, y_train, y_val = train_test_split(
        x_clean,
        y_clean,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=y_clean,
    )
    log("PREPROCESS", f"Split complete -> train={len(y_train)}, val={len(y_val)}")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train_t = scaler.fit_transform(imputer.fit_transform(x_train))
    x_val_t = scaler.transform(imputer.transform(x_val))
    log("PREPROCESS", f"Feature matrix shape after preprocessing: train={x_train_t.shape}, val={x_val_t.shape}")

    sample_weights = compute_sample_weights(y_train)
    class_counts = np.bincount(y_clean)
    log("PREPROCESS", f"Class counts after cleaning: {class_counts.tolist()}")

    baseline = LogisticRegression(max_iter=800, random_state=SEED, class_weight="balanced")
    log("TRAIN", "Training baseline with class-weight balancing")
    baseline.fit(x_train_t, y_train, sample_weight=sample_weights)
    base_train_pred = baseline.predict(x_train_t)
    base_val_pred = baseline.predict(x_val_t)
    baseline_result = {
        "name": "Baseline (Logistic Regression)",
        "train_acc": float(accuracy_score(y_train, base_train_pred)),
        "val_acc": float(accuracy_score(y_val, base_val_pred)),
        "gap": float(accuracy_score(y_train, base_train_pred) - accuracy_score(y_val, base_val_pred)),
        "train_metrics": compute_metrics(y_train, base_train_pred, classes),
        "val_metrics": compute_metrics(y_val, base_val_pred, classes),
    }

    non_anemia_scorer = make_scorer(fbeta_score, beta=0.5, pos_label=1)
    
    xgb_base = XGBClassifier(random_state=SEED, eval_metric="logloss", use_label_encoder=False)
    
    xgb_params = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "gamma": [0.1, 0.5, 1.0],
        "reg_alpha": [0.1, 1.0],
        "reg_lambda": [1.0, 5.0]
    }
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    log("TRAIN", f"Running XGBoost tuning: iterations={RF_SEARCH_ITERS}, cv_folds={CV_FOLDS}")
    xgb_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=xgb_params,
        n_iter=RF_SEARCH_ITERS,
        scoring=non_anemia_scorer,
        cv=cv,
        random_state=SEED,
        n_jobs=1,
        verbose=0,
    )
    rf_result = evaluate("Tuned XGBoost", xgb_search, x_train_t, y_train, x_val_t, y_val, classes)

    best_model = rf_result["model"].best_estimator_
    val_proba = best_model.predict_proba(x_val_t)
    threshold_result = select_decision_threshold(y_val, val_proba, classes)
    selected_threshold = threshold_result["threshold"]
    threshold_metrics = threshold_result["metrics"]

    artifact = {
        "model": best_model,
        "imputer": imputer,
        "scaler": scaler,
        "classes": classes,
        "image_size": TARGET_IMAGE_SIZE,
        "feature_type": "upgraded_color_texture_statistics_v2",
        "anemia_threshold": selected_threshold,
        "threshold_selection_rule": {
            "objective": "maximize_non_anemia_precision_with_recall_floor",
            "anemia_recall_floor": ANEMIA_RECALL_FLOOR,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    log("TRAIN", f"Saved model artifact: {MODEL_PATH}")

    elapsed = time.time() - t0
    report = {
        "mode": "UPGRADED_PRECISION_FOCUSED",
        "runtime_seconds": round(elapsed, 2),
        "runtime_target_seconds": MAX_RUNTIME_SEC,
        "classes": classes,
        "data_fraction": DATA_FRACTION,
        "image_size": TARGET_IMAGE_SIZE,
        "feature_type": artifact["feature_type"],
        "feature_dimension": int(x_train_t.shape[1]),
        "samples_total_scanned": total_scanned,
        "samples_used_for_training": sampled_count,
        "invalid_images_skipped": invalid_count,
        "outliers_removed": outlier_count,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "class_counts_after_cleaning": class_counts.tolist(),
        "baseline": {
            "name": baseline_result["name"],
            "train_accuracy": round(baseline_result["train_acc"], 4),
            "val_accuracy": round(baseline_result["val_acc"], 4),
            "overfit_gap": round(baseline_result["gap"], 4),
            "validation": baseline_result["val_metrics"],
        },
        "after_random_forest": {
            "name": rf_result["name"],
            "train_accuracy": round(rf_result["train_acc"], 4),
            "val_accuracy": round(rf_result["val_acc"], 4),
            "overfit_gap": round(rf_result["gap"], 4),
            "best_params": rf_result["model"].best_params_,
            "search_iterations": RF_SEARCH_ITERS,
            "cv_folds": CV_FOLDS,
            "cv_scoring": "fbeta_beta_0.5_non_anemia",
            "validation_default_threshold": rf_result["val_metrics"],
        },
        "selected_threshold": {
            "anemia_probability_threshold": selected_threshold,
            "selection_rule": threshold_result["score"],
            "validation_metrics": threshold_metrics,
        },
        "comparison": {
            "before_val_accuracy": round(baseline_result["val_acc"], 4),
            "after_val_accuracy": round(rf_result["val_acc"], 4),
            "improvement": round(rf_result["val_acc"] - baseline_result["val_acc"], 4),
            "validation_false_positives_default": rf_result["val_metrics"]["false_positives_non_anemia_as_anemia"],
            "validation_false_positives_thresholded": threshold_metrics["false_positives_non_anemia_as_anemia"],
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("TRAIN", f"Saved report: {REPORT_PATH}")

    log("RESULT", f"Baseline validation accuracy : {baseline_result['val_acc']:.4f}")
    log("RESULT", f"RF validation accuracy       : {rf_result['val_acc']:.4f}")
    log("RESULT", f"Chosen anemia threshold      : {selected_threshold:.3f}")
    log(
        "RESULT",
        "Val FP non_anemia->anemia   : "
        f"{threshold_metrics['false_positives_non_anemia_as_anemia']}",
    )
    log("RESULT", f"Runtime                     : {elapsed:.2f}s")
    if elapsed > MAX_RUNTIME_SEC:
        log("WARNING", "Runtime exceeded safe target. Reduce DATA_FRACTION or RF_SEARCH_ITERS.")
    else:
        log("RESULT", "Runtime is within safe target.")


if __name__ == "__main__":
    main()