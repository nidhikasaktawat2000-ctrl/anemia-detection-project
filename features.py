import numpy as np
from PIL import Image
from typing import Tuple

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])

import cv2

def rgb_to_lab(rgb_arr: np.ndarray) -> np.ndarray:
    # Use OpenCV for fast RGB to CIELAB conversion
    rgb_uint8 = (np.clip(rgb_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    return lab_uint8.astype(np.float32) / 255.0

def image_to_features(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    # Convert using OpenCV for preprocessing
    rgb_img = np.array(image.convert("RGB").resize(size))
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    arr = rgb_img.astype(np.float32) / 255.0
    hsv_arr = hsv_img.astype(np.float32) / 255.0

    gray = arr.mean(axis=2)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    h = hsv_arr[:, :, 0]
    s = hsv_arr[:, :, 1]
    v = hsv_arr[:, :, 2]

    g20, g80 = np.percentile(gray, [20, 80])
    denom = max(g80 - g20, 1e-6)
    gray_norm = np.clip((gray - g20) / denom, 0.0, 1.0)

    per_ch_mean = arr.mean(axis=(0, 1))
    per_ch_std = arr.std(axis=(0, 1))
    per_ch_q10 = np.percentile(arr, 10, axis=(0, 1))
    per_ch_q50 = np.percentile(arr, 50, axis=(0, 1))
    per_ch_q90 = np.percentile(arr, 90, axis=(0, 1))

    hsv_mean = hsv_arr.mean(axis=(0, 1))
    hsv_std = hsv_arr.std(axis=(0, 1))
    hsv_q10 = np.percentile(hsv_arr, 10, axis=(0, 1))
    hsv_q90 = np.percentile(hsv_arr, 90, axis=(0, 1))

    # CIELAB features
    lab_arr = rgb_to_lab(arr)
    lab_mean = lab_arr.mean(axis=(0, 1))
    lab_std = lab_arr.std(axis=(0, 1))
    lab_q10 = np.percentile(lab_arr, 10, axis=(0, 1))
    lab_q90 = np.percentile(lab_arr, 90, axis=(0, 1))

    grad_x = gray_norm[:, 1:] - gray_norm[:, :-1]
    grad_y = gray_norm[1:, :] - gray_norm[:-1, :]
    grad_mag = np.sqrt(grad_x[:-1, :] ** 2 + grad_y[:, :-1] ** 2)

    color_ratios = np.array(
        [
            float((r / (g + 1e-6)).mean()),
            float((r / (b + 1e-6)).mean()),
            float((g / (b + 1e-6)).mean()),
            float((r - g).mean()),
            float((r - b).mean()),
            float((g - b).mean()),
        ],
        dtype=np.float32,
    )

    # Improved masks for better feature separation and more informative statistics
    bright_mask = gray > np.percentile(gray, 85)
    mid_bright_mask = (gray >= np.percentile(gray, 60)) & (gray <= np.percentile(gray, 85))
    dark_mask = gray < np.percentile(gray, 15)
    mid_dark_mask = (gray >= np.percentile(gray, 15)) & (gray <= np.percentile(gray, 40))
    low_sat_mask = s < np.percentile(s, 20)
    high_sat_mask = s > np.percentile(s, 80)

    mask_features = np.array(
        [
            float(bright_mask.mean()),
            float(dark_mask.mean()),
            float(low_sat_mask.mean()),
            float(high_sat_mask.mean()),
            float(gray[bright_mask].mean()) if bright_mask.any() else 0.0,
            float(gray[dark_mask].mean()) if dark_mask.any() else 0.0,
            float(s[low_sat_mask].mean()) if low_sat_mask.any() else 0.0,
            float(s[high_sat_mask].mean()) if high_sat_mask.any() else 0.0,
        ],
        dtype=np.float32,
    )

    quad_stats = []
    for patch in (gray[:48, :48], gray[:48, 48:], gray[48:, :48], gray[48:, 48:]):
        quad_stats.extend([float(patch.mean()), float(patch.std())])
    quad_stats = np.array(quad_stats, dtype=np.float32)

    gray_stats = np.array(
        [
            gray.mean(),
            gray.std(),
            np.percentile(gray, 10),
            np.percentile(gray, 50),
            np.percentile(gray, 90),
            gray_norm.mean(),
            gray_norm.std(),
            grad_mag.mean(),
            grad_mag.std(),
            np.percentile(grad_mag, 90),
        ],
        dtype=np.float32,
    )

    corr_features = np.array(
        [
            _safe_corr(r, g),
            _safe_corr(r, b),
            _safe_corr(g, b),
            _safe_corr(gray, s),
            _safe_corr(gray, v),
            _safe_corr(s, v),
        ],
        dtype=np.float32,
    )

    hist, _ = np.histogram(gray.flatten(), bins=16, range=(0.0, 1.0), density=True)
    hist_norm, _ = np.histogram(gray_norm.flatten(), bins=16, range=(0.0, 1.0), density=True)
    sat_hist, _ = np.histogram(s.flatten(), bins=8, range=(0.0, 1.0), density=True)

    return np.concatenate(
        [
            per_ch_mean,
            per_ch_std,
            per_ch_q10.astype(np.float32),
            per_ch_q50.astype(np.float32),
            per_ch_q90.astype(np.float32),
            hsv_mean,
            hsv_std,
            hsv_q10.astype(np.float32),
            hsv_q90.astype(np.float32),
            lab_mean.astype(np.float32),
            lab_std.astype(np.float32),
            lab_q10.astype(np.float32),
            lab_q90.astype(np.float32),
            gray_stats,
            color_ratios,
            mask_features,
            quad_stats,
            corr_features,
            hist.astype(np.float32),
            hist_norm.astype(np.float32),
            sat_hist.astype(np.float32),
        ]
    )
