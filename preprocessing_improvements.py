"""Improved preprocessing pipeline for robust real-world performance.

Includes:
1. ROI detection via KMeans clustering
2. White balance correction
3. Artifact removal
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple

def apply_white_balance_correction(rgb_arr: np.ndarray) -> np.ndarray:
    """Reduce lighting variation using Gray World assumption.
    
    Helps model generalize across different lighting conditions
    and camera white balance settings.
    """
    # Calculate average color
    avg_r = rgb_arr[:, :, 0].mean()
    avg_g = rgb_arr[:, :, 1].mean()
    avg_b = rgb_arr[:, :, 2].mean()
    
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    # Normalize each channel
    if avg_r > 1e-6:
        rgb_arr[:, :, 0] *= avg_gray / avg_r
    if avg_g > 1e-6:
        rgb_arr[:, :, 1] *= avg_gray / avg_g
    if avg_b > 1e-6:
        rgb_arr[:, :, 2] *= avg_gray / avg_b
    
    return np.clip(rgb_arr, 0.0, 1.0)

def detect_skin_region_kmeans(image: Image.Image) -> Image.Image:
    """Use KMeans to isolate skin pixels and remove background.
    
    More robust than fixed cropping, adapts to different images.
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)
    
    # KMeans to find dominant colors (skin vs background)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Find skin-like cluster (moderate hue and saturation)
    centers = kmeans.cluster_centers_
    skin_cluster = np.argmax([
        (c[1] > 30) * (c[2] > 60) * ((c[0] < 25) or (c[0] > 170))  # Skin HSV range
        for c in centers
    ])
    
    # Create binary mask
    mask = (labels == skin_cluster).reshape(h, w).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find bounding box of skin region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Add padding
        pad = 10
        x, y = max(0, x - pad), max(0, y - pad)
        bw, bh = min(img_np.shape[1] - x, bw + 2*pad), min(img_np.shape[0] - y, bh + 2*pad)
        return image.crop((x, y, x + bw, y + bh))
    
    return image

def detect_and_crop_roi(image: Image.Image, region_type: str) -> Image.Image:
    """Detect and crop region of interest (nail, conjunctiva, or palm).
    
    Reduces background noise and focuses on clinically relevant area.
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    
    if region_type == "nail":
        # Nails typically occupy top 25-30% of image, center horizontally
        crop_height = int(h * 0.30)
        crop_width = int(w * 0.35)
        left = (w - crop_width) // 2
        top = int(h * 0.05)
        return image.crop((left, top, left + crop_width, top + crop_height))
    
    elif region_type == "eye":
        # Conjunctiva is in lower half of inner eyelid, compact region
        crop_height = int(h * 0.25)
        crop_width = int(w * 0.30)
        left = (w - crop_width) // 2
        top = int(h * 0.35)
        return image.crop((left, top, left + crop_width, top + crop_height))
    
    elif region_type == "palm":
        # Palm typically center, remove fingers at edges
        crop_height = int(h * 0.50)
        crop_width = int(w * 0.50)
        left = (w - crop_width) // 2
        top = (h - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))
    
    return image

def extract_pallor_specific_features(lab_arr: np.ndarray, rgb_arr: np.ndarray) -> np.ndarray:
    """Extract clinical pallor indicators directly.
    
    Pallor = reduced redness + reduced blood oxygenation
    In LAB: Pallor → ↓ a-channel (less red), ↑ L (lighter)
    In RGB: Pallor → ↓ R-G difference, ↑ B relative to R
    """
    
    L = lab_arr[:, :, 0]
    a = lab_arr[:, :, 1]  # In [-1, 1] range
    b = lab_arr[:, :, 2]
    
    R = rgb_arr[:, :, 0]
    G = rgb_arr[:, :, 1]
    B = rgb_arr[:, :, 2]
    
    pallor_features = np.array([
        # Clinical indicators
        float(a.mean()),  # Average redness (↓ in anemia)
        float(a.std()),   # Redness uniformity
        float(np.percentile(a, 25)),  # Lower quartile redness (MOST IMPORTANT)
        float(np.percentile(a, 75)),  # Upper quartile redness
        float(L.mean()),  # Lightness (↑ in anemia)
        float((R - G).mean()),  # Red-Green difference (↓ in anemia)
        float((R - B).mean()),  # Red-Blue difference
        float(((R - G) ** 2).mean()),  # Variance of red-green difference
        # Saturation proxies
        float(np.std([R.mean(), G.mean(), B.mean()])),  # Color saturation proxy
        float((np.max([R, G, B], axis=0) - np.min([R, G, B], axis=0)).mean()),  # HSV saturation
    ], dtype=np.float32)
    
    return pallor_features

def preprocess_image(image: Image.Image, use_kmeans: bool = True,
                    region_type: str = "eye") -> Image.Image:
    """Full preprocessing pipeline."""
    
    # Step 1: Use smart ROI detection
    if use_kmeans:
        image = detect_skin_region_kmeans(image)
    else:
        image = detect_and_crop_roi(image, region_type)
    
    # Step 2: Ensure RGB
    image = image.convert("RGB")
    
    return image
