"""Smart ROI (Region of Interest) Extraction for Clinical Images
==============================================================

Extracts and focuses on clinically relevant regions:
- Eye: Inner conjunctiva area
- Nail: Nail bed only, excluding skin around it
- Palm: Central palm region

Benefits:
1. Reduces background noise and lighting artifacts
2. Focuses model on relevant diagnostic features
3. Improves generalization across different image qualities
4. Reduces overfitting to background patterns
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

def detect_skin_region_kmeans(image: Image.Image, n_clusters: int = 3) -> np.ndarray:
    """
    Use KMeans clustering in HSV space to isolate skin pixels.
    More robust than fixed cropping - adapts to different images.
    
    Args:
        image: PIL Image in RGB
        n_clusters: Number of clusters for KMeans
    
    Returns:
        Binary mask of skin region
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    
    # Convert to HSV for better skin detection
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)
    
    # KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Find skin-like cluster
    # Skin HSV characteristics:
    # H: 0-25 (red-orange) or 170-180 (wrapping around)
    # S: 30-200 (moderate saturation)
    # V: 60-200 (brightness)
    centers = kmeans.cluster_centers_
    skin_scores = []
    
    for c in centers:
        h_val, s_val, v_val = c[0], c[1], c[2]
        # Score based on skin-like HSV values
        score = (
            ((h_val < 25 or h_val > 170) * 1.0) *  # Correct hue
            ((30 < s_val < 200) * 1.0) *  # Moderate saturation
            ((60 < v_val < 200) * 1.0)  # Reasonable brightness
        )
        skin_scores.append(score)
    
    skin_cluster = np.argmax(skin_scores) if max(skin_scores) > 0 else 0
    
    # Create binary mask
    mask = (labels == skin_cluster).reshape(h, w).astype(np.uint8) * 255
    
    return mask

def morph_cleanup_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological operations to clean up mask.
    Removes noise and fills small holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove small objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

def extract_bounding_box(mask: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """
    Find bounding box of masked region.
    
    Returns:
        (x, y, width, height)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding
    
    return x, y, w, h

def crop_to_roi(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop image to bounding box.
    """
    x, y, w, h = bbox
    img_np = np.array(image)
    h_img = img_np.shape[0]
    w_img = img_np.shape[1]
    
    # Ensure within bounds
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)
    
    cropped = image.crop((x, y, x2, y2))
    return cropped

def extract_eye_roi(image: Image.Image, use_kmeans: bool = True) -> Image.Image:
    """
    Extract Region of Interest for eye/conjunctiva images.
    
    Strategy:
    1. Detect skin region via KMeans or fixed cropping
    2. Focus on inner conjunctiva area (typically lower portion)
    3. Remove eyelashes and surrounding skin
    
    Args:
        image: PIL Image
        use_kmeans: If True, use KMeans for adaptive detection
    
    Returns:
        Cropped image focused on conjunctiva
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    
    if use_kmeans:
        # Adaptive skin detection
        mask = detect_skin_region_kmeans(image)
        mask = morph_cleanup_mask(mask)
        
        bbox = extract_bounding_box(mask, padding=5)
        if bbox:
            return crop_to_roi(image, bbox)
    
    # Fallback: fixed cropping
    # Conjunctiva typically in lower 40% of image, center horizontally
    crop_height = int(h * 0.30)
    crop_width = int(w * 0.35)
    left = (w - crop_width) // 2
    top = int(h * 0.35)  # Lower portion
    
    return image.crop((left, top, left + crop_width, top + crop_height))

def extract_nail_roi(image: Image.Image, use_kmeans: bool = True) -> Image.Image:
    """
    Extract Region of Interest for nail images.
    
    Strategy:
    1. Detect skin region
    2. Focus on nail bed (upper central region)
    3. Exclude surrounding skin and fingers
    
    Args:
        image: PIL Image
        use_kmeans: If True, use KMeans for adaptive detection
    
    Returns:
        Cropped image focused on nail bed
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    
    if use_kmeans:
        mask = detect_skin_region_kmeans(image)
        mask = morph_cleanup_mask(mask)
        
        bbox = extract_bounding_box(mask, padding=3)
        if bbox:
            return crop_to_roi(image, bbox)
    
    # Fallback: fixed cropping
    # Nail bed typically in top 35% of image, center horizontally
    crop_height = int(h * 0.30)
    crop_width = int(w * 0.35)
    left = (w - crop_width) // 2
    top = int(h * 0.05)  # Upper portion
    
    return image.crop((left, top, left + crop_width, top + crop_height))

def extract_palm_roi(image: Image.Image, use_kmeans: bool = True) -> Image.Image:
    """
    Extract Region of Interest for palm images.
    
    Strategy:
    1. Detect skin region
    2. Focus on central palm (remove fingers, edges)
    3. Keep uniform palm patch for consistent analysis
    
    Args:
        image: PIL Image
        use_kmeans: If True, use KMeans for adaptive detection
    
    Returns:
        Cropped image focused on palm center
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    
    if use_kmeans:
        mask = detect_skin_region_kmeans(image)
        mask = morph_cleanup_mask(mask, kernel_size=7)
        
        bbox = extract_bounding_box(mask, padding=15)
        if bbox:
            cropped = crop_to_roi(image, bbox)
            # Further refine to central region
            cropped_np = np.array(cropped)
            ch, cw = cropped_np.shape[:2]
            # Take central 60% x 60%
            margin_h = int(ch * 0.2)
            margin_w = int(cw * 0.2)
            return Image.fromarray(cropped_np[margin_h:ch-margin_h, margin_w:cw-margin_w])
    
    # Fallback: fixed cropping (central region)
    crop_height = int(h * 0.50)
    crop_width = int(w * 0.50)
    left = (w - crop_width) // 2
    top = (h - crop_height) // 2
    
    return image.crop((left, top, left + crop_width, top + crop_height))

def extract_roi(
    image: Image.Image,
    region_type: str = "eye",
    use_kmeans: bool = True
) -> Image.Image:
    """
    Generic ROI extraction based on region type.
    
    Args:
        image: PIL Image (RGB)
        region_type: 'eye', 'nail', or 'palm'
        use_kmeans: Use KMeans adaptive detection vs fixed cropping
    
    Returns:
        Cropped Image focused on region of interest
    """
    
    if region_type.lower() == "eye":
        return extract_eye_roi(image, use_kmeans)
    elif region_type.lower() == "nail":
        return extract_nail_roi(image, use_kmeans)
    elif region_type.lower() == "palm":
        return extract_palm_roi(image, use_kmeans)
    else:
        raise ValueError(f"Unknown region type: {region_type}. Use 'eye', 'nail', or 'palm'")

if __name__ == "__main__":
    # Example usage
    print("ROI Extraction module loaded.")
    print("\nUsage:")
    print("  from roi_extraction import extract_roi")
    print("  img = Image.open('image.jpg')")
    print("  roi = extract_roi(img, region_type='eye', use_kmeans=True)")
    print("  roi.save('roi_output.jpg')")
