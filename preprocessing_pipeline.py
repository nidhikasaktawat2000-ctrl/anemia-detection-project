"""Advanced Preprocessing Pipeline
================================

Includes:
1. White Balance Correction - reduces lighting variation
2. Adaptive Histogram Equalization - improves contrast
3. Color normalization - reduces camera sensor differences
4. Data augmentation pipeline for training
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

def apply_white_balance_correction(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Reduce lighting variation using Gray World assumption.
    
    Works by:
    1. Calculate average color in image
    2. Normalize each channel so average color becomes neutral gray
    3. Reduces impact of different lighting conditions
    
    Before: Image taken under warm light → reddish
    After: Color normalized regardless of lighting
    """
    # Ensure float values
    if rgb_arr.dtype != np.float32:
        rgb_arr = rgb_arr.astype(np.float32) / 255.0
    
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
    
    # Clip to valid range
    return np.clip(rgb_arr, 0.0, 1.0)

def apply_clahe(rgb_arr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    
    Benefits:
    - Improves local contrast without amplifying noise
    - Better for medical/clinical images
    - Reduces impact of uneven lighting
    
    Args:
        rgb_arr: RGB image in [0, 1] or [0, 255]
        clip_limit: Contrast enhancement strength (default 2.0)
        tile_size: Size of local regions (default 8x8)
    """
    # Convert to 8-bit if needed
    if rgb_arr.max() <= 1.0:
        rgb_arr_8bit = (rgb_arr * 255).astype(np.uint8)
    else:
        rgb_arr_8bit = rgb_arr.astype(np.uint8)
    
    # Convert to LAB color space for processing L channel
    lab = cv2.cvtColor(rgb_arr_8bit, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l_channel)
    
    # Replace L channel
    lab[:, :, 0] = l_clahe
    
    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    
    return result

def color_constancy_correction(rgb_arr: np.ndarray, method: str = "gray_world") -> np.ndarray:
    """
    Apply color constancy correction.
    
    Reduces impact of different illuminants on color perception.
    Important for clinical images taken under different lighting.
    
    Args:
        rgb_arr: RGB image
        method: 'gray_world', 'white_patch', or 'shade_of_gray'
    """
    if rgb_arr.max() <= 1.0:
        rgb_arr = rgb_arr * 255.0
    
    if method == "gray_world":
        result = cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2XYZ).astype(np.float32)
        result = result / (result.mean(axis=(0, 1)) + 1e-6)
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_XYZ2BGR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        result = apply_white_balance_correction(rgb_arr / 255.0)
    
    return np.clip(result, 0.0, 1.0)

def get_training_augmentation():
    """
    Strong augmentation pipeline for training.
    
    Helps model generalize to real-world variations:
    - Different angles and positions
    - Different lighting conditions
    - Different camera qualities
    - Different skin tones
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=45, p=0.8),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.Affine(scale=(0.7, 1.3), translate_percent=(-0.2, 0.2), p=0.6),
        
        # Lighting/color augmentations (CRITICAL for clinical images)
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.CLAHE(p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        
        # Blur/sharpness variations
        A.GaussBlur(blur_limit=(3, 7), p=0.4),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.UnsharpMask(blur_limit=(3, 5), p=0.3),
        
        # Noise (mimics camera sensor noise)
        A.GaussNoise(p=0.3),
        A.ISONoise(p=0.2),
        
        # White balance simulation
        A.RandomRain(p=0.05),
        A.RandomFog(p=0.05),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=None)

def get_validation_augmentation():
    """
    Minimal augmentation for validation/testing.
    Only normalization and resizing.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=None)

def preprocess_image(
    image: Image.Image,
    apply_white_balance: bool = True,
    apply_clahe: bool = True,
    apply_color_constancy: bool = False
) -> Image.Image:
    """
    Apply preprocessing pipeline to single image.
    
    Args:
        image: PIL Image
        apply_white_balance: Reduce lighting variation
        apply_clahe: Improve local contrast
        apply_color_constancy: Additional color correction
    
    Returns:
        Preprocessed PIL Image
    """
    # Convert to numpy array
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    
    # White balance correction
    if apply_white_balance:
        img_np = apply_white_balance_correction(img_np)
    
    # CLAHE for contrast
    if apply_clahe:
        img_np = apply_clahe(img_np)
    
    # Additional color constancy
    if apply_color_constancy:
        img_np = color_constancy_correction(img_np)
    
    # Convert back to PIL Image
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

if __name__ == "__main__":
    print("Preprocessing pipeline module loaded.")
    print("\nAvailable functions:")
    print("  - apply_white_balance_correction()")
    print("  - apply_clahe()")
    print("  - color_constancy_correction()")
    print("  - get_training_augmentation()")
    print("  - get_validation_augmentation()")
    print("  - preprocess_image()")
