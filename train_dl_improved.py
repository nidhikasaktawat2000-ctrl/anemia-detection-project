"""Improved Deep Learning Training with Strong Augmentation
=========================================================

Key improvements:
1. Aggressive data augmentation for real-world robustness
2. Patient-aware train-val split (no leakage)
3. Strong regularization to prevent overfitting
4. LR scheduling and early stopping
5. Detailed metrics logging
"""

import os
import time
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

# For reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Configuration
DATA_DIR = "dataset_split"  # From split_dataset_patient_aware.py
MODEL_SAVE_PATH = "model/anemia_model_improved.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# ============================================
# DATA AUGMENTATION (STRONG FOR ROBUSTNESS)
# ============================================

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    
    # Geometric augmentations
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=45),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3)),
    
    # Color/lighting augmentations (CRITICAL for clinical images)
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomAutocontrast(p=0.4),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
    transforms.RandomEqualize(p=0.3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# CUSTOM DATASET CLASS
# ============================================

class AnemiaDataset(Dataset):
    """Custom dataset for anemia images."""
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """
        Args:
            data_dir: Path to dataset_split directory
            split: 'train' or 'val'
            transform: Transform pipeline
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        # Load image paths and labels
        for label_name in ["anemia", "non_anemia"]:
            label_dir = os.path.join(data_dir, split, label_name)
            if not os.path.isdir(label_dir):
                continue
            
            label_idx = 0 if label_name == "anemia" else 1
            
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    fpath = os.path.join(label_dir, fname)
                    self.samples.append((fpath, label_idx))
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        
        try:
            image = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            # Return dummy image on error
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# ============================================
# TRAINING LOOP
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    
    return epoch_loss, epoch_acc

def main():
    """Main training function."""
    
    print("\n" + "="*80)
    print("IMPROVED DEEP LEARNING TRAINING (EfficientNet-B0)")
    print("="*80)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AnemiaDataset(DATA_DIR, split="train", transform=train_transforms)
    val_dataset = AnemiaDataset(DATA_DIR, split="val", transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(128, 2)  # Binary classification
    )
    
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = 5
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Overfit:    {train_acc - val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✅ New best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping (no improvement for {patience} epochs)")
                break
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # Load best model
    model.load_state_dict(best_model_wts)
    
    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(best_model_wts, MODEL_SAVE_PATH)
    print(f"✅ Best model saved to: {MODEL_SAVE_PATH}")
    
    # Save history
    history_path = "model/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training history saved to: {history_path}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
