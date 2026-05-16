import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile

# Allow loading of truncated images if any
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Configuration ──────────────────────────────────────────
DATA_DIR = "data"
MODEL_SAVE_PATH = "model/anemia_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Data Transforms (Augmentation) ───────────────────────
# We use aggressive augmentation on the training set to prevent overfitting
# and improve generalization on real-world clinical photos.
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomAutocontrast(p=0.3),
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

# ── Dataset Loading ────────────────────────────────────────
print(f"[INFO] Loading images from {DATA_DIR}...")
full_dataset = datasets.ImageFolder(root=DATA_DIR)

# 80% Train, 20% Validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Apply respective transforms
train_ds.dataset = copy.copy(full_dataset)
train_ds.dataset.transform = train_transforms
val_ds.dataset.transform = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class_names = full_dataset.classes
print(f"[INFO] Classes: {class_names}")
print(f"[INFO] Training images: {train_size}")
print(f"[INFO] Validation images: {val_size}")

# ── Model Architecture ─────────────────────────────────────
# We use EfficientNet-B0 as our base architecture and replace the classifier head
print("[INFO] Initializing EfficientNet-B0 model...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze early layers to retain generic feature extraction (optional, but speeds up early training)
for param in list(model.parameters())[:-30]:
    param.requires_grad = False

# Replace classifier for binary classification
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(in_features, 1) # Single output for BCEWithLogitsLoss
)

model = model.to(device)

# ── Training Components ────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ── Training Loop ──────────────────────────────────────────
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

print("[INFO] Starting Training...")
since = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1) # [batch_size, 1]

            # We assume 'anemia' is class 0, and 'non_anemia' is class 1
            # We want 'Anemic' to be 1 and 'Normal' to be 0
            # Let's check how ImageFolder mapped them:
            # Usually alphabetical: anemia -> 0, non_anemia -> 1
            # We want predicting 1 to mean Anemia.
            # So let's invert the labels if anemia is 0.
            if class_names[0] == 'anemia':
                labels = 1.0 - labels

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                preds = torch.sigmoid(outputs) >= 0.5

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples * 100

        print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        if phase == 'val':
            scheduler.step(epoch_acc)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("   [*] New best model found!")

time_elapsed = time.time() - since
print(f"\n[INFO] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(f"[INFO] Best Validation Accuracy: {best_acc:.2f}%")

# Save the best model
os.makedirs("model", exist_ok=True)
torch.save(best_model_wts, MODEL_SAVE_PATH)
print(f"[INFO] Model successfully saved to {MODEL_SAVE_PATH}")
