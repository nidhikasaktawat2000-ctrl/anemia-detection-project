import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# Configuration
DATA_DIR = "data"
MODEL_SAVE_PATH = "model/anemia_tf_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE = 15

os.makedirs("model", exist_ok=True)

# 1. Dataset Loading (Balanced across categories theoretically)
print(f"[INFO] Loading images from {DATA_DIR}...")
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    class_names=['anemia', 'non_anemia']
)

validation_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    class_names=['anemia', 'non_anemia']
)

class_names = train_dataset.class_names
print(f"[INFO] Discovered Classes: {class_names}")

# Fetching the dataset to calculate class weights for balancing
labels = []
for _, label_batch in train_dataset:
    labels.extend(label_batch.numpy().flatten())
counts = np.bincount(np.array(labels).astype(int))
total = sum(counts)
class_weight = {0: total / (2 * counts[0]), 1: total / (2 * counts[1])}
print(f"[INFO] Class Weights: {class_weight}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# 2. Data Augmentation
# We add GaussianNoise for noise reduction implicitly during training (model becomes robust)
# RandomFlip, Rotation, Zoom, Contrast handle real-world variations (pale palms, inner eyelids, fingernails)
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2),
    GaussianNoise(0.01)
], name="data_augmentation")

# 3. Model Architecture (Transfer Learning)
# Using MobileNetV2 for lightweight, high-accuracy inference
print("[INFO] Building Model architecture using MobileNetV2...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze the base model
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
# MobileNetV2 expects pixel values in [-1, 1], preprocess_input handles this scaling/normalization
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # High dropout to prevent overfitting
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# 4. Compilation and Callbacks
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 5. Training Loop
print("[INFO] Starting initial training phase...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weight,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# 6. Fine Tuning Phase
print("[INFO] Starting fine-tuning phase...")
base_model.trainable = True
# Freeze bottom layers of the base model
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_FINE,
    class_weight=class_weight,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

print(f"[INFO] High-Accuracy Model successfully saved to {MODEL_SAVE_PATH}")

# 7. Unseen Real-World Testing Function
def predict_and_display(image_path, model_path, class_names):
    if not os.path.exists(image_path):
        return
        
    loaded_model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = loaded_model.predict(img_array)
    score = predictions[0][0]
    
    # Threshold at 0.5
    predicted_class = class_names[1] if score >= 0.5 else class_names[0]
    confidence = score * 100 if score >= 0.5 else (1 - score) * 100
    
    print("-" * 40)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Detection Result: {predicted_class.upper()}")
    print(f"Confidence Score: {confidence:.2f}%")
    print("-" * 40)

# Provide a sample prediction if a sample image exists
sample_image = os.path.join(DATA_DIR, "anemia", os.listdir(os.path.join(DATA_DIR, "anemia"))[0]) if os.path.exists(os.path.join(DATA_DIR, "anemia")) else None
if sample_image:
    print("\n[INFO] Running evaluation on a sample image...")
    predict_and_display(sample_image, MODEL_SAVE_PATH, class_names)
