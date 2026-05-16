import os
import shutil
import random

# Paths
source_dir = "data"
dest_dir = "dataset"

classes = ["anemia", "non_anemia"]
split_ratio = 0.8  # 80% train, 20% val

for cls in classes:
    src_path = os.path.join(source_dir, cls)
    files = os.listdir(src_path)

    random.shuffle(files)

    split_index = int(len(files) * split_ratio)

    train_files = files[:split_index]
    val_files = files[split_index:]

    # Create folders
    os.makedirs(f"{dest_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{dest_dir}/val/{cls}", exist_ok=True)

    # Copy train files
    for f in train_files:
        shutil.copy(os.path.join(src_path, f),
                    os.path.join(dest_dir, "train", cls, f))

    # Copy val files
    for f in val_files:
        shutil.copy(os.path.join(src_path, f),
                    os.path.join(dest_dir, "val", cls, f))

    print(f"{cls} → Train: {len(train_files)}, Val: {len(val_files)}")

print("\n✅ Dataset split complete!")
