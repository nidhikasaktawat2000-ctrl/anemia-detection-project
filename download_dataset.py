"""
Dataset Download Script
========================
Downloads anemia classification dataset from Kaggle.

Before running:
1. Go to https://www.kaggle.com/settings → API → Create New Token
2. Place the downloaded kaggle.json in:
   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
   Mac/Linux: ~/.kaggle/kaggle.json
3. Run: python download_dataset.py
"""

import os
import zipfile
import shutil

def download():
    print("Downloading anemia dataset from Kaggle...")
    os.system("kaggle datasets download -d navoneel/anemia-classification -p ./raw_data")

    print("Extracting...")
    with zipfile.ZipFile("./raw_data/anemia-classification.zip", "r") as zf:
        zf.extractall("./raw_data/extracted")

    # ── Organise into data/anemia and data/non_anemia ──
    print("Organising dataset...")
    os.makedirs("data/anemia", exist_ok=True)
    os.makedirs("data/non_anemia", exist_ok=True)

    extracted = "./raw_data/extracted"
    for root, dirs, files in os.walk(extracted):
        for fname in files:
            fpath = os.path.join(root, fname)
            lower = fname.lower() + root.lower()
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                if "anemia" in lower and "non" not in lower:
                    shutil.copy(fpath, f"data/anemia/{fname}")
                elif "non" in lower or "normal" in lower or "healthy" in lower:
                    shutil.copy(fpath, f"data/non_anemia/{fname}")

    anemia_count    = len(os.listdir("data/anemia"))
    non_anemia_count = len(os.listdir("data/non_anemia"))
    print(f"\nDataset ready!")
    print(f"  Anemia images:     {anemia_count}")
    print(f"  Non-Anemia images: {non_anemia_count}")
    print(f"\nNext step: python train.py")

if __name__ == "__main__":
    download()
