# AnemAI — Anemia Detection System
## Full Setup Guide (VS Code + Local Machine)

---

## Project Structure
```
anemia_project/
├── app.py                 ← Flask backend API
├── train.py               ← Model training script
├── download_dataset.py    ← Kaggle dataset downloader
├── requirements.txt       ← Python dependencies
├── model/                 ← Trained model saved here (auto-created)
├── data/
│   ├── anemia/            ← Anemia images (put here)
│   └── non_anemia/        ← Non-anemia images (put here)
└── anemia-detection.html  ← Frontend (open in browser)
```

---

## Step 1 — Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## Step 2 — Download Dataset from Kaggle

### Option A: Auto-download (recommended)
1. Go to https://www.kaggle.com/settings → API → **Create New Token**
2. Place `kaggle.json` in:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`
3. Run:
```bash
python download_dataset.py
```

### Option B: Manual download
1. Go to: https://www.kaggle.com/datasets/navoneel/anemia-classification
2. Download and extract
3. Copy images manually into:
   - `data/anemia/`       ← all anemia images
   - `data/non_anemia/`   ← all non-anemia images

---

## Step 3 — Train the Model
```bash
python train.py
```
- Performs dataset audit (invalid files, missing values, outlier images)
- Extracts **EfficientNet-B0 embeddings** and runs feature selection
- Trains and compares:
  - Baseline Logistic Regression
  - Tuned Random Forest
  - Tuned XGBoost
- Saves best classical model to `model/anemia_classical_model.joblib`
- Saves report to `model/training_report.json` with before/after accuracy

Training time:
- CPU: ~30–60 min
- GPU (CUDA): ~5–10 min

---

## Step 4 — Start Flask Backend
```bash
python app.py
```
Backend runs at: **http://localhost:5000**

---

## Step 5 — Open Frontend
Open `anemia-detection.html` in your browser (or use VS Code Live Server).

Upload 3 images → Click **Run Analysis** → Get real AI predictions.

---

## Notes
- The frontend calls `http://localhost:5000/predict`
- Make sure Flask is running BEFORE clicking Run Analysis
- For best results: use clear, well-lit images
- Nail: no polish, flat against camera, natural light
- Conjunctiva: pull lower eyelid down, front camera close-up
- Palm: open hand, natural daylight

---

## Improving Accuracy Further
- Add more training data (more images = better accuracy)
- Try `EfficientNet-B3` or `ResNet-50` in `train.py`
- Increase `EPOCHS` to 50 in `train.py`
- Use a GPU for faster training
