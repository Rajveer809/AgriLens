# Image-Based Crop Disease Detection
### ResNet50 Transfer Learning · 38 Classes · ~87k Images

---

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download from Kaggle:  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Place it so your folder looks like:
```
data/
  train/   ← ~70,295 images across 38 class folders
  valid/   ← ~17,572 images
```

## Training

```bash
python train.py
```

Outputs are saved to `outputs/`:
- `best_model.pth`        — best weights (Phase 2 fine-tuning)
- `best_phase1.pth`       — best weights (Phase 1 feature extraction)
- `phase_1_curves.png`    — loss & accuracy curves
- `phase_2_curves.png`
- `confusion_matrix.png`
- `metrics.json`          — final test accuracy + classification report
- `class_names.json`      — ordered list of 38 class names

## Inference (CLI)

```bash
python predict.py --image path/to/leaf.jpg
```

---

## Web Interface (New)

We have added a modern, web-based UI for the project.

To run the web interface:
1. Activate the virtual environment (if you are using one):
   ```bash
   source venv/bin/activate
   ```
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://localhost:5001`.

---

## Architecture

```
Input (224×224×3)
        ↓
  ResNet50 backbone (ImageNet pretrained)
        ↓
  Global Average Pooling
        ↓
  Dense(512) + ReLU
        ↓
  Dropout(0.5)
        ↓
  Softmax Output (38 classes)
```

## Training Strategy

| Phase | Layers | LR    | Epochs |
|-------|--------|-------|--------|
| 1 – Feature Extraction | Backbone frozen, head only | 1e-3 | 10     |
| 2 – Fine-Tuning        | Last 30 layers unfrozen    | 1e-4 | 15–20  |

- **Optimizer:** Adam  
- **Loss:** CrossEntropyLoss  
- **Batch size:** 32  
- **Early stopping:** patience = 5 (monitors val loss)

## Expected Results

| Setting         | Accuracy     |
|-----------------|-------------|
| Controlled dataset | 95–99%   |
| Real field images  | 80–90%   |
