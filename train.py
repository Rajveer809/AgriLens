"""
Image-Based Crop Disease Detection Using Deep Learning
Methodology: ResNet50 Transfer Learning with 2-Phase Training
Dataset: New Plant Diseases Dataset (Kaggle) - 87k images, 38 classes
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from pathlib import Path

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
DATA_DIR        = "./data"          # root containing train/ and valid/ folders
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_CLASSES     = 38                # 38 disease classes in the dataset
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-4
PHASE1_EPOCHS   = 10
PHASE2_EPOCHS   = 20
EARLY_STOP_PAT  = 5
DROPOUT         = 0.5
DENSE_UNITS     = 512
UNFREEZE_LAYERS = 30               # unfreeze last N layers in phase 2
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────
#  4. Data Preprocessing Pipeline
# ─────────────────────────────────────────────

# 4.1–4.2  Resize + ImageNet normalisation
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Training transforms include augmentation (4.4)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),          # 4.1 resize to 224×224
    transforms.RandomHorizontalFlip(),                # 4.4 random horizontal flip
    transforms.RandomRotation(15),                    # 4.4 ±15° rotation
    transforms.ColorJitter(brightness=0.2),           # 4.4 brightness adjustment
    transforms.RandomResizedCrop(IMG_SIZE,            # 4.4 random zoom 0.8–1.2
                                  scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std) # 4.2 normalise to [0,1] + ImageNet stats
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


def load_data(data_dir: str):
    """
    Load dataset and apply 70/15/15 stratified split (4.3).
    Expects data_dir/train/ and optionally data_dir/valid/.
    """
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    if os.path.isdir(valid_dir):
        # Dataset already has train/valid split – merge and re-split 70/15/15
        full_dataset = datasets.ImageFolder(train_dir, transform=None)
        val_dataset  = datasets.ImageFolder(valid_dir, transform=None)
        # Combine (we'll re-split below)
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([full_dataset, val_dataset])
        n_total = len(combined)
    else:
        full_dataset = datasets.ImageFolder(train_dir, transform=None)
        combined = full_dataset
        n_total = len(combined)

    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        combined, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Wrap with transforms
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_set = TransformDataset(train_set, train_transforms)
    val_set   = TransformDataset(val_set,   val_test_transforms)
    test_set  = TransformDataset(test_set,  val_test_transforms)

    # Get class names from the base dataset
    if hasattr(combined, 'datasets'):
        class_names = combined.datasets[0].classes
    else:
        class_names = combined.classes

    print(f"[DATA] Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"[DATA] Classes: {len(class_names)}")
    return train_set, val_set, test_set, class_names


# ─────────────────────────────────────────────
#  5. Model Architecture
# ─────────────────────────────────────────────

class CropDiseaseModel(nn.Module):
    """
    ResNet50 backbone + custom head:
      GlobalAvgPool → Dense(512, ReLU) → Dropout(0.5) → Softmax(num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # 5.1 Base: pretrained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove original FC layer; keep everything up to the avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 5.2 Custom head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))   # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, DENSE_UNITS),                     # Dense(512)
            nn.ReLU(),
            nn.Dropout(DROPOUT),                              # Dropout(0.5)
            nn.Linear(DENSE_UNITS, num_classes)               # Output (logits → Softmax via loss)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)

    def freeze_backbone(self):
        """Phase 1: freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n(self, n: int = UNFREEZE_LAYERS):
        """Phase 2: unfreeze the last n layers of the backbone."""
        layers = list(self.backbone.parameters())
        for param in layers[-n:]:
            param.requires_grad = True


# ─────────────────────────────────────────────
#  6. Training Strategy
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PAT, path: str = "best_model.pth"):
        self.patience   = patience
        self.path       = path
        self.best_loss  = np.inf
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
            print(f"  ✔ Best model saved (val_loss={val_loss:.4f})")
        else:
            self.counter += 1
            print(f"  ⚠ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def run_phase(model, train_loader, val_loader, optimizer, criterion,
              epochs, phase_name, early_stopper, scheduler):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"[{phase_name}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.2f}%")

        early_stopper(vl_loss, model)
        if early_stopper.stop:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    return history


# ─────────────────────────────────────────────
#  Evaluation & Plots
# ─────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_history(history, phase_name, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title(f"{phase_name} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train")
    ax2.plot([a * 100 for a in history["val_acc"]],   label="Val")
    ax2.set_title(f"{phase_name} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{phase_name.lower().replace(' ', '_')}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ── Load Data ──────────────────────────────
    train_set, val_set, test_set, class_names = load_data(DATA_DIR)
    num_classes = len(class_names)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Build Model ────────────────────────────
    model = CropDiseaseModel(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Feature Extraction ───────────
    print("\n" + "="*55)
    print("  PHASE 1: Feature Extraction (backbone frozen)")
    print("="*55)
    model.freeze_backbone()

    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR
    )
    scheduler_p1    = ReduceLROnPlateau(optimizer_p1, patience=2, factor=0.5)
    early_stopper_p1 = EarlyStopping(patience=EARLY_STOP_PAT,
                                     path=os.path.join(output_dir, "best_phase1.pth"))

    history_p1 = run_phase(model, train_loader, val_loader,
                            optimizer_p1, criterion,
                            PHASE1_EPOCHS, "Phase 1",
                            early_stopper_p1, scheduler_p1)
    plot_history(history_p1, "Phase 1", output_dir)

    # ── Phase 2: Fine-Tuning ──────────────────
    print("\n" + "="*55)
    print("  PHASE 2: Fine-Tuning (last 30 layers unfrozen)")
    print("="*55)
    # Load best weights from phase 1 before fine-tuning
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_phase1.pth"),
                                     map_location=DEVICE))
    model.unfreeze_last_n(UNFREEZE_LAYERS)

    optimizer_p2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR
    )
    scheduler_p2     = ReduceLROnPlateau(optimizer_p2, patience=3, factor=0.5)
    early_stopper_p2 = EarlyStopping(patience=EARLY_STOP_PAT,
                                     path=os.path.join(output_dir, "best_model.pth"))

    history_p2 = run_phase(model, train_loader, val_loader,
                            optimizer_p2, criterion,
                            PHASE2_EPOCHS, "Phase 2",
                            early_stopper_p2, scheduler_p2)
    plot_history(history_p2, "Phase 2", output_dir)

    # ── Final Evaluation on Test Set ──────────
    print("\n" + "="*55)
    print("  FINAL EVALUATION ON TEST SET")
    print("="*55)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"),
                                     map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

    # Save metrics
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save class names
    with open(os.path.join(output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\n[DONE] All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
