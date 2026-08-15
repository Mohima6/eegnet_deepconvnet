import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataloader import read_ds004504
from models import EEGNet, DeepConvNet

def seed_everything(seed=500):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(500)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_DIR = r"C:\Users\mohimaCHAKRABORTY\PycharmProjects\PythonProject8\uav-env\EEGNet_DeepConvNet\dataset"
CLASSES = ('A', 'F', 'C')          # AD, FTD, CN
CLASS_NAMES = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
LABEL_NAMES = [CLASS_NAMES[c] for c in CLASSES]

EPOCHS = 20        # trimmed for time budget
PATIENCE = 6
BATCH = 64

# =========================================================
# STEP 1: Load + window + subject-wise split (runs once)
# =========================================================
cache_path = os.path.join(SAVE_DIR, "epoched_data_3class.npz")

if os.path.exists(cache_path):
    print(f"Found cached epoched data at {cache_path}, loading (skipping MNE reload)...")
    data = np.load(cache_path, allow_pickle=True)
    Xtr, ytr = data['Xtr'], data['ytr']
    Xval, yval = data['Xval'], data['yval']
    Xte, yte = data['Xte'], data['yte']
else:
    print("No cache found — loading raw .set files (this is the slow step)...")
    t0 = time.time()
    Xtr, ytr, Xval, yval, Xte, yte, subj_split = read_ds004504(
        DATA_DIR, classes=CLASSES,
        window_sec=4.0, overlap=0.5, resample_hz=128.0,
        val_size=0.15, test_size=0.2, seed=42
    )
    print(f"Data loading took {(time.time()-t0)/60:.1f} min")
    np.savez(cache_path, Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval, Xte=Xte, yte=yte,
             classes=np.array(CLASSES))
    with open(os.path.join(SAVE_DIR, "subject_split_3class.txt"), "w") as f:
        for name, sset in zip(["train", "val", "test"], subj_split):
            f.write(f"{name} ({len(sset)} subjects): {sorted(sset)}\n\n")

print("Xtr:", Xtr.shape, "Xval:", Xval.shape, "Xte:", Xte.shape)
print("Train class counts:", np.bincount(ytr, minlength=len(CLASSES)))
print("Val class counts:  ", np.bincount(yval, minlength=len(CLASSES)))
print("Test class counts: ", np.bincount(yte, minlength=len(CLASSES)))

train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval)), batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=BATCH, shuffle=False)


# =========================================================
# STEP 2: Reusable train+eval function for one model
# =========================================================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


def run_model(model_name, model_cls):
    print(f"\n{'='*60}\nTraining {model_name} (3-class: AD vs FTD vs CN)\n{'='*60}")

    model = model_cls(n_channels=Xtr.shape[2], num_classes=len(CLASSES)).to(device)
    dummy = torch.zeros(2, 1, Xtr.shape[2], Xtr.shape[3]).to(device)
    model(dummy)  # materialize LazyLinear

    class_counts = np.bincount(ytr, minlength=len(CLASSES))
    weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
    weights = weights / weights.sum() * len(CLASSES)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state, no_improve = 0.0, None, 0

    t0 = time.time()
    for epoch in range(EPOCHS):
        ep_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        epoch_loss = train_loss / total
        epoch_acc = correct / total
        val_acc = evaluate(model, val_loader)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} | loss={epoch_loss:.4f} | "
              f"train_acc={epoch_acc:.4f} | val_acc={val_acc:.4f} | best={best_val_acc:.4f} | "
              f"{time.time()-ep_start:.1f}s/epoch")

        if no_improve >= PATIENCE:
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break

    print(f"[{model_name}] Total training time: {(time.time()-t0)/60:.1f} min")

    # Load best-on-val checkpoint, touch test set exactly once
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    test_acc = (all_preds == all_labels).mean()

    print(f"[{model_name}] Best val acc: {best_val_acc:.4f} | FINAL test acc: {test_acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES,
                                    digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    print(report)
    print("Confusion matrix:\n", cm)

    tag = model_name.lower()

    # ---- Save weights + npz ----
    torch.save(best_state, os.path.join(SAVE_DIR, f"{tag}_3class_best_model.pt"))
    np.savez(os.path.join(SAVE_DIR, f"{tag}_3class_results.npz"),
             test_acc=test_acc, best_val_acc=best_val_acc,
             train_loss=history["train_loss"], train_acc=history["train_acc"],
             val_acc=history["val_acc"], preds=all_preds, labels=all_labels,
             classes=np.array(CLASSES))

    # ---- Save txt ----
    with open(os.path.join(SAVE_DIR, f"{tag}_3class_results.txt"), "w") as f:
        f.write(f"{model_name} - 3-Class Classification (AD vs FTD vs CN)\n" + "="*55 + "\n\n")
        f.write(f"Train/Val/Test windows: {Xtr.shape[0]}/{Xval.shape[0]}/{Xte.shape[0]}\n")
        f.write(f"Train class counts: {dict(zip(LABEL_NAMES, class_counts.tolist()))}\n\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"FINAL test accuracy (best-on-val checkpoint, test touched once): {test_acc:.4f}\n\n")
        f.write("Classification report:\n" + report + "\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(f"Order: {LABEL_NAMES}\n" + str(cm) + "\n")

    # ---- Confusion matrix figure ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues' if tag == 'eegnet' else 'Greens')
    ax.set_xticks(range(len(LABEL_NAMES))); ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks(range(len(LABEL_NAMES))); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} - Confusion Matrix (3-class)")
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{tag}_3class_confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- Training curves figure ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"]); axes[0].set_title(f"{model_name} Training Loss"); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{model_name} Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{tag}_3class_training_curves.png"), dpi=150)
    plt.close()

    print(f"[{model_name}] Saved .pt, .npz, .txt, and 2 .png figures to {SAVE_DIR}/")

    return {"model_name": model_name, "test_acc": test_acc, "best_val_acc": best_val_acc}


# =========================================================
# STEP 3: Run both models back to back
# =========================================================
results_summary = []
results_summary.append(run_model("EEGNet", EEGNet))
results_summary.append(run_model("DeepConvNet", DeepConvNet))

# =========================================================
# STEP 4: Combined comparison summary
# =========================================================
print(f"\n{'='*60}\nFINAL COMPARISON (3-class: AD vs FTD vs CN)\n{'='*60}")
with open(os.path.join(SAVE_DIR, "comparison_3class_summary.txt"), "w") as f:
    f.write("3-Class Classification Comparison (AD vs FTD vs CN)\n" + "="*55 + "\n\n")
    for r in results_summary:
        line = f"{r['model_name']:15s} best_val_acc={r['best_val_acc']:.4f}  test_acc={r['test_acc']:.4f}"
        print(line)
        f.write(line + "\n")

print(f"\nSaved comparison to {SAVE_DIR}/comparison_3class_summary.txt")
print("Done.")