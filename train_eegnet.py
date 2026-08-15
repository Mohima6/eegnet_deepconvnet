import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from models import EEGNet
def seed_everything(seed=500):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(500)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
SAVE_DIR = "outputs"
FIG_DIR = os.path.join(SAVE_DIR, "figures")
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints_eegnet")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
# (75/15/10 subject-wise split)
data = np.load(os.path.join(SAVE_DIR, "epoched_data.npz"), allow_pickle=True)
Xtr, ytr = data['Xtr'], data['ytr']
Xval, yval = data['Xval'], data['yval']
Xte, yte = data['Xte'], data['yte']
classes = list(data['classes'])
n_channels = int(data['n_channels'])
window_samples = int(data['window_samples'])
class_names = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
label_names = [class_names.get(c, c) for c in classes]
print("Classes:", classes, "->", label_names)
print("Xtr:", Xtr.shape, "Xval:", Xval.shape, "Xte:", Xte.shape)
print(f"Channels: {n_channels} | Window samples: {window_samples}")
BATCH = 32
train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval)), batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=BATCH, shuffle=False)
model = EEGNet(n_channels=n_channels, num_classes=len(classes), dropout=0.4).to(device)
dummy = torch.zeros(2, 1, n_channels, window_samples).to(device)
model(dummy)  
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"EEGNet trainable parameters: {n_params:,}")
# Class weights (imbalance handling) 
class_counts = np.bincount(ytr, minlength=len(classes))
weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
weights = weights / weights.sum() * len(classes)
print("Train class counts:", dict(zip(label_names, class_counts.tolist())))
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / total, correct / total
# Training loop: val-based checkpointing, early stopping, periodic saves 
EPOCHS = 60
PATIENCE = 12
CKPT_EVERY = 5   # save a checkpoint every 5 epochs
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
checkpoint_log = []  # (epoch, val_acc) for every saved checkpoint
best_val_acc = 0.0
best_epoch = -1
best_state = None
no_improve = 0
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    ep_t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    if epoch % CKPT_EVERY == 0 or epoch == EPOCHS:
        ckpt_path = os.path.join(CKPT_DIR, f"eegnet_epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        checkpoint_log.append((epoch, val_acc, tr_acc, val_loss))
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | best_val={best_val_acc:.4f}@ep{best_epoch} | "
          f"{time.time()-ep_t0:.1f}s")
    if no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
        break
print(f"Total training time: {(time.time()-t0)/60:.1f} min")
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
print(f"\n[EEGNet] Best epoch: {best_epoch} | Best val_acc: {best_val_acc:.4f} | "
      f"FINAL test_acc (test touched once): {test_acc:.4f}")
report = classification_report(all_labels, all_preds, target_names=label_names, digits=4, zero_division=0)
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
print(report)
print("Confusion matrix:\n", cm)
torch.save(best_state, os.path.join(SAVE_DIR, "eegnet_best_model.pt"))
np.savez(os.path.join(SAVE_DIR, "eegnet_results.npz"),
         test_acc=test_acc, best_val_acc=best_val_acc, best_epoch=best_epoch,
         train_loss=history["train_loss"], train_acc=history["train_acc"],
         val_loss=history["val_loss"], val_acc=history["val_acc"],
         preds=all_preds, labels=all_labels, classes=np.array(classes))
with open(os.path.join(SAVE_DIR, "eegnet_results.txt"), "w") as f:
    f.write("EEGNet — 3-Class Classification (AD vs FTD vs CN)\n" + "=" * 55 + "\n\n")
    f.write(f"Split: 75% train / 15% val / 10% test (subject-wise, stratified per class)\n")
    f.write(f"Channels: {n_channels} | Window: {window_samples} samples\n")
    f.write(f"Trainable parameters: {n_params:,}\n")
    f.write(f"Windows -> train={Xtr.shape[0]}, val={Xval.shape[0]}, test={Xte.shape[0]}\n\n")
    f.write(f"Trained for {len(history['train_loss'])} epochs (early stopping patience={PATIENCE})\n")
    f.write(f"Best epoch: {best_epoch} | Best validation accuracy: {best_val_acc:.4f}\n")
    f.write(f"FINAL test accuracy (best-on-val checkpoint, test set touched exactly once): {test_acc:.4f}\n\n")
    f.write("Classification report (test set):\n" + report + "\n\n")
    f.write(f"Confusion matrix (order: {label_names}):\n" + str(cm) + "\n\n")
    f.write("Checkpoint log (epoch, val_acc, train_acc, val_loss):\n")
    for ep, va, ta, vl in checkpoint_log:
        f.write(f"  epoch {ep:3d} | val_acc={va:.4f} | train_acc={ta:.4f} | val_loss={vl:.4f}\n")
# Figure 1: train/val loss + accuracy curves 
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.5, label=f"best epoch ({best_epoch})")
axes[0].set_title("EEGNet — Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.5, label=f"best epoch ({best_epoch})")
axes[1].set_title("EEGNet — Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eegnet_train_val_curves.png"), dpi=150)
plt.close(fig)
# Figure 2: confusion matrix 
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(label_names))); ax.set_xticklabels(label_names)
ax.set_yticks(range(len(label_names))); ax.set_yticklabels(label_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"EEGNet — Confusion Matrix (test acc={test_acc:.3f})")
for i in range(len(label_names)):
    for j in range(len(label_names)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eegnet_confusion_matrix.png"), dpi=150)
plt.close(fig)
# Figure 3: checkpoint log — epoch vs val_acc, with data shape annotation 
fig, ax = plt.subplots(figsize=(8, 4.5))
ckpt_epochs = [c[0] for c in checkpoint_log]
ckpt_vals = [c[1] for c in checkpoint_log]
ax.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], color='gray', alpha=0.5, label="val_acc (every epoch)")
ax.scatter(ckpt_epochs, ckpt_vals, color='blue', zorder=5, label=f"checkpoints saved (every {CKPT_EVERY} epochs)")
ax.scatter([best_epoch], [best_val_acc], color='red', zorder=6, s=100, marker='*', label=f"best checkpoint (epoch {best_epoch})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy")
ax.set_title(f"EEGNet — Checkpoints | {n_channels} channels, window={window_samples} samples")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eegnet_checkpoints.png"), dpi=150)
plt.close(fig)
print(f"\nSaved: eegnet_results.txt, eegnet_best_model.pt, "
      f"{CKPT_DIR}/ (periodic checkpoints), and 3 figures in {FIG_DIR}/")
