import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter

from models import EEGNet

def seed_everything(seed=500):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(500)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

SAVE_DIR = "outputs"

# ---- Load pre-epoched data (fast, no MNE) ----
data = np.load(os.path.join(SAVE_DIR, "epoched_data.npz"), allow_pickle=True)
Xtr, ytr, Xte, yte = data['Xtr'], data['ytr'], data['Xte'], data['yte']
classes = tuple(data['classes'])
print("Loaded classes:", classes)
print("Xtr:", Xtr.shape, "ytr:", ytr.shape)
print("Xte:", Xte.shape, "yte:", yte.shape)

train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
test_ds  = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# ---- Model ----
model = EEGNet(n_channels=Xtr.shape[2], num_classes=len(classes)).to(device)
dummy = torch.zeros(2, 1, Xtr.shape[2], Xtr.shape[3]).to(device)
model(dummy)  # materialize LazyLinear before creating optimizer

# ---- Class weights ----
class_counts = np.bincount(ytr, minlength=len(classes))
weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
weights = weights / weights.sum() * len(classes)
print("Class counts (train):", class_counts, "-> weights:", weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# ---- Training loop ----
EPOCHS = 100
history = {"train_loss": [], "train_acc": []}

for epoch in range(EPOCHS):
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
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    epoch_loss = train_loss / total
    epoch_acc = correct / total
    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(epoch_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={epoch_loss:.4f} | train_acc={epoch_acc:.4f}")

# ---- Test evaluation ----
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

test_acc = correct / total
print(f"\nTest accuracy: {test_acc:.4f}  ({total} epochs from held-out subjects)")
print("Predicted class distribution:", Counter(all_preds))
print("True class distribution:     ", Counter(all_labels))

# ---- Save model + results ----
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "eegnet_weights.pt"))
np.savez(os.path.join(SAVE_DIR, "eegnet_results.npz"),
         test_acc=test_acc, train_loss=history["train_loss"],
         train_acc=history["train_acc"],
         preds=np.array(all_preds), labels=np.array(all_labels))
print(f"Saved EEGNet model + results to {SAVE_DIR}/")