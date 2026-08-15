import os
import numpy as np
from dataloader import read_ds004504


# ============================================================
# Paths
# ============================================================

DATA_DIR = r"C:\Users\mohimaCHAKRABORTY\PycharmProjects\PythonProject8\uav-env\EEGNet_DeepConvNet\dataset"

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Three classes
# A = Alzheimer's disease
# F = Frontotemporal dementia
# C = Healthy control
# ============================================================

classes = ("A", "F", "C")


# ============================================================
# Load and split the EEG data
# ============================================================

Xtr, ytr, Xval, yval, Xte, yte, subj_split = read_ds004504(
    DATA_DIR,
    classes=classes,
    window_sec=4.0,
    overlap=0.5,
    resample_hz=128.0,
    val_size=0.15,
    test_size=0.20,
    seed=42
)


# ============================================================
# Save processed data
# ============================================================

save_path = os.path.join(
    SAVE_DIR,
    "epoched_data_3class.npz"
)

np.savez(
    save_path,
    Xtr=Xtr,
    ytr=ytr,
    Xval=Xval,
    yval=yval,
    Xte=Xte,
    yte=yte,
    classes=np.array(classes)
)

print(f"\nSaved to: {save_path}")

print("Xtr :", Xtr.shape)
print("ytr :", ytr.shape)

print("Xval:", Xval.shape)
print("yval:", yval.shape)

print("Xte :", Xte.shape)
print("yte :", yte.shape)


# ============================================================
# Save subject-wise split
# ============================================================

split_path = os.path.join(
    SAVE_DIR,
    "subject_split_3class.txt"
)

with open(split_path, "w") as f:

    for name, sset in zip(
        ["train", "validation", "test"],
        subj_split
    ):

        f.write(
            f"{name} ({len(sset)} subjects): "
            f"{sorted(sset)}\n\n"
        )

print(f"Subject split saved to: {split_path}")