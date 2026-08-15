import os
import numpy as np
import matplotlib.pyplot as plt

from dataloader import read_ds004504

DATA_DIR = r"C:\Users\mohimaCHAKRABORTY\PycharmProjects\PythonProject8\uav-env\EEGNet_DeepConvNet\dataset"
SAVE_DIR = "outputs"
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

classes = ('A', 'F', 'C')  # 3-class: AD, FTD, CN — per mentor instruction

Xtr, ytr, Xval, yval, Xte, yte, split_info = read_ds004504(
    DATA_DIR, classes=classes,
    window_sec=4.0, overlap=0.5, resample_hz=128.0,
    train_frac=0.75, val_frac=0.15, test_frac=0.10
)


save_path = os.path.join(SAVE_DIR, "epoched_data.npz")
np.savez(
    save_path,
    Xtr=Xtr, ytr=ytr,
    Xval=Xval, yval=yval,
    Xte=Xte, yte=yte,
    classes=np.array(classes),
    n_channels=split_info["n_channels"],
    window_samples=split_info["window_samples"],
    window_sec=split_info["window_sec"],
    resample_hz=split_info["resample_hz"],
    train_subjects=np.array(split_info["train_subjects"]),
    val_subjects=np.array(split_info["val_subjects"]),
    test_subjects=np.array(split_info["test_subjects"]),
)

print(f"Saved epoched data to {save_path}")
print("Xtr:", Xtr.shape, "ytr:", ytr.shape)
print("Xval:", Xval.shape, "yval:", yval.shape)
print("Xte:", Xte.shape, "yte:", yte.shape)
print("Train subjects:", len(split_info["train_subjects"]))
print("Val subjects:  ", len(split_info["val_subjects"]))
print("Test subjects: ", len(split_info["test_subjects"]))


splits_txt = os.path.join(SAVE_DIR, "subject_splits.txt")
with open(splits_txt, "w") as f:
    f.write("Train subjects:\n" + "\n".join(split_info["train_subjects"]) + "\n\n")
    f.write("Val subjects:\n" + "\n".join(split_info["val_subjects"]) + "\n\n")
    f.write("Test subjects:\n" + "\n".join(split_info["test_subjects"]) + "\n")
print(f"Saved subject split lists to {splits_txt}")


def class_counts(y, n_classes):
    return np.bincount(y, minlength=n_classes)

train_counts = class_counts(ytr, len(classes))
val_counts   = class_counts(yval, len(classes))
test_counts  = class_counts(yte, len(classes))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: class distribution per split (grouped bar chart) 
x = np.arange(len(classes))
width = 0.25
axes[0].bar(x - width, train_counts, width, label="Train")
axes[0].bar(x, val_counts, width, label="Val")
axes[0].bar(x + width, test_counts, width, label="Test")
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes)
axes[0].set_ylabel("Number of windows (epochs)")
axes[0].set_title("Class distribution across splits")
axes[0].legend()

# --- Panel 2: subject counts per split, per class ---
def subj_class_counts(subj_list, split_info_key=None):
    from dataloader import get_group
    counts = {c: 0 for c in classes}
    for s in subj_list:
        num = int(s.split('-')[1])
        g = get_group(num)
        if g in counts:
            counts[g] += 1
    return counts

train_sc = subj_class_counts(split_info["train_subjects"])
val_sc   = subj_class_counts(split_info["val_subjects"])
test_sc  = subj_class_counts(split_info["test_subjects"])

axes[1].bar(x - width, [train_sc[c] for c in classes], width, label="Train")
axes[1].bar(x, [val_sc[c] for c in classes], width, label="Val")
axes[1].bar(x + width, [test_sc[c] for c in classes], width, label="Test")
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].set_ylabel("Number of subjects")
axes[1].set_title("Subject counts across splits")
axes[1].legend()

#  Panel 3: example epoch waveform, all channels, one window 
example = Xtr[0, 0]  # shape (n_channels, window_samples)
n_ch_plot = min(example.shape[0], 19)
offset = np.arange(n_ch_plot)[:, None] * (np.std(example) * 6 + 1e-6)
axes[2].plot((example[:n_ch_plot] + offset).T, linewidth=0.6)
axes[2].set_title(f"Example epoch — {n_ch_plot} channels, "
                   f"{split_info['window_samples']} samples "
                   f"({split_info['window_sec']}s @ {split_info['resample_hz']}Hz)")
axes[2].set_xlabel("Sample")
axes[2].set_yticks([])

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "data_summary.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"Saved data summary figure to {fig_path}")
