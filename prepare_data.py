import os
import numpy as np
from dataloader import read_ds004504

DATA_DIR = r"C:\Users\mohimaCHAKRABORTY\PycharmProjects\PythonProject8\uav-env\EEGNet_DeepConvNet\dataset"
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

classes = ('A', 'C')  # AD vs HC; change to ('A','F','C') for 3-class later

Xtr, ytr, Xte, yte = read_ds004504(
    DATA_DIR, classes=classes,
    window_sec=4.0, overlap=0.5, resample_hz=128.0
)

save_path = os.path.join(SAVE_DIR, "epoched_data.npz")
np.savez(save_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
         classes=np.array(classes))

print(f"Saved epoched data to {save_path}")
print("Xtr:", Xtr.shape, "ytr:", ytr.shape)
print("Xte:", Xte.shape, "yte:", yte.shape)