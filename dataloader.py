import os
import numpy as np
import mne

# --- Hardcoded label map (no participants.tsv available) ---
# ds004504: sub-001..036 = AD, sub-037..059 = FTD, sub-060..088 = HC
def get_group(subj_num):
    if 1 <= subj_num <= 36:
        return 'A'   # Alzheimer's
    elif 37 <= subj_num <= 59:
        return 'F'   # FTD
    elif 60 <= subj_num <= 88:
        return 'C'   # Healthy control
    else:
        return None


def load_subject_epochs(set_path, window_sec=4.0, overlap=0.5, resample_hz=128.0):
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    if resample_hz is not None:
        raw.resample(resample_hz, verbose=False)

    sf = raw.info['sfreq']
    win = int(window_sec * sf)
    step = int(win * (1 - overlap))

    data = raw.get_data()          # (channels, n_times)
    n_ch, n_t = data.shape

    epochs, start = [], 0
    while start + win <= n_t:
        epochs.append(data[:, start:start + win])
        start += step

    return np.stack(epochs, 0) if epochs else np.empty((0, n_ch, win))


def read_ds004504(data_dir, classes=('A', 'C'), window_sec=4.0, overlap=0.5,
                   resample_hz=128.0, test_size=0.2, seed=42):
    """
    data_dir: folder containing sub-XXX_task-eyesclosed_eeg.set files (flat, no BIDS subfolders)
    classes: tuple of group codes to include, e.g. ('A','C') for AD vs HC,
             or ('A','F','C') for 3-class
    """
    cls2idx = {c: i for i, c in enumerate(classes)}

    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.set'))

    subj_X, subj_y, subj_ids = {}, {}, []
    for fname in files:
        # fname like: sub-012_task-eyesclosed_eeg.set
        subj_str = fname.split('_')[0]            # 'sub-012'
        subj_num = int(subj_str.split('-')[1])     # 12

        group = get_group(subj_num)
        if group not in classes:
            continue

        fpath = os.path.join(data_dir, fname)
        try:
            ep = load_subject_epochs(fpath, window_sec, overlap, resample_hz)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {subj_str}: missing companion file (.fdt?) -> {e}")
            continue

        if ep.shape[0] == 0:
            print(f"[WARN] Skipping {subj_str}: no full windows extracted")
            continue

        subj_X[subj_str] = ep
        subj_y[subj_str] = cls2idx[group]
        subj_ids.append(subj_str)

    if len(subj_ids) == 0:
        raise RuntimeError("No subjects loaded — check data_dir path and .set/.fdt files.")

    # subject-wise train/test split (NOT epoch-wise, to avoid leakage)
    rng = np.random.RandomState(seed)
    rng.shuffle(subj_ids)
    n_test = max(1, int(len(subj_ids) * test_size))
    test_subs = set(subj_ids[:n_test])
    train_subs = set(subj_ids[n_test:])

    def build(sset):
        X = np.concatenate([subj_X[s] for s in sset], 0)
        y = np.concatenate([np.full(subj_X[s].shape[0], subj_y[s]) for s in sset], 0)
        return X, y

    Xtr, ytr = build(train_subs)
    Xte, yte = build(test_subs)

    # z-score normalize per channel, fit stats on train only
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    Xtr = np.expand_dims(Xtr, 1).astype(np.float32)   # (N,1,C,T)
    Xte = np.expand_dims(Xte, 1).astype(np.float32)

    print(f"Train subjects: {len(train_subs)} -> {Xtr.shape[0]} epochs")
    print(f"Test subjects:  {len(test_subs)} -> {Xte.shape[0]} epochs")

    return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64)