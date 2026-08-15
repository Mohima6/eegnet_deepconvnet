import os
import numpy as np
import mne

# --- Hardcoded label map (ds004504 subject numbering convention) ---
# sub-001..036 = AD, sub-037..059 = FTD, sub-060..088 = HC/CN
def get_group(subj_num):
    if 1 <= subj_num <= 36:
        return 'A'   # Alzheimer's Disease
    elif 37 <= subj_num <= 59:
        return 'F'   # Frontotemporal Dementia
    elif 60 <= subj_num <= 88:
        return 'C'   # Cognitively Normal / Healthy Control
    else:
        return None


def load_subject_epochs(set_path, window_sec=4.0, overlap=0.5, resample_hz=128.0):
    """
    Loads one subject's continuous EEG (.set file), resamples, and cuts it
    into fixed-length overlapping windows ("epochs" in the EEG sense).
    Returns array of shape (n_windows, n_channels, n_samples_per_window).
    """
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    if resample_hz is not None:
        raw.resample(resample_hz, verbose=False)

    sf = raw.info['sfreq']
    win = int(window_sec * sf)
    step = int(win * (1 - overlap))

    data = raw.get_data()          # (n_channels, n_timepoints)
    n_ch, n_t = data.shape

    epochs, start = [], 0
    while start + win <= n_t:
        epochs.append(data[:, start:start + win])
        start += step

    epochs_arr = np.stack(epochs, 0) if epochs else np.empty((0, n_ch, win))
    return epochs_arr, n_ch, win


def read_ds004504(data_dir, classes=('A', 'F', 'C'), window_sec=4.0, overlap=0.5,
                   resample_hz=128.0, train_frac=0.75, val_frac=0.15, test_frac=0.10,
                   seed=42):
    """
    Loads ds004504 .set files, windows them, and performs a SUBJECT-WISE,
    per-class stratified split at the requested ratio (default 75/15/10).

    classes: e.g. ('A','C') for AD vs HC, or ('A','F','C') for 3-class AD/FTD/CN
             Default is 3-class, per mentor instruction.

    Returns:
        Xtr, ytr, Xval, yval, Xte, yte,
        split_info: dict with subject ID lists per split, n_channels, window_samples
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, \
        "train_frac + val_frac + test_frac must sum to 1.0"

    cls2idx = {c: i for i, c in enumerate(classes)}
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.set'))

    subj_X, subj_y, subj_ids = {}, {}, []
    n_channels, window_samples = None, None

    for fname in files:
        subj_str = fname.split('_')[0]              # e.g. 'sub-012'
        subj_num = int(subj_str.split('-')[1])
        group = get_group(subj_num)
        if group not in classes:
            continue

        fpath = os.path.join(data_dir, fname)
        try:
            ep, n_ch, win = load_subject_epochs(fpath, window_sec, overlap, resample_hz)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {subj_str}: missing companion file (.fdt?) -> {e}")
            continue

        if ep.shape[0] == 0:
            print(f"[WARN] Skipping {subj_str}: no full windows extracted")
            continue

        subj_X[subj_str] = ep
        subj_y[subj_str] = cls2idx[group]
        subj_ids.append(subj_str)
        n_channels, window_samples = n_ch, win  # same for every subject

    if len(subj_ids) == 0:
        raise RuntimeError("No subjects loaded — check data_dir path and .set/.fdt files.")

    # ---- Stratified subject-wise split: shuffle within each class separately ----
    rng = np.random.RandomState(seed)
    by_class = {c: [] for c in classes}
    for s in subj_ids:
        by_class[classes[subj_y[s]]].append(s)

    train_subs, val_subs, test_subs = [], [], []
    for c, subs in by_class.items():
        subs = subs.copy()
        rng.shuffle(subs)
        n = len(subs)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        n_train = n - n_test - n_val
        if n_train < 1:
            # tiny class edge case: guarantee at least 1 train subject
            n_train = 1
            n_val = max(0, n - n_train - n_test)

        test_subs += subs[:n_test]
        val_subs += subs[n_test:n_test + n_val]
        train_subs += subs[n_test + n_val:n_test + n_val + n_train]

        print(f"  Class {CLASS_LABEL_HELPER(c)}: total={n} -> train={n_train}, val={n_val}, test={n_test}")

    def build(sset):
        sset = sorted(sset)
        X = np.concatenate([subj_X[s] for s in sset], 0)
        y = np.concatenate([np.full(subj_X[s].shape[0], subj_y[s]) for s in sset], 0)
        return X, y

    Xtr, ytr = build(train_subs)
    Xval, yval = build(val_subs)
    Xte, yte = build(test_subs)

    # z-score normalize per channel, using TRAIN stats only (no leakage)
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xval = (Xval - mean) / std
    Xte = (Xte - mean) / std

    Xtr = np.expand_dims(Xtr, 1).astype(np.float32)    # -> (N, 1, C, T)
    Xval = np.expand_dims(Xval, 1).astype(np.float32)
    Xte = np.expand_dims(Xte, 1).astype(np.float32)

    print(f"\nClasses: {classes}")
    print(f"Channels: {n_channels} | Window length: {window_samples} samples "
          f"({window_sec}s @ {resample_hz}Hz)")
    print(f"Train subjects: {len(train_subs)} -> {Xtr.shape[0]} windows")
    print(f"Val subjects:   {len(val_subs)} -> {Xval.shape[0]} windows")
    print(f"Test subjects:  {len(test_subs)} -> {Xte.shape[0]} windows")

    split_info = {
        "train_subjects": sorted(train_subs),
        "val_subjects": sorted(val_subs),
        "test_subjects": sorted(test_subs),
        "n_channels": n_channels,
        "window_samples": window_samples,
        "window_sec": window_sec,
        "resample_hz": resample_hz,
        "classes": classes,
    }

    return Xtr, ytr.astype(np.int64), Xval, yval.astype(np.int64), Xte, yte.astype(np.int64), split_info


def CLASS_LABEL_HELPER(c):
    return {'A': 'AD', 'F': 'FTD', 'C': 'CN'}.get(c, c)