from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SUBJECTS = ("S4b", "X11b")


def _read_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing EEG data: {path}")
    with np.load(path) as archive:
        x = np.asarray(archive["signal"], dtype=np.float32)
        y = np.asarray(archive["label"], dtype=np.int64) - 1
    if x.ndim != 3 or x.shape[1:] != (750, 2):
        raise ValueError(f"Bad shape in {path.name}: {x.shape}; expected [N,750,2]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"Labels in {path.name} should be 1 or 2 before conversion.")
    return x, y


def _split_stratified(train_labels: np.ndarray, subjects: np.ndarray, fraction: float, seed: int):

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for subject in np.unique(subjects):
        for label in (0, 1):
            candidates = np.where((subjects == subject) & (train_labels == label))[0]
            candidates = rng.permutation(candidates)
            n_val = min(len(candidates) - 1, max(1, int(round(fraction * len(candidates)))))
            val_idx.extend(candidates[:n_val])
            train_idx.extend(candidates[n_val:])
    return np.asarray(sorted(train_idx)), np.asarray(sorted(val_idx))


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize: bool = False,
):
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    root = Path(data_dir)
    tr_x, tr_y, tr_s, te_x, te_y, te_s = [], [], [], [], [], []
    for subject in SUBJECTS:
        x, y = _read_file(root / f"{subject}_train.npz")
        tr_x.append(x)
        tr_y.append(y)
        tr_s.append(np.repeat(subject, len(y)))
        x, y = _read_file(root / f"{subject}_test.npz")
        te_x.append(x)
        te_y.append(y)
        te_s.append(np.repeat(subject, len(y)))
    all_train_x = np.concatenate(tr_x, axis=0)
    all_train_y = np.concatenate(tr_y)
    all_train_s = np.concatenate(tr_s)
    test_x = np.concatenate(te_x, axis=0)
    test_y = np.concatenate(te_y)
    test_s = np.concatenate(te_s)
    tr_idx, va_idx = _split_stratified(all_train_y, all_train_s, val_fraction, seed)
    arrays = {
        "train": (all_train_x[tr_idx].copy(), all_train_y[tr_idx], all_train_s[tr_idx]),
        "val": (all_train_x[va_idx].copy(), all_train_y[va_idx], all_train_s[va_idx]),
        "test": (test_x.copy(), test_y, test_s),
    }

    fit_x = arrays["train"][0]
    channel_means = np.nanmean(fit_x, axis=(0, 1))
    channel_means = np.nan_to_num(channel_means, nan=0.0)
    for split, (x, y, subjects) in arrays.items():
        n_nan = int(np.isnan(x).sum())
        if n_nan:
            for ch in range(2):
                x[:, :, ch] = np.nan_to_num(x[:, :, ch], nan=float(channel_means[ch]))
            print(f"[{split}] Replaced {n_nan} NaNs with train-derived channel means.")
        arrays[split] = (x, y, subjects)

    if normalize:
        fit_x = arrays["train"][0]
        mean = fit_x.mean(axis=(0, 1))
        std = fit_x.std(axis=(0, 1))
        std = np.maximum(std, 1e-6)
        for split, (x, y, subjects) in arrays.items():
            arrays[split] = ((x - mean) / std, y, subjects)

    dataloaders, counts = {}, {}
    for split, (x, y, subjects) in arrays.items():
        x = np.ascontiguousarray(x.transpose(0, 2, 1)[:, None, :, :], dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(np.asarray(y, dtype=np.int64)))
        gen = torch.Generator().manual_seed(seed) if split == "train" else None
        dataloaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=pin_memory, generator=gen,
        )
        counts[split] = {
            "total": len(y), "class_0": int((y == 0).sum()), "class_1": int((y == 1).sum()),
            "S4b": int((subjects == "S4b").sum()), "X11b": int((subjects == "X11b").sum()),
        }
    return dataloaders, counts


def read_bci_data(data_dir: str | Path = "data"):
    xs, ys = [], []
    for split in ("train", "test"):
        xx, yy = [], []
        for subject in SUBJECTS:
            x, y = _read_file(Path(data_dir) / f"{subject}_{split}.npz")
            xx.append(x)
            yy.append(y)
        xs.append(np.concatenate(xx))
        ys.append(np.concatenate(yy))
    train_mean = np.nanmean(xs[0], axis=(0, 1))
    for split_index, x in enumerate(xs):
        for ch in range(2):
            x[:, :, ch] = np.nan_to_num(x[:, :, ch], nan=float(train_mean[ch]))
        xs[split_index] = x.transpose(0, 2, 1)[:, None, :, :]
    return xs[0], ys[0], xs[1], ys[1]
