"""Optional EEG preprocessing/augmentation for the Lab2 V2 exploration.

Requires the ORIGINAL dataloader.py in the same folder. No original file
is overwritten. Subject-wise normalization is fit on training data only.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataloader import build_loaders


class TrainJitter(Dataset):
    def __init__(self, base: TensorDataset, max_shift: int = 10):
        self.base = base
        self.max_shift = max_shift

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        # Shift BOTH EEG channels together and replicate the edge values.
        # Never use circular wrap-around: it invents a false end/start join.
        s = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        if s == 0:
            return x, y
        out = torch.empty_like(x)
        if s > 0:
            out[..., s:] = x[..., :-s]
            out[..., :s] = x[..., :1]
        else:
            k = -s
            out[..., :-k] = x[..., k:]
            out[..., -k:] = x[..., -1:]
        return out, y


def build_loaders_v2(data_dir, batch_size, val_fraction=0.15, seed=42,
                     num_workers=0, pin_memory=False, subject_norm=False,
                     time_jitter=0):
    loaders, counts = build_loaders(
        data_dir, batch_size=batch_size, val_fraction=val_fraction, seed=seed,
        num_workers=num_workers, pin_memory=pin_memory, normalize=False,
    )
    split_data = {}
    for split, loader in loaders.items():
        # Original class/subj stratification builds a sorted-by-source order:
        # S4b observations first, followed by X11b observations.
        original_ds = loader.dataset
        xx, yy = original_ds.tensors
        xx = xx.clone()
        split_data[split] = [xx, yy]
        assert len(xx) == counts[split]['S4b'] + counts[split]['X11b']

    if subject_norm:
        original_train = split_data['train'][0]
        start, end = 0, counts['train']['S4b']
        stats = {}
        for subject, tensor in [('S4b', original_train[start:end]),
                                ('X11b', original_train[end:])]:
            # Input [N,1,2,750], independent statistics for the two EEG channels.
            mean = tensor.mean(dim=(0, 3), keepdim=True)
            std = tensor.std(dim=(0, 3), keepdim=True).clamp_min(1e-5)
            stats[subject] = (mean, std)
        for split, (xx, yy) in split_data.items():
            s4len = counts[split]['S4b']
            for subject, sl in [('S4b', slice(0, s4len)),
                                ('X11b', slice(s4len, None))]:
                mean, std = stats[subject]
                xx[sl] = (xx[sl] - mean) / std
            if not torch.isfinite(xx).all():
                raise ValueError(f'Non-finite values after subject normalization: {split}')

    new_loaders = {}
    for split, (xx, yy) in split_data.items():
        ds = TensorDataset(xx.contiguous(), yy)
        if split == 'train' and time_jitter:
            ds = TrainJitter(ds, max_shift=time_jitter)
        gen = torch.Generator().manual_seed(seed) if split == 'train' else None
        new_loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=split == 'train',
            num_workers=num_workers, pin_memory=pin_memory, generator=gen,
        )
    return new_loaders, counts
