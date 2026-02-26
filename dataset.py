"""
Unified HDF5 dataset for all benchmarks.

Both the 1-D and 3-D generators write the same schema:
  x_grid          : [M, spatial_dim]
  stats/p_mean    : [n_params]
  stats/p_std     : [n_params]
  stats/u_mean    : [M]
  stats/u_std     : [M]
  {train,val,test}/P : [N, n_params]
  {train,val,test}/U : [N, M]

The Dataset class is benchmark-agnostic — it just reads P and U.
"""

import h5py
import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    """Lazy-loading HDF5 dataset.

    Thread-safe with num_workers > 0: the file is re-opened per __getitem__
    call rather than held open on the object.

    Args:
        h5_path:   Path to the HDF5 file created by generate.py.
        split:     One of "train", "val", "test".
        normalize: If True, z-score normalise P and U using train-split stats.
                   Disabled by default because sharp-peaked functions have a
                   near-zero global mean, which distorts the z-score.
    """

    def __init__(self, h5_path: str, split: str = "train",
                 normalize: bool = False):
        self.h5_path   = h5_path
        self.split     = split
        self.normalize = normalize

        with h5py.File(h5_path, "r") as f:
            self.length = len(f[split]["P"])
            self.p_mean = torch.from_numpy(f["stats"]["p_mean"][:])
            self.p_std  = torch.from_numpy(f["stats"]["p_std"][:])
            self.u_mean = torch.from_numpy(f["stats"]["u_mean"][:])
            self.u_std  = torch.from_numpy(f["stats"]["u_std"][:])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            p = torch.from_numpy(f[self.split]["P"][idx])
            u = torch.from_numpy(f[self.split]["U"][idx])

        if self.normalize:
            p = (p - self.p_mean) / self.p_std
            u = (u - self.u_mean) / self.u_std

        return p, u
