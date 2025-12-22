import h5py
import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    def __init__(self, h5_path, split="train"):
        self.h5_path = h5_path
        self.split = split

        # Open once just to get length and stats
        with h5py.File(h5_path, "r") as f:
            self.length = len(f[split]["P"])
            self.p_mean = torch.from_numpy(f["stats"]["p_mean"][:])
            self.p_std = torch.from_numpy(f["stats"]["p_std"][:])
            self.u_mean = torch.from_numpy(f["stats"]["u_mean"][:])
            self.u_std = torch.from_numpy(f["stats"]["u_std"][:])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file inside getitem to ensure thread safety with NumWorkers > 0
        with h5py.File(self.h5_path, "r") as f:
            p_raw = torch.from_numpy(f[self.split]["P"][idx])
            u_raw = torch.from_numpy(f[self.split]["U"][idx])

        # Normalize
        p = (p_raw - self.p_mean) / self.p_std
        u = (u_raw - self.u_mean) / self.u_std

        return p, u
