"""
Unified HDF5 dataset for all benchmarks (Gaussian 1D/3D + Alucell).

Both the benchmark generators and prepare_alucell.py write the same core:
  x_grid          : [M, spatial_dim]
  stats/p_mean    : [n_params]
  stats/p_std     : [n_params]
  stats/u_mean    : [M_out]
  stats/u_std     : [M_out]
  {train,val,test}/P : [N, n_params]
  {train,val,test}/U : [N, M_out]

Alucell additionally stores:
  meta/                    attrs: mapping, delta_learning, field_shape, …
  reconstruction/u_ref     [M_field]         (if delta-learning)
  reconstruction/V         [M_field, k]      (if velocity_full_pod)
  reconstruction/u_pod_mean [M_field]        (if velocity_full_pod)
  {split}/modes            [N]  string       (campaign type labels)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    """Lazy-loading HDF5 dataset — benchmark-agnostic.

    Thread-safe with num_workers > 0: the file is re-opened per
    __getitem__ call.
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


# ======================================================================
# Reconstruction helpers (used by evaluate.py)
# ======================================================================

def load_reconstruction_context(h5_path: str) -> dict:
    """Load all metadata needed to map model outputs back to physical fields.

    Returns a dict with keys:
        mapping          : str
        delta_learning   : bool
        field_shape      : tuple
        u_ref            : np.ndarray or None     [M_field]
        V                : np.ndarray or None     [M_field, k]
        u_pod_mean       : np.ndarray or None     [M_field]
    """
    ctx = {}
    with h5py.File(h5_path, "r") as f:
        meta = f.get("meta")
        if meta is not None:
            ctx["mapping"]        = meta.attrs.get("mapping", "unknown")
            ctx["delta_learning"] = bool(meta.attrs.get("delta_learning", False))
            ctx["field_shape"]    = tuple(meta.attrs.get("field_shape", []))
        else:
            ctx["mapping"]        = "unknown"
            ctx["delta_learning"] = False
            ctx["field_shape"]    = ()

        recon = f.get("reconstruction")
        ctx["u_ref"]      = recon["u_ref"][:] if (recon and "u_ref" in recon) else None
        ctx["V"]          = recon["V"][:] if (recon and "V" in recon) else None
        ctx["u_pod_mean"] = recon["u_pod_mean"][:] if (recon and "u_pod_mean" in recon) else None

    return ctx


def reconstruct_field(model_output: np.ndarray, ctx: dict) -> np.ndarray:
    """Map model predictions back to the physical field.

    Args:
        model_output:  [N, M_out]  raw model output (delta or POD coeffs).
        ctx:           dict from load_reconstruction_context().

    Returns:
        [N, M_field]   reconstructed physical field.
    """
    out = model_output.copy()

    # Step 1: if POD coefficients → decode to full field
    if ctx["V"] is not None:
        V          = ctx["V"]                   # [M_field, k]
        u_pod_mean = ctx["u_pod_mean"]          # [M_field]
        out = out @ V.T + u_pod_mean[None, :]   # [N, M_field]

    # Step 2: if delta-learning → add reference
    if ctx["delta_learning"] and ctx["u_ref"] is not None:
        out = out + ctx["u_ref"][None, :]

    return out


def load_test_modes(h5_path: str) -> list[str]:
    """Load the campaign-mode labels for the test split."""
    with h5py.File(h5_path, "r") as f:
        if "test/modes" in f:
            raw = f["test"]["modes"][:]
            return [m.decode() if isinstance(m, bytes) else str(m) for m in raw]
    return []