"""
1-D Gaussian benchmark generator.

Dataset
-------
  P : [N, n_gaussians * 3]
        Interleaved (amplitude A, center c, log10_sigma) per Gaussian.
        Centers are fixed and evenly spaced; only A and sigma vary per sample.
  U : [N, M]
        Sum-of-Gaussians evaluated on a uniform grid of M points in [x_min, x_max].
  x_grid : [M, 1]
        The uniform spatial grid (saved once in the HDF5 file).
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _compute_fields(P: np.ndarray, x_grid: np.ndarray,
                    n_gaussians: int) -> np.ndarray:
    """Vectorised evaluation of sum-of-Gaussians for all N samples.

    Args:
        P:           [N, n_gaussians * 3]  interleaved (A, c, log10_s)
        x_grid:      [M]
        n_gaussians: number of Gaussian components

    Returns:
        U of shape [N, M], float32.
    """
    N, M = P.shape[0], x_grid.shape[0]
    U = np.zeros((N, M), dtype=np.float32)
    for i in range(n_gaussians):
        A = P[:, 3 * i + 0][:, None]                     # [N, 1]
        c = P[:, 3 * i + 1][:, None]
        s = 10 ** P[:, 3 * i + 2][:, None] + 1e-9
        U += A * np.exp(-0.5 * ((x_grid[None, :] - c) / s) ** 2)
    return U


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(cfg: dict, plot: bool = False):
    """Generate the 1-D Gaussian benchmark and save it to HDF5.

    Args:
        cfg:  Full config dict (loaded from config_1d.yaml).
        plot: If True, display 5 random training samples before saving.
    """
    rng = np.random.default_rng(cfg["seed"])
    d = cfg["data"]

    M           = d["M"]
    n_gaussians = d["n_gaussians"]
    x_min, x_max = d["x_min"], d["x_max"]
    x_grid = np.linspace(x_min, x_max, M, dtype=np.float32)

    n_train = d["samples"]["train"]
    n_val   = d["samples"]["val"]
    n_test  = d["samples"]["test"]
    N       = n_train + n_val + n_test

    print(f"[1D] Generating {N} samples "
          f"({n_train} train / {n_val} val / {n_test} test)…")

    # --- Sample parameters ---
    A     = rng.uniform(-1, 1, (N, n_gaussians)).astype(np.float32)
    s_log = np.log10(
        rng.uniform(0.02, 0.25, (N, n_gaussians))
    ).astype(np.float32)

    # Fixed centers mirroring a "fixed anode" physical setup.
    # Every sample shares the same centers; only A and sigma vary.
    centers = np.linspace(x_min + 0.1, x_max - 0.1, n_gaussians,
                          dtype=np.float32)
    c = np.tile(centers, (N, 1))

    # Interleave: [A1, c1, s1, A2, c2, s2, …]  -> [N, n_gaussians * 3]
    P = np.stack([A, c, s_log], axis=2).reshape(N, -1)

    # --- Evaluate fields ---
    print("[1D] Computing ground-truth function values…")
    U = _compute_fields(P, x_grid, n_gaussians)

    if plot:
        _plot_samples(x_grid, U, rng, n_train)

    _save(cfg, x_grid[:, None], P, U, n_train, n_val, n_test)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _plot_samples(x_grid, U, rng, n_train, n=5):
    indices = rng.integers(0, n_train, n)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    for ax, i in zip(axes, indices):
        ax.plot(x_grid, U[i])
        ax.set_ylabel(f"Sample {i}")
    plt.xlabel("x")
    plt.suptitle("1-D benchmark — random training samples")
    plt.tight_layout()
    plt.show()


def _save(cfg, x_grid, P, U, n_train, n_val, n_test):
    """Write everything to HDF5 in the canonical schema."""
    h5_path = cfg["data"]["h5_path"]
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    # Normalization stats from the training split only
    p_mean = P[:n_train].mean(0).astype(np.float32)
    p_std  = (P[:n_train].std(0)  + 1e-9).astype(np.float32)
    u_mean = U[:n_train].mean(0).astype(np.float32)
    u_std  = (U[:n_train].std(0)  + 1e-9).astype(np.float32)

    splits = {
        "train": (0,                  n_train),
        "val":   (n_train,            n_train + n_val),
        "test":  (n_train + n_val,    n_train + n_val + n_test),
    }

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("x_grid", data=x_grid)          # [M, 1]

        stats = f.create_group("stats")
        stats.create_dataset("p_mean", data=p_mean)
        stats.create_dataset("p_std",  data=p_std)
        stats.create_dataset("u_mean", data=u_mean)
        stats.create_dataset("u_std",  data=u_std)

        for split, (start, end) in splits.items():
            print(f"  [1D] Saving {split} ({end - start} samples)…")
            g = f.create_group(split)
            g.create_dataset("P", data=P[start:end], compression="gzip")
            g.create_dataset("U", data=U[start:end], compression="gzip")

    print(f"[1D] Dataset saved → {h5_path}")
