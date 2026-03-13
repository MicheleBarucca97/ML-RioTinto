"""
3-D Gaussian benchmark generator.

Dataset
-------
  P : [N, n_gaussians * 2]
        Concatenated (amplitudes A, spreads sigma) — one value per Gaussian.
        Gaussian centers are fixed at generation time and shared across all
        samples, so they are NOT part of the input vector P.
  U : [N, M]
        Sum-of-Gaussians evaluated at M = grid_res^3 mesh nodes.
  x_grid : [M, 3]
        Flattened 3-D mesh coordinates (saved once in the HDF5 file).

Notes
-----
  The inner loop over samples has been replaced by a fully vectorised
  NumPy computation, which is ~100x faster for large N.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _build_mesh(grid_res: int) -> np.ndarray:
    """Return an [M, 3] float32 array of mesh node coordinates in [-1, 1]^3.

    M = grid_res^3.
    """
    ax = np.linspace(-1, 1, grid_res, dtype=np.float32)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)   # [M, 3]


def _compute_fields(A: np.ndarray, sigma: np.ndarray,
                    coords: np.ndarray,
                    centers: np.ndarray) -> np.ndarray:
    """Vectorised sum-of-3-D-Gaussians for all N samples.

    Args:
        A:       [N, G]   amplitudes
        sigma:   [N, G]   isotropic spreads
        coords:  [M, 3]   mesh node positions
        centers: [G, 3]   fixed Gaussian centers

    Returns:
        U of shape [N, M], float32.
    """
    N, G = A.shape
    M    = coords.shape[0]

    # Squared distances: [G, M]
    # coords[None] - centers[:, None] → [G, M, 3] → sum over last dim → [G, M]
    diff  = coords[None, :, :] - centers[:, None, :]    # [G, M, 3]
    r2    = (diff ** 2).sum(axis=-1)                     # [G, M]

    # Gaussian activations per source: [G, M]
    # sigma: [N, G] → need [N, G, M] for broadcasting
    # A:     [N, G] → same

    # Expand dims for broadcasting
    r2_exp    = r2[None, :, :]                           # [1, G, M]
    sigma_exp = sigma[:, :, None]                        # [N, G, 1]
    A_exp     = A[:, :, None]                            # [N, G, 1]

    activations = A_exp * np.exp(-r2_exp / (2 * sigma_exp ** 2))  # [N, G, M]
    return activations.sum(axis=1).astype(np.float32)              # [N, M]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(cfg: dict, plot: bool = False):
    """Generate the 3-D Gaussian benchmark and save it to HDF5.

    Args:
        cfg:  Full config dict (loaded from config_3d.yaml).
        plot: If True, display a z=0 slice of 5 random training samples.
    """
    rng = np.random.default_rng(cfg["seed"])
    d = cfg["data"]

    grid_res    = d["grid_res"]
    n_gaussians = d["n_gaussians"]
    n_train     = d["samples"]["train"]
    n_val       = d["samples"]["val"]
    n_test      = d["samples"]["test"]
    N           = n_train + n_val + n_test

    print(f"[3D] Generating {N} samples "
          f"({n_train} train / {n_val} val / {n_test} test)…")
    print(f"[3D] Grid: {grid_res}^3 = {grid_res**3} nodes")

    # --- Mesh ---
    coords = _build_mesh(grid_res)                       # [M, 3]
    M = coords.shape[0]

    # --- Fixed Gaussian centers (seeded separately for reproducibility) ---
    centers_rng = np.random.default_rng(cfg["seed"] + 999)
    centers = centers_rng.uniform(-0.8, 0.8,
                                  (n_gaussians, 3)).astype(np.float32)

    # --- Sample per-sample parameters ---
    n_varying_dims = 2 * n_gaussians 
    sampler = qmc.Sobol(d=n_varying_dims, scramble=True, seed=cfg["seed"])
    sobol_samples = sampler.random(n=N)
    sobol_A = sobol_samples[:, :n_gaussians]
    sobol_s = sobol_samples[:, n_gaussians:]
    # Scale from [0, 1] bounds to your physical bounds using qmc.scale
    A = qmc.scale(sobol_A, 0.1, 1.0).astype(np.float32)
    sigma = qmc.scale(sobol_s, 0.1, 0.3).astype(np.float32)
    '''A     = rng.uniform(0.1, 1.0, (N, n_gaussians)).astype(np.float32)
    sigma = rng.uniform(0.1, 0.3, (N, n_gaussians)).astype(np.float32)'''

    # Input vector: [amplitudes | spreads]  →  [N, n_gaussians * 2]
    P = np.concatenate([A, sigma], axis=1)

    # --- Evaluate fields (vectorised) ---
    print("[3D] Computing ground-truth fields (vectorised)…")
    U = _compute_fields(A, sigma, coords, centers)       # [N, M]

    if plot:
        _plot_slices(coords, U, grid_res, rng, n_train)

    _save(cfg, coords, P, U, n_train, n_val, n_test)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _plot_slices(coords, U, grid_res, rng, n_train, n=5):
    """Show the z=0 slice of n random training samples."""
    # Find nodes near z=0
    z_vals  = np.unique(coords[:, 2])
    z_mid   = z_vals[len(z_vals) // 2]
    z_mask  = np.isclose(coords[:, 2], z_mid)
    x_slice = coords[z_mask, 0].reshape(grid_res, grid_res)
    y_slice = coords[z_mask, 1].reshape(grid_res, grid_res)

    indices = rng.integers(0, n_train, n)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, i in zip(axes, indices):
        field_slice = U[i][z_mask].reshape(grid_res, grid_res)
        im = ax.pcolormesh(x_slice, y_slice, field_slice, cmap="RdBu_r",
                           shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Sample {i}  (z≈{z_mid:.2f})")
        ax.set_aspect("equal")
    plt.suptitle("3-D benchmark — z=0 slice of random training samples")
    plt.tight_layout()
    plt.show()


def _save(cfg, coords, P, U, n_train, n_val, n_test):
    """Write everything to HDF5 in the canonical schema."""
    h5_path = cfg["data"]["h5_path"]
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    p_mean = P[:n_train].mean(0).astype(np.float32)
    p_std  = (P[:n_train].std(0) + 1e-9).astype(np.float32)
    u_mean = U[:n_train].mean(0).astype(np.float32)
    u_std  = (U[:n_train].std(0) + 1e-9).astype(np.float32)

    splits = {
        "train": (0,                  n_train),
        "val":   (n_train,            n_train + n_val),
        "test":  (n_train + n_val,    n_train + n_val + n_test),
    }

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("x_grid", data=coords)          # [M, 3]

        stats = f.create_group("stats")
        stats.create_dataset("p_mean", data=p_mean)
        stats.create_dataset("p_std",  data=p_std)
        stats.create_dataset("u_mean", data=u_mean)
        stats.create_dataset("u_std",  data=u_std)

        for split, (start, end) in splits.items():
            print(f"  [3D] Saving {split} ({end - start} samples)…")
            g = f.create_group(split)
            g.create_dataset("P", data=P[start:end], compression="gzip")
            g.create_dataset("U", data=U[start:end], compression="gzip")

    print(f"[3D] Dataset saved → {h5_path}")
