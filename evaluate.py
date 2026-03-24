"""
Evaluation script — works with any benchmark (1-D or 3-D).

Usage
-----
    python evaluate.py --config config_1d.yaml
    python evaluate.py --config config_3d.yaml
    python evaluate.py --config config_1d.yaml --model models/best_model.pth --n_plots 8
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import GaussianDataset
from utils import build_model, load_x_grid


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, x_grid, device):
    all_preds, all_targets = [], []
    for p, u in loader:
        p, u = p.to(device), u.to(device)
        all_preds.append(model(p, x_grid).cpu().numpy())
        all_targets.append(u.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    rmse   = np.sqrt(np.mean((preds - targets) ** 2, axis=1))
    rel_l2 = (
        np.linalg.norm(preds - targets, axis=1) /
        (np.linalg.norm(targets, axis=1) + 1e-12)
    )
    return {"rmse": rmse, "rel_l2": rel_l2,
            "rmse_mean":   rmse.mean(),   "rmse_std":   rmse.std(),
            "rel_l2_mean": rel_l2.mean(), "rel_l2_std": rel_l2.std()}


def print_metrics(metrics: dict, n_samples: int):
    print("=" * 50)
    print(f"Test results — {n_samples} samples")
    print(f"  Mean RMSE    : {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
    print(f"  Mean Rel-L2  : {metrics['rel_l2_mean']:.6f} ± {metrics['rel_l2_std']:.6f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(preds, targets, x_grid, metrics, cfg, n_plots=5, seed=0):
    benchmark = cfg.get("benchmark", "1d")
    rng       = np.random.default_rng(seed)
    indices   = rng.choice(len(preds), size=n_plots, replace=False)

    if benchmark == "1d":
        _plot_1d(preds, targets, x_grid, metrics, indices)
    else:
        _plot_3d(preds, targets, x_grid, metrics, indices, cfg)


def _plot_1d(preds, targets, x_grid, metrics, indices):
    x   = x_grid.cpu().numpy().flatten()
    fig, axes = plt.subplots(len(indices), 1,
                             figsize=(9, 3 * len(indices)), sharex=True)
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(x, targets[idx], label="Ground Truth", color="black", lw=2, alpha=0.7)
        ax.plot(x, preds[idx],   label="Prediction",   color="tomato",
                lw=2, linestyle="--")
        ax.set_title(
            f"Sample {idx}  |  "
            f"RMSE={metrics['rmse'][idx]:.4f}  "
            f"Rel-L2={metrics['rel_l2'][idx]:.4f}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=120)
    print("Plots saved → evaluation_plots.png")
    plt.show()


def _plot_3d(preds, targets, x_grid, metrics, indices, cfg):
    """Show the z≈0 slice for each selected sample (truth vs prediction)."""
    grid_res = cfg["data"]["grid_res"]
    coords   = x_grid.cpu().numpy()
    z_vals   = np.unique(coords[:, 2])
    z_mid    = z_vals[len(z_vals) // 2]
    mask     = np.isclose(coords[:, 2], z_mid)

    xs = coords[mask, 0].reshape(grid_res, grid_res)
    ys = coords[mask, 1].reshape(grid_res, grid_res)
    n  = len(indices)

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for row, idx in zip(axes, indices):
        u_slice    = targets[idx][mask].reshape(grid_res, grid_res)
        pred_slice = preds[idx][mask].reshape(grid_res, grid_res)
        vmin = min(u_slice.min(), pred_slice.min())
        vmax = max(u_slice.max(), pred_slice.max())

        for ax, data, title in zip(row,
                                    [u_slice, pred_slice],
                                    ["Ground Truth", "Prediction"]):
            im = ax.pcolormesh(xs, ys, data, cmap="RdBu_r", shading="auto",
                               vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            ax.set_title(
                f"Sample {idx} — {title}  |  "
                f"Rel-L2={metrics['rel_l2'][idx]:.4f}"
            )
            ax.set_aspect("equal")

    plt.suptitle(f"3-D benchmark evaluation  (z≈{z_mid:.2f} slice)")
    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=120)
    print("Plots saved → evaluation_plots.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(config_path: str, model_path: str | None = None, n_plots: int = 5):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Default model path uses the model type from config
    if model_path is None:
        model_tag  = cfg["model"]["type"]
        model_path = f"models/best_model_{model_tag}.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmark : {cfg.get('benchmark', '1d')}")
    print(f"Device    : {device}")

    # --- Data ---
    h5_path = cfg["data"]["h5_path"]
    test_ds = GaussianDataset(h5_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    print(f"Test set  : {len(test_ds)} samples")

    x_grid = load_x_grid(cfg, device)
    print(f"x_grid    : {tuple(x_grid.shape)}")

    # --- Model ---
    model = build_model(cfg).to(device)
    print(f"Loading weights from {model_path}…")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Inference ---
    preds, targets = run_inference(model, test_loader, x_grid, device)

    # --- Metrics ---
    metrics = compute_metrics(preds, targets)
    print_metrics(metrics, n_samples=len(preds))

    # --- Plots ---
    plot_results(preds, targets, x_grid, metrics, cfg, n_plots=n_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Gaussian model")
    parser.add_argument("--config",  required=True)
    parser.add_argument("--model",   default=None,
                        help="Path to .pth file (default: models/best_model_<type>.pth)")
    parser.add_argument("--n_plots", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.config, args.model, args.n_plots)
