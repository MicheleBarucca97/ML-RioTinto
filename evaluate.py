"""
Evaluation script — works with any benchmark (1-D, 3-D, or Alucell).

For Alucell with delta-learning or POD coefficients, this script:
  1. Runs inference to get raw model output (Δu or POD coefficients).
  2. Reconstructs the physical field (adds u_ref, decodes POD basis).
  3. Computes metrics in BOTH model-output space and physical space.
  4. Breaks down errors by campaign mode (gaussian, gradient, cluster, …).

Usage
-----
    python evaluate.py --config config_alucell.yaml
    python evaluate.py --config config_alucell.yaml --model models/best_model_POD_MLP.pth
"""

import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import (
    GaussianDataset,
    load_reconstruction_context,
    reconstruct_field,
    load_test_modes,
)
from utils import build_model, load_x_grid, resolve_checkpoint, run_tag

# plot_alucell_3d is imported lazily inside plot_results: it pulls in PyVista,
# which is only needed for the 3-D Alucell renders and is not installed in
# every environment.  A module-level import would break the 1-D/3-D benchmarks.


def _plot_path(cfg: dict) -> str:
    """Output figure, named per (dataset, model) so runs do not overwrite."""
    return f"evaluation_plots_{run_tag(cfg)}.png"


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
    # 1. Root Mean Square Error (Absolute physical magnitude)
    rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=1))
    
    # 2. Relative L2 Norm
    rel_l2 = (
        np.linalg.norm(preds - targets, axis=1) /
        (np.linalg.norm(targets, axis=1) + 1e-12)
    )
    
    # 3. Explained Variance (R^2 Score)
    ss_res = np.sum((targets - preds) ** 2, axis=1)
    ss_tot = np.sum((targets - np.mean(targets, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    return {
        "rmse": rmse, "rel_l2": rel_l2, "r2": r2,
        "rmse_mean":   rmse.mean(),   "rmse_std":   rmse.std(),
        "rel_l2_mean": rel_l2.mean(), "rel_l2_std": rel_l2.std(),
        "r2_mean":     r2.mean(),     "r2_std":     r2.std()
    }

def print_metrics(metrics: dict, n_samples: int, label: str = ""):
    tag = f" ({label})" if label else ""
    print("=" * 65)
    print(f"Test results{tag} — {n_samples} samples")
    print(f"  Mean R^2 Score: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
    print(f"  Mean RMSE     : {metrics['rmse_mean']:.6e} ± {metrics['rmse_std']:.6e}")
    print(f"  Mean Rel-L2   : {metrics['rel_l2_mean']:.6f} ± {metrics['rel_l2_std']:.6f}")
    print(f"  Max Rel-L2    : {metrics['rel_l2'].max():.6f}")
    print("=" * 65)


def print_per_mode_metrics(metrics: dict, modes: list[str]):
    """Print per-campaign-type error breakdown."""
    if not modes:
        return

    groups = defaultdict(list)
    for i, m in enumerate(modes):
        groups[m].append(i)

    print("\n  Per-campaign-type breakdown:")
    print(f"  {'mode':12s}   {'n':>5s}   {'rel-L2 mean':>12s}   {'rel-L2 max':>11s}")
    print("  " + "-" * 50)

    for mode in sorted(groups):
        idx = groups[mode]
        rl2 = metrics["rel_l2"][idx]
        print(f"  {mode:12s}   {len(idx):5d}   "
              f"{rl2.mean():12.6f}   {rl2.max():11.6f}")


# ---------------------------------------------------------------------------
# Plots — Alucell
# ---------------------------------------------------------------------------

def _plot_alucell(preds, targets, x_grid, metrics, indices, cfg, recon_preds=None, recon_targets=None):
    """Plot results on mid-ACD plane or interface.

    When recon_preds is not None (delta/POD case), plots the reconstructed
    physical field.  Otherwise plots the raw model output.
    """
    coords = x_grid.cpu().numpy()
    M_grid = coords.shape[0]

    # Decide what to plot: reconstructed field if available, else raw
    plot_preds   = recon_preds if recon_preds is not None else preds
    plot_targets = recon_targets if recon_targets is not None else targets

    # Detect field type
    M_out = plot_preds.shape[1]
    is_vector  = (M_out == M_grid * 3)
    is_spatial = (M_out == M_grid) or is_vector
    is_coeffs  = not is_spatial

    if is_coeffs:
        # POD coefficients — bar chart
        n = len(indices)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
        if n == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices):
            k = plot_targets.shape[1]
            x = np.arange(k)
            ax.bar(x - 0.15, plot_targets[idx], width=0.3,
                   label="Truth", color="steelblue")
            ax.bar(x + 0.15, plot_preds[idx], width=0.3,
                   label="Pred",  color="tomato")
            ax.set_xlabel("POD mode")
            ax.set_ylabel("Coefficient")
            ax.set_title(f"Sample {idx}  |  Rel-L2={metrics['rel_l2'][idx]:.4f}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(_plot_path(cfg), dpi=120)
        print(f"Plots saved → {_plot_path(cfg)}")
        plt.close(fig)
        return

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        if is_vector:
            ft = np.linalg.norm(plot_targets[idx].reshape(M_grid, 3), axis=1)
            fp = np.linalg.norm(plot_preds[idx].reshape(M_grid, 3), axis=1)
            err = np.linalg.norm(
                (plot_targets[idx] - plot_preds[idx]).reshape(M_grid, 3), axis=1)
            label = "|u| (m/s)"
        else:
            ft = plot_targets[idx]
            fp = plot_preds[idx]
            err = np.abs(ft - fp)
            label = "h (m)"

        vmin = min(ft.min(), fp.min())
        vmax = max(ft.max(), fp.max())

        ax0 = axes[row, 0]
        sc = ax0.tricontourf(coords[:, 0], coords[:, 1], ft,
                             levels=32, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax0, label=label)
        ax0.set_title(f"Sample {idx} — Truth")
        ax0.set_aspect("equal")

        ax1 = axes[row, 1]
        sc = ax1.tricontourf(coords[:, 0], coords[:, 1], fp,
                             levels=32, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax1, label=label)
        ax1.set_title(f"Prediction  |  Rel-L2={metrics['rel_l2'][idx]:.4f}")
        ax1.set_aspect("equal")

        ax2 = axes[row, 2]
        sc = ax2.tricontourf(coords[:, 0], coords[:, 1], err,
                             levels=32, cmap="hot_r")
        plt.colorbar(sc, ax=ax2, label="error")
        ax2.set_title("Pointwise error")
        ax2.set_aspect("equal")

    plt.suptitle(f"Alucell — {cfg['model']['type']}", fontsize=14)
    plt.tight_layout()
    plt.savefig(_plot_path(cfg), dpi=120)
    print(f"Plots saved → {_plot_path(cfg)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — original benchmarks
# ---------------------------------------------------------------------------

def _plot_1d(preds, targets, x_grid, metrics, indices, cfg):
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
    plt.savefig(_plot_path(cfg), dpi=120)
    print(f"Plots saved → {_plot_path(cfg)}")
    plt.close(fig)


def _plot_3d(preds, targets, x_grid, metrics, indices, cfg):
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
    plt.savefig(_plot_path(cfg), dpi=120)
    print(f"Plots saved → {_plot_path(cfg)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot router
# ---------------------------------------------------------------------------

def plot_results(preds, targets, x_grid, metrics, cfg,
                 n_plots=5, seed=0, recon_preds=None, recon_targets=None):
    benchmark = cfg.get("benchmark", "1d")
    rng       = np.random.default_rng(seed)
    indices   = rng.choice(len(preds), size=min(n_plots, len(preds)),
                           replace=False)

    if benchmark == "alucell":
        spatial_dim = cfg["data"].get("spatial_dim", 2)
        if spatial_dim == 3 and recon_preds is not None:
            try:
                from plot_alucell_3d import plot_alucell_3d
            except ImportError as e:
                print(f"3-D Alucell plots need PyVista ({e}); "
                      f"install with: pip install 'pyvista[all]'")
                return
            plot_alucell_3d(preds, targets, x_grid, metrics, indices, cfg,
                            recon_preds=recon_preds, recon_targets=recon_targets)
        else:
            _plot_alucell(preds, targets, x_grid, metrics, indices, cfg,
                          recon_preds=recon_preds, recon_targets=recon_targets)
    elif benchmark == "1d":
        _plot_1d(preds, targets, x_grid, metrics, indices, cfg)
    else:
        _plot_3d(preds, targets, x_grid, metrics, indices, cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(config_path: str, model_path: str | None = None, n_plots: int = 5):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if model_path is None:
        model_path = resolve_checkpoint(cfg)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = cfg.get("benchmark", "1d")
    h5_path   = cfg["data"]["h5_path"]

    print(f"Benchmark : {benchmark}")
    print(f"Device    : {device}")

    # --- Data ---
    # Must match training: if the targets were standardised there, the model
    # predicts in standardised space and we undo it below so that every metric
    # and every reconstruction stays in physical units.
    normalize   = cfg["data"].get("normalize", False)
    test_ds     = GaussianDataset(h5_path, split="test", normalize=normalize)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    print(f"Test set  : {len(test_ds)} samples")
    if normalize:
        print("Targets   : standardised — de-normalising model output for metrics")

    x_grid = load_x_grid(cfg, device)
    print(f"x_grid    : {tuple(x_grid.shape)}")

    # --- Model ---
    model = build_model(cfg).to(device)
    print(f"Loading weights from {model_path}…")
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    # --- Inference (model-output space) ---
    preds, targets = run_inference(model, test_loader, x_grid, device)

    if normalize:
        u_mean = test_ds.u_mean.numpy()
        u_std  = test_ds.u_std.numpy()
        preds   = preds   * u_std + u_mean
        targets = targets * u_std + u_mean

    # --- Metrics in model-output space ---
    metrics_raw = compute_metrics(preds, targets)
    print_metrics(metrics_raw, len(preds), label="model-output space")

    # --- Reconstruction & physical-space metrics (Alucell only) ---
    recon_preds   = None
    recon_targets = None
    needs_recon   = False
    if benchmark == "alucell":
        ctx = load_reconstruction_context(h5_path)
        needs_recon = ctx["delta_learning"] or ctx["V"] is not None

        if ctx["delta_learning"]:
            print("  note: rel-L2 above is NOT meaningful — the target is a velocity"
                  " perturbation\n"
                  "        Δu, so ||Δu_true|| → 0 for runs close to the reference."
                  "  Use RMSE / R^2\n"
                  "        here, and the reconstructed-field metrics below for"
                  " engineering accuracy.")

        if needs_recon:
            recon_preds   = reconstruct_field(preds, ctx)
            recon_targets = reconstruct_field(targets, ctx)
            metrics_phys  = compute_metrics(recon_preds, recon_targets)
            print_metrics(metrics_phys, len(recon_preds),
                          label="reconstructed physical field")

            # The physical-space metrics are the ones that matter
            report_metrics = metrics_phys
        else:
            report_metrics = metrics_raw

        # Per-campaign-type breakdown
        modes = load_test_modes(h5_path)
        print_per_mode_metrics(report_metrics, modes)
    else:
        report_metrics = metrics_raw

    # --- Plots ---
    if needs_recon:
        plot_results(preds, targets, x_grid, report_metrics, cfg,
                     n_plots=n_plots, recon_preds=recon_preds, recon_targets=recon_targets)
    else:
        plot_results(preds, targets, x_grid, report_metrics, cfg,
                     n_plots=n_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config",  required=True)
    parser.add_argument("--model",   default=None)
    parser.add_argument("--n_plots", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.config, args.model, args.n_plots)