"""
Training script — works with any benchmark (1-D, 3-D, or Alucell).

Usage
-----
    python train.py --config config_1d.yaml
    python train.py --config config_3d.yaml
    python train.py --config config_alucell.yaml
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")   # headless-safe; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import GaussianDataset
from utils import (
    METRIC_LABELS, batch_val_metrics, build_model, fit_pod_basis_if_needed,
    is_better_metric, load_x_grid, plot_dir, run_tag,
    select_val_metric, set_seed,
    worst_metric_value,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sobolev_loss(preds: torch.Tensor, targets: torch.Tensor,
                 grad_weight: float = 0.1) -> torch.Tensor:
    """MSE on values + weighted MSE on finite-difference gradients.

    The gradient term penalises wrong slopes and helps with sharp features.
    Set grad_weight=0 to use plain MSE (recommended for 3-D / unstructured /
    POD coefficient targets).
    """
    loss = nn.functional.mse_loss(preds, targets)
    if grad_weight > 0:
        slope_pred = preds[:, 1:]   - preds[:, :-1]
        slope_true = targets[:, 1:] - targets[:, :-1]
        loss = loss + grad_weight * nn.functional.mse_loss(slope_pred, slope_true)
    return loss


# ---------------------------------------------------------------------------
# Per-epoch routines
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, x_grid, grad_weight, device):
    model.train()
    total_loss = 0.0
    for p, u in tqdm(loader, desc="  train", leave=False):
        p, u = p.to(device), u.to(device)
        optimizer.zero_grad()
        loss = sobolev_loss(model(p, x_grid), u, grad_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch_lbfgs(model, loader, optimizer, x_grid, grad_weight, device):
    """Specialized training loop for the L-BFGS optimizer."""
    model.train()
    total_loss = 0.0
    
    for p, u in tqdm(loader, desc="  train (L-BFGS)", leave=False):
        p, u = p.to(device), u.to(device)
        
        def closure():
            optimizer.zero_grad()
            output = model(p, x_grid)
            preds = output[0] if isinstance(output, tuple) else output
            loss = sobolev_loss(preds, u, grad_weight)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        with torch.no_grad():
            output = model(p, x_grid)
            preds = output[0] if isinstance(output, tuple) else output
            loss = sobolev_loss(preds, u, grad_weight)
            total_loss += loss.item()
            
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, x_grid, device, epoch, cfg, monitored,
              save_plot_every: int = 5):
    """Mean RMSE, rel-L2 and R^2 over the whole validation split.

    All three are always computed and logged; *monitored* only decides which
    one drives checkpointing / early stopping and labels the validation plot.
    """
    model.eval()
    benchmark = cfg.get("benchmark", "1d")
    totals, n_seen = None, 0.0

    for batch_idx, (p, u) in enumerate(loader):
        p, u = p.to(device), u.to(device)
        preds = model(p, x_grid)

        sums = batch_val_metrics(preds, u)
        n_seen += sums.pop("n")
        totals = sums if totals is None else {k: totals[k] + v
                                              for k, v in sums.items()}

        if batch_idx == 0 and epoch % save_plot_every == 0:
            first = batch_val_metrics(preds[:1], u[:1])
            _save_val_plot(x_grid, u[0], preds[0], first[monitored], monitored,
                           epoch, benchmark, cfg)

    return {k: v / n_seen for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Validation plotting
# ---------------------------------------------------------------------------

def _save_val_plot(x_grid, u, pred, score, monitored, epoch, benchmark, cfg):
    path = os.path.join(plot_dir(cfg), f"val_epoch_{epoch + 1:04d}.png")
    label = f"{METRIC_LABELS[monitored]} = {score:.4g}"

    if benchmark == "alucell":
        _plot_alucell_val(x_grid, u, pred, label, epoch, path)
    elif benchmark == "1d":
        _plot_1d_val(x_grid, u, pred, label, epoch, path)
    else:
        _plot_3d_val(x_grid, u, pred, label, epoch, cfg, path)


def _plot_alucell_val(x_grid, u, pred, score_label, epoch, path):
    """Validation plot for Alucell: contour on mid-ACD plane or interface."""
    coords = x_grid.cpu().numpy()
    M_grid = coords.shape[0]
    u_np   = u.cpu().numpy()
    pred_np = pred.cpu().numpy()

    # Vector field (M×3 → magnitude) vs scalar
    if u_np.shape[0] == M_grid * 3:
        field_true = np.linalg.norm(u_np.reshape(M_grid, 3), axis=1)
        field_pred = np.linalg.norm(pred_np.reshape(M_grid, 3), axis=1)
        label = "|u| (m/s)"
    elif u_np.shape[0] == M_grid:
        field_true = u_np
        field_pred = pred_np
        label = "field"
    else:
        # POD coefficients — bar chart instead of contour
        fig, ax = plt.subplots(figsize=(10, 4))
        k = len(u_np)
        x = np.arange(k)
        ax.bar(x - 0.15, u_np, width=0.3, label="Truth", color="steelblue")
        ax.bar(x + 0.15, pred_np, width=0.3, label="Pred",  color="tomato")
        ax.set_xlabel("POD mode")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"Epoch {epoch + 1}  |  {score_label}")
        ax.legend()
        fig.savefig(path, dpi=80)
        plt.close(fig)
        return

    vmin = min(field_true.min(), field_pred.min())
    vmax = max(field_true.max(), field_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(axes,
                                [field_true, field_pred],
                                ["Ground Truth", "Prediction"]):
        sc = ax.tricontourf(coords[:, 0], coords[:, 1], data,
                            levels=32, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.suptitle(f"Epoch {epoch + 1}  |  {score_label}")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _plot_1d_val(x_grid, u, pred, score_label, epoch, path):
    x = x_grid.cpu().numpy().flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, u.cpu().numpy(),    label="Ground Truth", color="steelblue", lw=1.5)
    ax.plot(x, pred.cpu().numpy(), label="Prediction",   color="tomato",
            linestyle="--", lw=1.5)
    ax.set_title(f"Epoch {epoch + 1}  |  {score_label}")
    ax.legend()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _plot_3d_val(x_grid, u, pred, score_label, epoch, cfg, path):
    """z≈mid slice colour map."""
    grid_res = cfg["data"]["grid_res"]
    coords   = x_grid.cpu().numpy()
    z_vals   = np.unique(coords[:, 2])
    z_mid    = z_vals[len(z_vals) // 2]
    mask     = np.isclose(coords[:, 2], z_mid)

    xs = coords[mask, 0].reshape(grid_res, grid_res)
    ys = coords[mask, 1].reshape(grid_res, grid_res)
    u_slice    = u.cpu().numpy()[mask].reshape(grid_res, grid_res)
    pred_slice = pred.cpu().numpy()[mask].reshape(grid_res, grid_res)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(axes,
                                [u_slice, pred_slice],
                                ["Ground Truth", "Prediction"]):
        im = ax.pcolormesh(xs, ys, data, cmap="RdBu_r", shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.suptitle(f"Epoch {epoch + 1}  |  z≈{z_mid:.2f}  |  {score_label}")
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------

#: Columns shared by train.py and train_dpp.py — all metrics are always
#: logged so runs from either script stay directly comparable.
STATS_COLUMNS = ["epoch", "train_loss", "val_rmse", "val_rel_l2", "val_r2",
                 "lr", "epoch_time_s", "cumulative_time_s"]


def _init_stats_csv(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(STATS_COLUMNS)


def _append_stats_csv(path: str, row: dict):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([row[c] for c in STATS_COLUMNS])



def _plot_training_curves(stats_path: str, out_path: str,
                          monitored: str = "rel_l2"):
    col = f"val_{monitored}"
    epochs, train_loss, val_metric, lr, epoch_time = [], [], [], [], []
    with open(stats_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_metric.append(float(row[col]))
            lr.append(float(row["lr"]))
            epoch_time.append(float(row["epoch_time_s"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_loss, color="steelblue", lw=1.5)
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Train Loss")

    label = METRIC_LABELS[monitored]
    axes[0, 1].plot(epochs, val_metric, color="tomato", lw=1.5)
    axes[0, 1].set_ylabel(f"Val {label}")
    axes[0, 1].set_xlabel("Epoch")
    # R^2 can be negative early on, so log scale is not always usable
    if all(v > 0 for v in val_metric):
        axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title(f"Validation {label}  (monitored)")

    axes[1, 0].plot(epochs, lr, color="seagreen", lw=1.5)
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Learning Rate Schedule")

    axes[1, 1].bar(epochs, epoch_time, color="slategray", alpha=0.7)
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].set_title("Wall-Clock Time per Epoch")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Training curves saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = cfg.get("benchmark", "1d")
    monitored = select_val_metric(cfg)
    print(f"Benchmark : {benchmark}")
    print(f"Device    : {device}")
    print(f"Monitoring: val {METRIC_LABELS[monitored]} "
          f"(checkpointing + early stopping)")
    if benchmark == "alucell" and monitored != "rel_l2":
        print("            rel-L2 is still logged, but it is not meaningful here: "
              "the target\n"
              "            is a velocity perturbation, so ||target|| → 0 near the "
              "reference run.")

    # --- Data ---
    h5_path    = cfg["data"]["h5_path"]
    train_cfg  = cfg["training"]
    normalize  = cfg["data"].get("normalize", False)

    train_ds = GaussianDataset(h5_path, split="train", normalize=normalize)
    val_ds   = GaussianDataset(h5_path, split="val",   normalize=normalize)
    if normalize:
        print("Targets   : standardised with the train-split u_mean / u_std "
              "(evaluate.py undoes this)")

    if cfg["data"].get("use_subset", False):
        n_sub    = cfg["data"].get("subset_size", cfg["data"]["samples"]["train"])
        rng      = np.random.default_rng(cfg["seed"])
        indices  = rng.permutation(len(train_ds))[:n_sub]
        train_ds = Subset(train_ds, indices)
        print(f"Subset    : {n_sub} training samples")
    else:
        print(f"Train set : {len(train_ds)} samples")

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=num_workers,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"],
                              shuffle=False, num_workers=num_workers)

    x_grid = load_x_grid(cfg, device)
    print(f"x_grid    : {tuple(x_grid.shape)}")

    # --- Model ---
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model     : {cfg['model']['type']}  ({n_params:,} parameters)")

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(train_cfg["lr"]),
                            weight_decay=float(train_cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
    )

    # --- POD basis fitting (no-op for all models except POD_MLP) ---
    fit_pod_basis_if_needed(model, train_ds, device)

    # --- Training loop ---
    grad_weight      = train_cfg.get("grad_loss_weight", 0.1)
    patience         = train_cfg.get("patience", 30)
    best_val         = worst_metric_value(monitored)
    patience_counter = 0

    os.makedirs("models", exist_ok=True)
    model_tag  = run_tag(cfg)
    save_path  = f"models/best_model_{model_tag}.pth"

    total_epochs = train_cfg["epochs"]
    lbfgs_epochs = 0
    adam_epochs  = total_epochs - lbfgs_epochs

    stats_csv_path = f"models/training_stats_{model_tag}.csv"
    _init_stats_csv(stats_csv_path)
    cumulative_time = 0.0

    print(f"\nTraining for up to {total_epochs} epochs…")
    print(f"  Phase 1: AdamW for {adam_epochs} epochs")
    if lbfgs_epochs:
        print(f"  Phase 2: L-BFGS for {lbfgs_epochs} epochs")
    print()

    for epoch in range(total_epochs):
        epoch_start = time.perf_counter()

        if epoch == adam_epochs and lbfgs_epochs > 0:
            print("\n>>> Switching to L-BFGS…\n")
            optimizer = optim.LBFGS(
                model.parameters(), lr=0.01, max_iter=20,
                history_size=50, line_search_fn="strong_wolfe")
            scheduler = None

        if epoch < adam_epochs:
            train_loss = train_epoch(model, train_loader, optimizer,
                                     x_grid, grad_weight, device)
            if scheduler:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
        else:
            train_loss = train_epoch_lbfgs(model, train_loader, optimizer,
                                           x_grid, grad_weight, device)
            lr = optimizer.param_groups[0]["lr"]

        val = val_epoch(model, val_loader, x_grid, device, epoch, cfg, monitored)

        epoch_time = time.perf_counter() - epoch_start
        cumulative_time += epoch_time

        _append_stats_csv(stats_csv_path, {
            "epoch":           epoch + 1,
            "train_loss":      f"{train_loss:.6f}",
            "val_rmse":        f"{val['rmse']:.6e}",
            "val_rel_l2":      f"{val['rel_l2']:.6f}",
            "val_r2":          f"{val['r2']:.6f}",
            "lr":              f"{lr:.2e}",
            "epoch_time_s":    f"{epoch_time:.2f}",
            "cumulative_time_s": f"{cumulative_time:.2f}",
        })

        print(f"Epoch {epoch + 1:4d}  "
              f"train={train_loss:.5f}  val_rmse={val['rmse']:.5e}  "
              f"val_rel_l2={val['rel_l2']:.5f}  val_R2={val['r2']:.4f}  "
              f"lr={lr:.2e}  time={epoch_time:.1f}s")

        if is_better_metric(monitored, val[monitored], best_val):
            best_val         = val[monitored]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → New best {METRIC_LABELS[monitored]} ({best_val:.5e})")
        else:
            patience_counter += 1
            if epoch < adam_epochs and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"\nDone. Best val {METRIC_LABELS[monitored]}: {best_val:.5e}")
    print(f"Total training time : {cumulative_time:.1f}s")
    print(f"Best model → {save_path}")

    _plot_training_curves(stats_csv_path,
                          f"models/training_curves_{model_tag}.png",
                          monitored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)