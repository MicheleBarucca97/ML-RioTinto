"""
Training script — works with any benchmark (1-D or 3-D).

Usage
-----
    python train.py --config config_1d.yaml
    python train.py --config config_3d.yaml
"""

import argparse
import os

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
from utils import build_model, load_x_grid, set_seed


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sobolev_loss(preds: torch.Tensor, targets: torch.Tensor,
                 grad_weight: float = 0.1) -> torch.Tensor:
    """MSE on values + weighted MSE on finite-difference gradients.

    The gradient term penalises wrong slopes and helps with sharp features.
    Set grad_weight=0 to use plain MSE (recommended for 3-D / unstructured meshes).
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


@torch.no_grad()
def val_epoch(model, loader, x_grid, device, epoch, cfg,
              save_plot_every: int = 5):
    model.eval()
    total_rel_l2 = 0.0
    benchmark = cfg.get("benchmark", "1d")

    for batch_idx, (p, u) in enumerate(loader):
        p, u = p.to(device), u.to(device)
        preds = model(p, x_grid)
        rel_l2 = (
            torch.norm(preds - u, dim=1) /
            (torch.norm(u, dim=1) + 1e-6)
        ).mean()
        total_rel_l2 += rel_l2.item()

        if batch_idx == 0 and epoch % save_plot_every == 0:
            _save_val_plot(x_grid, u[0], preds[0], rel_l2.item(),
                           epoch, benchmark, cfg)

    return total_rel_l2 / len(loader)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_val_plot(x_grid, u, pred, rel_l2, epoch, benchmark, cfg):
    os.makedirs("plots", exist_ok=True)
    path = f"plots/val_epoch_{epoch + 1:04d}.png"

    if benchmark == "1d":
        _plot_1d(x_grid, u, pred, rel_l2, epoch, path)
    else:
        _plot_3d_slice(x_grid, u, pred, rel_l2, epoch, cfg, path)


def _plot_1d(x_grid, u, pred, rel_l2, epoch, path):
    x = x_grid.cpu().numpy().flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, u.cpu().numpy(),    label="Ground Truth", color="steelblue", lw=1.5)
    ax.plot(x, pred.cpu().numpy(), label="Prediction",   color="tomato",
            linestyle="--", lw=1.5)
    ax.set_title(f"Epoch {epoch + 1}  |  Rel-L2 = {rel_l2:.4f}")
    ax.legend()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _plot_3d_slice(x_grid, u, pred, rel_l2, epoch, cfg, path):
    """Show the z≈0 slice as a colour map (ground truth vs prediction)."""
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
    fig.suptitle(f"Epoch {epoch + 1}  |  z≈{z_mid:.2f}  |  Rel-L2 = {rel_l2:.4f}")
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmark : {cfg.get('benchmark', '1d')}")
    print(f"Device    : {device}")

    # --- Data ---
    h5_path    = cfg["data"]["h5_path"]
    train_cfg  = cfg["training"]

    train_ds = GaussianDataset(h5_path, split="train")
    val_ds   = GaussianDataset(h5_path, split="val")

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

    # Spatial grid — loaded from the HDF5 file (correct shape for any benchmark)
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

    # --- Training loop ---
    grad_weight      = train_cfg.get("grad_loss_weight", 0.1)
    patience         = train_cfg.get("patience", 30)
    best_val_loss    = float("inf")
    patience_counter = 0

    os.makedirs("models", exist_ok=True)
    save_path = "models/best_model.pth"

    print(f"\nTraining for up to {train_cfg['epochs']} epochs…\n")
    for epoch in range(train_cfg["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 x_grid, grad_weight, device)
        val_loss   = val_epoch(model, val_loader, x_grid, device, epoch, cfg)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:4d}  "
              f"train={train_loss:.5f}  val_rel_l2={val_loss:.5f}  lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best saved ({best_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"\nDone. Best val rel-L2: {best_val_loss:.5f}")
    print(f"Best model → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
