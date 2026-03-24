"""
Training script — works with any benchmark (1-D or 3-D).

Usage
-----
    python train.py --config config_1d.yaml
    python train.py --config config_3d.yaml
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
from utils import build_model, load_x_grid, set_seed, fit_pod_basis_if_needed


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

def train_epoch_lbfgs(model, loader, optimizer, x_grid, grad_weight, device):
    """
    Specialized training loop for the L-BFGS optimizer.
    Requires a closure function to re-evaluate the loss multiple times per step.
    """
    model.train()
    total_loss = 0.0
    
    for p, u in tqdm(loader, desc="  train (L-BFGS)", leave=False):
        p, u = p.to(device), u.to(device)
        
        # 1. Define the closure inside the batch loop
        def closure():
            optimizer.zero_grad()
            output = model(p, x_grid)
            
            # Handle tuple returns if you kept the duck-typing, otherwise just standard
            preds = output[0] if isinstance(output, tuple) else output
            
            loss = sobolev_loss(preds, u, grad_weight)
            loss.backward()
            return loss
            
        # 2. Step the optimizer using the closure
        optimizer.step(closure)
        
        # 3. Accumulate loss for logging (run a single forward pass without tracking gradients)
        with torch.no_grad():
            output = model(p, x_grid)
            preds = output[0] if isinstance(output, tuple) else output
            loss = sobolev_loss(preds, u, grad_weight)
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
# Training statistics
# ---------------------------------------------------------------------------

def _init_stats_csv(path: str):
    """Create the CSV file and write the header row."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_rel_l2", "lr",
                          "epoch_time_s", "cumulative_time_s"])


def _append_stats_csv(path: str, row: dict):
    """Append a single epoch row to the CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row["epoch"], row["train_loss"], row["val_rel_l2"],
                          row["lr"], row["epoch_time_s"],
                          row["cumulative_time_s"]])


def _plot_training_curves(stats_path: str, out_path: str):
    """Read the CSV log and produce a summary training curves figure."""
    epochs, train_loss, val_rel_l2, lr, epoch_time = [], [], [], [], []
    with open(stats_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_rel_l2.append(float(row["val_rel_l2"]))
            lr.append(float(row["lr"]))
            epoch_time.append(float(row["epoch_time_s"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train loss
    axes[0, 0].plot(epochs, train_loss, color="steelblue", lw=1.5)
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Train Loss")

    # Val rel-L2
    axes[0, 1].plot(epochs, val_rel_l2, color="tomato", lw=1.5)
    axes[0, 1].set_ylabel("Val Rel-L2")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Validation Rel-L2")

    # Learning rate
    axes[1, 0].plot(epochs, lr, color="seagreen", lw=1.5)
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Learning Rate Schedule")

    # Epoch wall-clock time
    axes[1, 1].bar(epochs, epoch_time, color="slategray", alpha=0.7)
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].set_title("Wall-Clock Time per Epoch")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Training curves saved -> {out_path}")


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

    # --- POD basis fitting (no-op for all models except POD_MLP) ---
    fit_pod_basis_if_needed(model, train_ds, device)

    # --- Training loop ---
    grad_weight      = train_cfg.get("grad_loss_weight", 0.1)
    patience         = train_cfg.get("patience", 30)
    best_val_loss    = float("inf")
    patience_counter = 0

    os.makedirs("models", exist_ok=True)
    model_tag  = cfg["model"]["type"]
    save_path  = f"models/best_model_{model_tag}.pth"

    # NEW: Define when to switch to L-BFGS
    total_epochs = train_cfg["epochs"]
    lbfgs_epochs = 0
    adam_epochs = total_epochs - lbfgs_epochs

    # --- Training statistics log ---
    stats_csv_path = f"models/training_stats_{model_tag}.csv"
    _init_stats_csv(stats_csv_path)
    cumulative_time = 0.0

    print(f"\nTraining for up to {total_epochs} epochs…")
    print(f"  Phase 1: AdamW for {adam_epochs} epochs")
    print(f"  Phase 2: L-BFGS for {lbfgs_epochs} epochs\n")

    for epoch in range(total_epochs):
        epoch_start = time.perf_counter()

        # --- Optimizer Switch Logic ---
        if epoch == adam_epochs:
            print("\n>>> Switching optimizer from AdamW to L-BFGS for fine-tuning...\n")
            optimizer = optim.LBFGS(
                model.parameters(),
                lr=0.01,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe"
            )
            scheduler = None
        # -----------------------------------

        # Route to the correct training function
        if epoch < adam_epochs:
            train_loss = train_epoch(model, train_loader, optimizer, x_grid, grad_weight, device)
            if scheduler:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
        else:
            train_loss = train_epoch_lbfgs(model, train_loader, optimizer, x_grid, grad_weight, device)
            lr = optimizer.param_groups[0]["lr"]

        val_loss = val_epoch(model, val_loader, x_grid, device, epoch, cfg)

        epoch_time = time.perf_counter() - epoch_start
        cumulative_time += epoch_time

        # Log to CSV
        _append_stats_csv(stats_csv_path, {
            "epoch":           epoch + 1,
            "train_loss":      f"{train_loss:.6f}",
            "val_rel_l2":      f"{val_loss:.6f}",
            "lr":              f"{lr:.2e}",
            "epoch_time_s":    f"{epoch_time:.2f}",
            "cumulative_time_s": f"{cumulative_time:.2f}",
        })

        print(f"Epoch {epoch + 1:4d}  "
              f"train={train_loss:.5f}  val_rel_l2={val_loss:.5f}  "
              f"lr={lr:.2e}  time={epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved ({best_val_loss:.5f})")
        else:
            patience_counter += 1
            if epoch < adam_epochs and patience_counter >= patience:
                print(f"Early stopping triggered during Adam phase at epoch {epoch + 1}.")
                epoch = adam_epochs - 1
                continue

    # --- Summary ---
    print(f"\nDone. Best val rel-L2: {best_val_loss:.5f}")
    print(f"Total training time : {cumulative_time:.1f}s")
    print(f"Best model -> {save_path}")
    print(f"Stats log  -> {stats_csv_path}")

    # Generate training curves plot
    _plot_training_curves(stats_csv_path, f"models/training_curves_{model_tag}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
