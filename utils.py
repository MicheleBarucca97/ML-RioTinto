"""
Shared utilities: seeding, validation metrics, model factory, POD basis fitting.

The model factory (build_model) is the single source of truth for
instantiation, used by both train.py and evaluate.py.
"""

import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import (
    FFNN, ResFFNN, DirectFieldNet, CNNDecoder, CNNDecoder3D,
    DeepONet, EfficientCoordinateNet, DeepSetsCoordinateNet,
    POD_MLP, MeshGraphNet, PointwiseFFNN,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Artefact naming
# ---------------------------------------------------------------------------

def run_tag(cfg: dict) -> str:
    """Identifier for a (dataset, model) pair, used to name every artefact.

    Naming artefacts by model type alone collides whenever two mappings share
    an architecture.  The Alucell mid-ACD and interface surrogates are both
    POD_MLP, so both wrote models/best_model_POD_MLP.pth and whichever was
    trained second silently destroyed the first — along with its training log
    and its validation plots.  Including the dataset stem separates them.
    """
    stem = Path(cfg["data"]["h5_path"]).stem
    return f"{cfg['model']['type']}__{stem}"


def checkpoint_path(cfg: dict) -> str:
    """Where training writes this run's weights."""
    return f"models/best_model_{run_tag(cfg)}.pth"


def resolve_checkpoint(cfg: dict) -> str:
    """Checkpoint to load, tolerating weights trained before the rename.

    Falls back to the old type-only name when only that exists, so existing
    checkpoints keep working; the note tells the user how to migrate.
    """
    preferred = checkpoint_path(cfg)
    if os.path.exists(preferred):
        return preferred
    legacy = f"models/best_model_{cfg['model']['type']}.pth"
    if os.path.exists(legacy):
        print(f"[note] loading legacy checkpoint {legacy}\n"
              f"       (retrain, or rename it to {preferred}, to keep this "
              f"dataset's weights separate)")
        return legacy
    return preferred


def plot_dir(cfg: dict) -> str:
    """Per-run directory for validation plots, created on demand."""
    d = os.path.join("plots", run_tag(cfg))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# x_grid loader — reads the saved spatial coordinates from HDF5
# ---------------------------------------------------------------------------

def load_x_grid(cfg: dict, device: str) -> torch.Tensor:
    """Load the spatial grid stored in the benchmark HDF5 file.

    Returns:
        Tensor of shape [M, spatial_dim] on *device*.
    """
    with h5py.File(cfg["data"]["h5_path"], "r") as f:
        x_grid = torch.from_numpy(f["x_grid"][:])
    return x_grid.to(device)


# ---------------------------------------------------------------------------
# Auto-detect output dimension M from HDF5
# ---------------------------------------------------------------------------

def _detect_M_from_h5(cfg: dict) -> int:
    """Read the output dimension from the HDF5 file.

    For structured-grid benchmarks, M = grid_res ** 3.
    For Alucell / unstructured data, reads the shape of the training target.
    """
    if "grid_res" in cfg["data"]:
        return cfg["data"]["grid_res"] ** 3

    h5_path = cfg["data"]["h5_path"]
    with h5py.File(h5_path, "r") as f:
        # Try meta first (written by prepare_alucell.py)
        if "meta" in f and "M_out" in f["meta"].attrs:
            return int(f["meta"].attrs["M_out"])
        # Fallback: read from training data shape
        return f["train"]["U"].shape[1]


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

#: Human-readable names, used in logs and plot titles.
METRIC_LABELS = {"rmse": "RMSE", "rel_l2": "rel-L2", "r2": "R2"}

#: Metric monitored for checkpointing / early stopping, per benchmark.
#:
#: Relative L2 is meaningful on the analytic benchmarks: the target IS the
#: field, so ||target|| is O(1).  It is NOT meaningful for Alucell, where the
#: target is a velocity perturbation (delta-learning against the uniform-current
#: reference run) or its POD coefficients.  There ||target|| -> 0 for samples
#: close to the reference, so the ratio blows up precisely where the model is
#: most accurate.  Alucell is monitored on RMSE instead, with R2 reporting how
#: much of the perturbation was captured.
_DEFAULT_VAL_METRIC = {"alucell": "rmse"}


def select_val_metric(cfg: dict) -> str:
    """Return the metric name used for early stopping and checkpointing.

    Override per run with training.val_metric in the config file.
    """
    default = _DEFAULT_VAL_METRIC.get(cfg.get("benchmark", "1d"), "rel_l2")
    metric  = cfg.get("training", {}).get("val_metric", default)
    if metric not in METRIC_LABELS:
        raise ValueError(
            f"Unknown training.val_metric '{metric}'. "
            f"Available: {list(METRIC_LABELS)}"
        )
    return metric


def worst_metric_value(metric: str) -> float:
    """Initial 'best so far' value for *metric* (R2 is maximised)."""
    return -float("inf") if metric == "r2" else float("inf")


def is_better_metric(metric: str, value: float, best: float) -> bool:
    """True if *value* improves on *best* (R2 is maximised, others minimised)."""
    return value > best if metric == "r2" else value < best


@torch.no_grad()
def batch_val_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Per-sample validation metrics, SUMMED over the batch.

    Sums (plus the sample count under key "n") rather than means, so callers can
    accumulate across batches — and, under DDP, across ranks — and divide once.
    Averaging per-batch means would weight a short final batch too heavily.

    Args:
        preds, targets: [B, M] model output and ground truth.

    Returns:
        dict with keys rmse, rel_l2, r2, n — all Python floats.
    """
    diff = preds - targets

    rmse   = torch.sqrt(diff.pow(2).mean(dim=1))
    rel_l2 = diff.norm(dim=1) / (targets.norm(dim=1) + 1e-12)

    ss_res = diff.pow(2).sum(dim=1)
    ss_tot = (targets - targets.mean(dim=1, keepdim=True)).pow(2).sum(dim=1)
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        "rmse":   float(rmse.sum()),
        "rel_l2": float(rel_l2.sum()),
        "r2":     float(r2.sum()),
        "n":      float(preds.shape[0]),
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_1D_ONLY = {"CNNDecoder", "DeepSetsCoordinateNet"}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model specified in *cfg["model"]*.

    Reads n_params, M, and spatial_dim from the data section of the config
    so there is never a mismatch between dataset dimensions and model dims.

    For Alucell datasets without a structured grid, M is auto-detected
    from the HDF5 file.

    Raises:
        ValueError: for unknown model types or incompatible benchmark/model pairs.
    """
    model_cfg   = cfg["model"]
    model_type  = model_cfg["type"]
    n_params    = cfg["data"]["n_params"]
    M           = cfg["data"].get("M") or _detect_M_from_h5(cfg)
    spatial_dim = cfg["data"]["spatial_dim"]
    benchmark   = cfg.get("benchmark", "1d")
    grid_res    = cfg["data"].get("grid_res", 15)

    if model_type in _1D_ONLY and benchmark != "1d":
        raise ValueError(
            f"Model '{model_type}' is only compatible with the 1-D benchmark "
            f"(got benchmark='{benchmark}'). "
            f"See models.py docstring for details."
        )

    hidden_dim = model_cfg.get("hidden_dim", 256)
    latent_dim = model_cfg.get("latent_dim", 128)
    num_blocks = model_cfg.get("num_blocks", 4)
    n_modes    = model_cfg.get("n_modes", 40)

    if model_type == "FFNN":
        return FFNN(in_dim=n_params, out_dim=M,
                    hidden_dims=[hidden_dim] * 3)

    if model_type == "ResFFNN":
        return ResFFNN(in_dim=n_params, out_dim=M,
                       hidden_dim=hidden_dim, num_blocks=num_blocks)

    if model_type == "DirectFieldNet":
        return DirectFieldNet(n_params=n_params, n_nodes=M,
                              hidden_dim=hidden_dim)

    if model_type == "CNNDecoder":
        return CNNDecoder(in_dim=n_params)

    if model_type == "CNNDecoder3D":
        return CNNDecoder3D(n_params=n_params, grid_res=grid_res,
                            base_ch=model_cfg.get("base_ch", 128))

    if model_type == "POD_MLP":
        return POD_MLP(n_params=n_params, n_nodes=M, n_modes=n_modes,
                       hidden_dim=hidden_dim, num_blocks=num_blocks)

    if model_type == "DeepONet":
        return DeepONet(n_params=n_params, spatial_dim=spatial_dim,
                        latent_dim=latent_dim)

    if model_type == "EfficientCoordinateNet":
        return EfficientCoordinateNet(n_params=n_params, spatial_dim=spatial_dim,
                                      hidden_dim=hidden_dim)

    if model_type == "DeepSetsCoordinateNet":
        return DeepSetsCoordinateNet(
            n_gaussians=cfg["data"]["n_gaussians"],
            hidden_dim=hidden_dim,
        )
    
    if model_type == "MeshGraphNet":
        return MeshGraphNet(n_params=n_params, grid_res=grid_res,
                            hidden_dim=hidden_dim, num_layers=num_blocks)

    if model_type == "PointwiseFFNN":
        return PointwiseFFNN(n_params=n_params, spatial_dim=spatial_dim,
                             hidden_dim=hidden_dim, num_blocks=num_blocks)

    raise ValueError(
        f"Unknown model type '{model_type}'. "
        f"Add it to utils.build_model() and models.py."
    )


# ---------------------------------------------------------------------------
# POD basis fitting helper
# ---------------------------------------------------------------------------

def fit_pod_basis_if_needed(model: nn.Module, dataset, device: str):
    """Fit the POD basis for POD_MLP models.

    This must be called once BEFORE the training loop starts.  It loads all
    training field snapshots U in one pass, moves them to CPU for SVD, and
    calls model.fit().  The fitted basis (V, u_mean) is stored as registered
    buffers so it will be saved and restored with the model's state_dict.

    For all other model types this is a no-op.

    Args:
        model:   The model returned by build_model().
        dataset: The training Dataset (or Subset thereof).
        device:  Target device string (used only for reporting).
    """
    if not isinstance(model, POD_MLP):
        return

    print("Fitting POD basis on all training snapshots…")
    # Load all U at once; SVD is done on CPU regardless of training device
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                        num_workers=0)
    _, U_all = next(iter(loader))        # [N_train, M]
    model.fit(U_all.cpu())
    # Move fitted buffers to the training device
    model.V      = model.V.to(device)
    model.u_mean = model.u_mean.to(device)