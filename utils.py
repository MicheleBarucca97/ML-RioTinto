"""
Shared utilities: seeding, model factory, POD basis fitting.

The model factory (build_model) is the single source of truth for
instantiation, used by both train.py and evaluate.py.
"""

import random

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