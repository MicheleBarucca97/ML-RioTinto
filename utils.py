"""
Shared utilities: seeding, model factory.

The model factory (build_model) is the single source of truth for
instantiation, used by both train.py and evaluate.py.
"""

import random

import h5py
import numpy as np
import torch
import torch.nn as nn

from models import (
    FFNN, ResFFNN, DirectFieldNet, CNNDecoder,
    DeepONet, EfficientCoordinateNet, DeepSetsCoordinateNet,
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
# Model factory
# ---------------------------------------------------------------------------

# Models that do not generalise across benchmarks
_1D_ONLY = {"CNNDecoder", "DeepSetsCoordinateNet"}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model specified in *cfg["model"]*.

    Reads n_params, M, and spatial_dim from the data section of the config
    so there is never a mismatch between dataset dimensions and model dims.

    Raises:
        ValueError: for unknown model types or incompatible benchmark/model pairs.
    """
    model_cfg   = cfg["model"]
    model_type  = model_cfg["type"]
    n_params    = cfg["data"]["n_params"]
    M           = cfg["data"].get("M") or cfg["data"]["grid_res"] ** 3
    spatial_dim = cfg["data"]["spatial_dim"]
    benchmark   = cfg.get("benchmark", "1d")

    if model_type in _1D_ONLY and benchmark != "1d":
        raise ValueError(
            f"Model '{model_type}' is only compatible with the 1-D benchmark "
            f"(got benchmark='{benchmark}'). "
            f"See models.py docstring for details."
        )

    hidden_dim = model_cfg.get("hidden_dim", 256)
    latent_dim = model_cfg.get("latent_dim", 128)
    num_blocks = model_cfg.get("num_blocks", 4)

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

    raise ValueError(
        f"Unknown model type '{model_type}'. "
        f"Add it to utils.build_model() and models.py."
    )
