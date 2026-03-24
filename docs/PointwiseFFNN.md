# PointwiseFFNN — Architecture and Evaluation

## Overview

`PointwiseFFNN` is a coordinate-based neural network that learns the mapping

```
f(P, x) -> u(x)
```

where **P** is a parameter vector describing the physical system (e.g.
amplitudes and spreads of Gaussians) and **x** is a single spatial coordinate.
The output is the scalar field value at that coordinate.

To predict the full field on a mesh of M points, the model evaluates all M
coordinates in a single batched forward pass.

---

## Architecture

```
                     P [B, n_params]                x [M, spatial_dim]
                           |                               |
                   ┌───────┴───────┐               ┌───────┴────────┐
                   │ param_encoder │               │ Fourier encode │
                   │  Linear→GELU  │               │ + coord_proj   │
                   │  Linear       │               │   Linear       │
                   └───────┬───────┘               └───────┬────────┘
                           |                               |
                      p_emb [B, H]                   x_emb [M, H]
                           |                               |
                     unsqueeze(1)                    unsqueeze(0)
                     expand → [B, M, H]              expand → [B, M, H]
                           |                               |
                           └──────────┬────────────────────┘
                                      |
                              concat → [B, M, 2H]
                                      |
                              ┌───────┴───────┐
                              │  Linear(2H→H) │
                              │               │
                              │ ResidualBlock  │ × num_blocks
                              │  (Linear→SiLU │
                              │   →Linear+skip)│
                              └───────┬───────┘
                                      |
                                  [B, M, H]
                                      |
                              ┌───────┴───────┐
                              │  Linear(H→1)  │
                              └───────┬───────┘
                                      |
                                 squeeze → [B, M]
```

### Components

| Component | Description | Shape transformation |
|-----------|-------------|---------------------|
| **MultiScaleFourierEncoding** | Deterministic NeRF-style positional encoding with frequencies 2^0 ... 2^(L-1). Overcomes the spectral bias of plain MLPs toward low-frequency functions. | `[M, 3]` -> `[M, 3 + 3*2*num_freqs]` |
| **coord_proj** | Linear projection from Fourier feature space to hidden_dim. | `[M, fourier_dim]` -> `[M, H]` |
| **param_encoder** | Two-layer MLP (Linear->GELU->Linear) that encodes the parameter vector. | `[B, n_params]` -> `[B, H]` |
| **Broadcasting** | `expand` replicates p_emb across M points and x_emb across B samples. No memory copy — just a view. | `[B, H]` + `[M, H]` -> `[B, M, 2H]` |
| **trunk** | Linear(2H->H) followed by `num_blocks` ResidualBlocks. Each block: Linear->SiLU->Linear + skip connection. | `[B, M, 2H]` -> `[B, M, H]` |
| **head** | Single linear layer mapping to a scalar. | `[B, M, H]` -> `[B, M, 1]` -> squeeze -> `[B, M]` |

### Key design choices

1. **Fourier coordinate encoding** — Without it, MLPs cannot learn
   high-frequency spatial variation (known as spectral bias). The encoding
   maps each coordinate dimension through L frequency bands:
   `[x, sin(2*pi*2^0*x), cos(2*pi*2^0*x), ..., sin(2*pi*2^(L-1)*x), cos(2*pi*2^(L-1)*x)]`.

2. **Separate encoders, shared trunk** — The parameter encoder and coordinate
   encoder operate independently before merging. This lets the network build
   useful intermediate representations for each input modality before they
   interact.

3. **Residual blocks** — Skip connections stabilise training for deeper
   networks and help gradients flow. Each block computes `x + MLP(x)`.

4. **Batched evaluation** — Although the model conceptually maps a single
   `(P, x)` pair to a scalar, the forward pass evaluates all M grid points
   simultaneously via `expand` (a zero-copy broadcast). There is no Python
   loop over coordinates.

### Default hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_dim` | 256 | Width of all hidden layers |
| `num_blocks` | 4 | Depth of the residual trunk |
| `num_freqs` | 10 | Fourier frequency bands (output dim = `spatial_dim + spatial_dim * 2 * num_freqs`) |

With `n_params=48, spatial_dim=3`, this gives ~752k trainable parameters.

---

## Training

Training uses the standard pipeline in `train.py`. No special handling is
needed — the model conforms to the uniform `forward(p, x_grid) -> [B, M]`
signature used by all models.

```bash
# Set model.type in config_3d.yaml:
#   model:
#     type: "PointwiseFFNN"
#     hidden_dim: 256
#     num_blocks: 4

python train.py --config config_3d.yaml
```

### What happens during training

1. Each batch loads `(P, U)` pairs where `P: [B, n_params]` and `U: [B, M]`.
2. The model predicts `U_pred = model(P, x_grid)` with shape `[B, M]`.
3. Loss is computed as MSE between `U_pred` and `U` (Sobolev gradient term is
   disabled for 3-D via `grad_loss_weight: 0.0`).
4. The loss backpropagates through all M point evaluations simultaneously.

### Memory footprint

The peak intermediate tensor is the concatenated embedding `[B, M, 2H]`.
For `B=32, M=3375, H=256`: 32 * 3375 * 512 * 4 bytes = **~221 MB**.
This fits comfortably on most GPUs.

### Outputs

| File | Path |
|------|------|
| Best weights | `models/best_model_PointwiseFFNN.pth` |
| Training stats | `models/training_stats_PointwiseFFNN.csv` |
| Training curves | `models/training_curves_PointwiseFFNN.png` |

---

## Evaluation

```bash
# Evaluates using the model type from config to find the weights automatically
python evaluate.py --config config_3d.yaml

# Or specify weights explicitly
python evaluate.py --config config_3d.yaml --model models/best_model_PointwiseFFNN.pth
```

### Evaluation pipeline

```
 Load test set          Load trained model        Load x_grid
 (P, U) pairs           from .pth file            from HDF5
       |                       |                       |
       └───────────────────────┼───────────────────────┘
                               |
                     ┌─────────┴──────────┐
                     │   run_inference()   │
                     │                     │
                     │  for each batch:    │
                     │    pred = model(P,  │
                     │           x_grid)   │
                     │    collect preds    │
                     │    and targets      │
                     └─────────┬──────────┘
                               |
                    preds [N_test, M],  targets [N_test, M]
                               |
                     ┌─────────┴──────────┐
                     │  compute_metrics() │
                     │                    │
                     │  Per-sample:       │
                     │   RMSE = sqrt(     │
                     │     mean((pred -   │
                     │     target)^2))    │
                     │                    │
                     │   Rel-L2 =         │
                     │     ||pred-target||│
                     │     / ||target||   │
                     └─────────┬──────────┘
                               |
                     ┌─────────┴──────────┐
                     │   print_metrics()  │
                     │   plot_results()   │
                     └────────────────────┘
```

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | `sqrt(mean((pred_i - target_i)^2))` per sample, then averaged | Absolute error scale. Depends on field magnitude. |
| **Rel-L2** | `norm(pred - target) / norm(target)` per sample, then averaged | Scale-invariant. 0.01 = 1% error. Primary comparison metric. |

### Plots (3-D benchmark)

For each selected test sample, two panels are shown side by side:
- **Ground truth** field at the z=0 mid-slice
- **Prediction** field at the same slice

Both use the same colour scale for direct visual comparison, with the
per-sample Rel-L2 printed in the title.

---

## Comparison with other models

Run all three models on the same data, then compare the CSV logs:

```bash
# Train all three (change model.type in config_3d.yaml between runs)
python train.py --config config_3d.yaml   # type: POD_MLP
python train.py --config config_3d.yaml   # type: FFNN
python train.py --config config_3d.yaml   # type: PointwiseFFNN
```

Results are saved to separate files — no overwriting:

| Model | Weights | Stats CSV | Curves PNG |
|-------|---------|-----------|------------|
| POD_MLP | `best_model_POD_MLP.pth` | `training_stats_POD_MLP.csv` | `training_curves_POD_MLP.png` |
| FFNN | `best_model_FFNN.pth` | `training_stats_FFNN.csv` | `training_curves_FFNN.png` |
| PointwiseFFNN | `best_model_PointwiseFFNN.pth` | `training_stats_PointwiseFFNN.csv` | `training_curves_PointwiseFFNN.png` |

### Expected trade-offs

| | POD_MLP | FFNN | PointwiseFFNN |
|--|---------|------|---------------|
| **Sample efficiency** | Best (linear basis is optimal) | Good | Moderate |
| **Training speed** | Fast (regresses ~40 coefficients) | Fast (single forward pass) | Slower (B*M pointwise evals) |
| **Resolution flexibility** | Fixed mesh only | Fixed mesh only | Any coordinates |
| **Memory at inference** | Low (small MLP + matmul) | Low (single forward) | Higher ([B, M, 2H] tensor) |
| **Inductive bias** | Linear subspace (PCA) | None | Spatial smoothness via Fourier |
