"""
Model zoo for the Gaussian benchmark.

Uniform forward signature
-------------------------
    forward(p, x_grid) -> Tensor[B, M]

    p      : [B, n_params]      parameter vector
    x_grid : [M, spatial_dim]   spatial coordinates (1-D or 3-D)

Grid-free models (FFNN, ResFFNN, DirectFieldNet, CNNDecoder*) ignore x_grid
but accept it so callers never need to branch on model type.

Benchmark compatibility
-----------------------
    1-D and 3-D : FFNN, ResFFNN, DirectFieldNet,
                  DeepONet, EfficientCoordinateNet, POD_MLP, CNNDecoder3D  
    1-D only    : CNNDecoder          (fixed 1-D convolutional topology)
                  DeepSetsCoordinateNet  (needs (A, center, sigma) triplets)

Architecture guide for the 3-D fixed-mesh case
-----------------------------------------------
Why not plain DeepONet?
  DeepONet's trunk re-evaluates ALL M coordinates at every forward pass.
  On a fixed mesh this wastes compute — the trunk output is always the same
  matrix.  You can cache it, but that just recovers POD_MLP (see below).

POD_MLP  (recommended starting point)
  Offline:  compute the top-k POD/PCA modes V from training snapshots.
  Online:   MLP predicts the k scalar coefficients; decode = c @ V.T + mean.
  Why it works:  sum-of-Gaussians fields live on a very low-dimensional
  manifold (≈10-30 modes capture >99 % variance).  The regression problem
  shrinks from [B→3375] to [B→k], which is far easier with 1 000 samples.
  Cost:    one SVD offline; then a small MLP at inference.  Very fast.

CNNDecoder3D
  Learns a spatial inductive bias (nearby voxels are correlated) via 3-D
  transposed convolutions.  Better than a flat MLP when fields are smooth.
  Works on the structured 15³ grid; AdaptiveAvgPool3d makes it resolution-
  agnostic at the end.

Nonlinear autoencoder + MLP  (future work — not implemented here)
  Replace the linear POD basis with a convolutional autoencoder trained on
  U.  Superior when fields have sharp gradients that POD cannot compress well.
  Needs more data (≥5 000 samples) to train the encoder stably.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------

class FourierEncoding(nn.Module):
    """Random Fourier Features for any input dimensionality."""

    def __init__(self, in_dim: int = 1, num_feats: int = 128,
                 sigma: float = 3.0):
        super().__init__()
        B = torch.randn(in_dim, num_feats) * sigma
        self.register_buffer("B", B)
        self.out_dim = num_feats * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MultiScaleFourierEncoding(nn.Module):
    """Deterministic NeRF-style positional encoding: frequencies 2^0 … 2^(L-1).

    Works for any spatial_dim (1-D or 3-D).
    Output dimension = in_dim + in_dim * 2 * num_frequencies.
    """

    def __init__(self, in_dim: int = 1, num_frequencies: int = 10):
        super().__init__()
        self.in_dim = in_dim
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim + in_dim * 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        scaled = x.unsqueeze(-1) * self.freqs       # [..., in_dim, L]
        scaled = scaled.view(*x.shape[:-1], -1)     # [..., in_dim * L]
        proj   = 2 * torch.pi * scaled
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Plain skip-connection MLP block."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Pre-norm residual block — more stable for deep networks."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# Grid-free models  (1-D and 3-D compatible)
# ---------------------------------------------------------------------------

class FFNN(nn.Module):
    """Vanilla MLP: parameters → full output grid (grid-free)."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 256]
        layers, curr = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(curr, h), nn.SiLU()]
            curr = h
        layers.append(nn.Linear(curr, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, p, x_grid=None):
        return self.net(p)


class ResFFNN(nn.Module):
    """MLP with residual blocks: parameters → full output grid (grid-free)."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: int = 256, num_blocks: int = 3):
        super().__init__()
        self.input_proj  = nn.Linear(in_dim, hidden_dim)
        self.blocks      = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, p, x_grid=None):
        h = self.input_proj(p)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class DirectFieldNet(nn.Module):
    """Compact MLP with dropout: parameters → full output grid.
    Good regularisation baseline.
    """

    def __init__(self, n_params: int, n_nodes: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_nodes),
        )

    def forward(self, p, x_grid=None):
        return self.net(p)


# ---------------------------------------------------------------------------
# 1-D-only grid-free model
# ---------------------------------------------------------------------------

class CNNDecoder(nn.Module):
    """1-D only. Project parameters to a low-res latent and upsample via
    transposed convolutions. Encodes the inductive bias that nearby output
    points are correlated.

    Fixed topology: 40 → 160 → 640 → 2560 (requires M = 2560).
    """

    def __init__(self, in_dim: int = 30):
        super().__init__()
        self.fc = nn.Linear(in_dim, 64 * 40)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4), nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4), nn.GELU(),
            nn.ConvTranspose1d(16,  1, kernel_size=4, stride=4),
        )

    def forward(self, p, x_grid=None):
        h = self.fc(p).view(-1, 64, 40)
        return self.decoder(h).squeeze(1)                    # [B, 2560]


# ---------------------------------------------------------------------------
# 3-D convolutional decoder  (3-D structured grids)
# ---------------------------------------------------------------------------

class CNNDecoder3D(nn.Module):
    """3-D extension of CNNDecoder using ConvTranspose3d.

    Architecture
    ------------
    p [B, n_params]
        → Linear → [B, C * seed³]
        → reshape → [B, C, seed, seed, seed]           (seed = 2)
        → ConvTranspose3d ×3  (each doubles spatial dims)
        → [B, 32, 16, 16, 16]
        → AdaptiveAvgPool3d(grid_res)                  (handles any target size)
        → Conv3d(32→1, k=1)
        → flatten → [B, grid_res³]

    Memory note: for grid_res=15, the peak activation is [B, 32, 16, 16, 16]
    ≈ 2 MB per sample at float32.  Scale back the channel widths if you use
    a much larger grid.

    Args:
        n_params:  Number of input parameters.
        grid_res:  Spatial resolution of the target cube (output is grid_res³).
        base_ch:   Number of channels in the widest convolutional layer.
    """

    _SEED = 2          # initial spatial size before upsampling

    def __init__(self, n_params: int = 48, grid_res: int = 15,
                 base_ch: int = 128):
        super().__init__()
        self.grid_res = grid_res
        seed = self._SEED

        # Project param vector to a tiny 3-D feature map
        self.fc = nn.Linear(n_params, base_ch * (seed ** 3))

        # Three doubling stages: seed → 2s → 4s → 8s
        # ConvTranspose3d(in, out, k=4, s=2, p=1) doubles spatial dims exactly
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_ch,      base_ch // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch // 2),
            nn.GELU(),

            nn.ConvTranspose3d(base_ch // 2, base_ch // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch // 4),
            nn.GELU(),

            nn.ConvTranspose3d(base_ch // 4, 32,           kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        # After decoder: [B, 32, 8*seed, 8*seed, 8*seed]

        # Resolution-agnostic pooling to exactly grid_res³
        self.pool = nn.AdaptiveAvgPool3d(grid_res)

        # 1×1×1 conv: 32 channels → 1 (the scalar field value)
        self.head = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, p, x_grid=None):
        B = p.shape[0]
        seed = self._SEED
        base_ch = self.fc.out_features // (seed ** 3)

        h = self.fc(p).view(B, base_ch, seed, seed, seed)   # [B, C, 2, 2, 2]
        h = self.decoder(h)                                  # [B, 32, 16, 16, 16]
        h = self.pool(h)                                     # [B, 32, G, G, G]
        h = self.head(h)                                     # [B, 1, G, G, G]
        return h.view(B, -1)                                 # [B, G³]


# ---------------------------------------------------------------------------
# POD-MLP  (best default for 3-D fixed-mesh problems)
# ---------------------------------------------------------------------------

class POD_MLP(nn.Module):
    """POD/PCA-based linear decoder + MLP coefficient predictor.

    Theory
    ------
    Any set of training fields U ∈ R^{N×M} admits a low-rank approximation

        U ≈ U_mean  +  C  @  V.T          C ∈ R^{N×k},  V ∈ R^{M×k}

    where V contains the top-k POD modes (right singular vectors of the
    centred snapshot matrix), and C are the projection coefficients.

    At training time we learn an MLP:  P → C_pred.
    At inference:  U_pred = C_pred @ V.T + U_mean.

    Why this beats plain DeepONet on a fixed mesh:
      - DeepONet's trunk is a LEARNED linear decoder: branch(p) @ trunk(x).T
        On a fixed mesh, trunk(x) is constant — it is equivalent to a learned
        basis matrix.  POD_MLP replaces this with the analytically-optimal
        linear basis (Eckart-Young theorem), which requires far fewer training
        samples to work well.
      - The regression target shrinks from M ≈ 3 375 to k ≈ 20–50 scalars,
        which is trivial even with 1 000 training samples.

    How many modes k?
      Run `model.explained_variance_ratio_` after calling fit() to see the
      cumulative variance curve.  For sum-of-Gaussians with 24 fixed sources,
      typically k ≈ 30 captures >99 % of variance.

    Usage
    -----
        model = POD_MLP(n_params=48, n_nodes=3375, n_modes=40)

        # Before training — fit the POD basis on all training U snapshots
        model.fit(U_train_tensor)            # U_train: [N_train, M]

        # Standard training loop (no changes needed)
        preds = model(p_batch, x_grid)       # [B, M]

    State dict / checkpointing
    --------------------------
    The mode matrix V and u_mean ARE included in state_dict (they are
    registered buffers), so torch.save / torch.load works transparently.
    You do NOT need to call fit() again after loading a checkpoint.

    Args:
        n_params:   Dimensionality of the input parameter vector.
        n_nodes:    Number of mesh nodes M (output dimensionality).
        n_modes:    Number of POD modes k to keep.
        hidden_dim: Width of the residual MLP.
        num_blocks: Depth of the residual MLP.
    """

    def __init__(self, n_params: int, n_nodes: int, n_modes: int = 40,
                 hidden_dim: int = 256, num_blocks: int = 4):
        super().__init__()
        self.n_modes = n_modes

        # Buffers are saved in state_dict; initialized to zeros so the key
        # always exists even before fit() is called.
        self.register_buffer("V",      torch.zeros(n_nodes, n_modes))
        self.register_buffer("u_mean", torch.zeros(n_nodes))
        self._fitted = False

        # MLP: predicts k coefficients from the parameter vector
        self.mlp = ResFFNN(n_params, n_modes, hidden_dim, num_blocks)

    # ------------------------------------------------------------------
    # Offline fitting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, U: torch.Tensor):
        """Compute the POD basis from training snapshots.

        Uses the economy Gram-matrix trick (O(N²M) rather than O(NM²)) when
        the number of samples N is smaller than the number of nodes M, which
        is the common case in engineering surrogate modelling.

        Args:
            U: Tensor of shape [N, M] containing ALL training field snapshots
               (un-normalised raw values, as returned by GaussianDataset).
        """
        U = U.float()
        N, M = U.shape

        u_mean = U.mean(0)                      # [M]
        Uc = U - u_mean                         # [N, M] centred

        if N <= M:
            # Gram trick: eigendecompose the small N×N covariance matrix
            G = Uc @ Uc.T / (N - 1)            # [N, N]
            eigvals, eigvecs = torch.linalg.eigh(G)   # ascending order

            # Take top-k (eigh returns ascending, so reverse)
            k = min(self.n_modes, N - 1)
            idx = torch.arange(N - 1, N - 1 - k, -1)
            eigvecs_top = eigvecs[:, idx]        # [N, k]
            eigvals_top = eigvals[idx]           # [k]

            # Right singular vectors: V = Uc.T @ phi / ||...||
            V = Uc.T @ eigvecs_top               # [M, k]
            norms = V.norm(dim=0, keepdim=True).clamp(min=1e-9)
            V = V / norms                        # [M, k]  orthonormal modes
        else:
            # Full SVD (only if N > M, unusual)
            _, _, Vh = torch.linalg.svd(Uc, full_matrices=False)
            V = Vh[:self.n_modes].T              # [M, k]

        self.V.copy_(V)
        self.u_mean.copy_(u_mean)
        self._fitted = True

        # Store explained variance for diagnostics
        total_var = (Uc ** 2).sum()
        recon_var = ((Uc @ V) @ V.T).pow(2).sum()
        self.explained_variance_ratio_ = float(recon_var / (total_var + 1e-9))
        print(f"  POD basis fitted: {self.n_modes} modes, "
              f"explained variance = {self.explained_variance_ratio_:.4f}")

    # ------------------------------------------------------------------
    # Encode / decode helpers (useful for analysis)
    # ------------------------------------------------------------------

    def encode(self, U: torch.Tensor) -> torch.Tensor:
        """Project physical fields → coefficient space. [B, M] → [B, k]"""
        return (U - self.u_mean) @ self.V

    def decode(self, C: torch.Tensor) -> torch.Tensor:
        """Reconstruct physical fields from coefficients. [B, k] → [B, M]"""
        return C @ self.V.T + self.u_mean

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, p, x_grid=None):
        C_pred = self.mlp(p)                     # [B, k]
        return self.decode(C_pred)               # [B, M]


# ---------------------------------------------------------------------------
# Coordinate-based models  (1-D and 3-D compatible)
# ---------------------------------------------------------------------------

class DeepONet(nn.Module):
    """Classical DeepONet.

    Branch network encodes the parameter vector; trunk network encodes spatial
    coordinates. Output is their inner product (dot-product).

    Compatible with any spatial_dim via MultiScaleFourierEncoding.
    """

    def __init__(self, n_params: int = 30, spatial_dim: int = 1,
                 latent_dim: int = 128, num_freqs: int = 6):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_params, 128), nn.GELU(),
            nn.Linear(128, 128),      nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        self.coord_encoder = MultiScaleFourierEncoding(spatial_dim, num_freqs)
        self.trunk = nn.Sequential(
            nn.Linear(self.coord_encoder.out_dim, 128), nn.GELU(),
            nn.Linear(128, 128),                         nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, p, x_grid):
        # x_grid: [M, spatial_dim]
        branch_out = self.branch(p)                          # [B, latent]
        trunk_out  = self.trunk(self.coord_encoder(x_grid))  # [M, latent]
        return torch.matmul(branch_out, trunk_out.t()) + self.bias  # [B, M]


class EfficientCoordinateNet(nn.Module):
    """Concatenation-based coordinate network.

    Avoids the dot-product gradient bottleneck of DeepONet by concatenating
    the parameter embedding with the Fourier-encoded coordinate at each point.
    Compatible with any spatial_dim.
    """

    def __init__(self, n_params: int = 30, spatial_dim: int = 1,
                 hidden_dim: int = 256, num_freqs: int = 6):
        super().__init__()
        self.coord_encoder = MultiScaleFourierEncoding(spatial_dim, num_freqs)
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 128), nn.GELU(),
            nn.Linear(128, 128),      nn.GELU(),
        )
        in_dim = 128 + self.coord_encoder.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),     nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p, x_grid):
        B, M = p.shape[0], x_grid.shape[0]
        p_emb = self.param_encoder(p).unsqueeze(1).expand(-1, M, -1)
        x_emb = self.coord_encoder(x_grid).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([p_emb, x_emb], dim=-1)
        return self.net(h).squeeze(-1)                       # [B, M]


# ---------------------------------------------------------------------------
# 1-D-only coordinate model
# ---------------------------------------------------------------------------

class DeepSetsCoordinateNet(nn.Module):
    """1-D only. Permutation-invariant encoding of individual Gaussians via
    DeepSets (sum + max aggregation) with cross-attention over coordinates,
    followed by a pre-norm residual trunk.

    Requires the 1-D parameter format (A, center, log10_sigma) per Gaussian,
    so it is NOT compatible with the 3-D benchmark (which has no centers in P).
    """

    def __init__(self, n_gaussians: int = 10, hidden_dim: int = 256,
                 n_layers: int = 6, fourier_feats: int = 128,
                 fourier_sigma: float = 50.0):
        super().__init__()
        self.n_gaussians = n_gaussians
        coord_dim = fourier_feats * 2

        self.param_encoder = nn.Sequential(
            nn.Linear(3, 128),          nn.GELU(),
            nn.Linear(128, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_encoder  = FourierEncoding(1, fourier_feats, fourier_sigma)
        self.attn_q         = nn.Linear(coord_dim, hidden_dim)
        self.attn_k         = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v         = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale     = hidden_dim ** -0.5
        self.global_proj    = nn.Linear(hidden_dim * 2, hidden_dim)
        trunk_in            = hidden_dim + hidden_dim + coord_dim
        self.trunk_in_proj  = nn.Linear(trunk_in, hidden_dim)
        self.trunk          = nn.Sequential(
            *[ResidualMLP(hidden_dim) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p, x_grid):
        B, M = p.shape[0], x_grid.shape[0]

        g_enc      = self.param_encoder(p.view(B, self.n_gaussians, 3))
        global_ctx = self.global_proj(
            torch.cat([g_enc.sum(1), g_enc.max(1).values], dim=-1)
        )

        x_emb    = self.coord_encoder(x_grid)                # [M, coord_dim]
        Q        = self.attn_q(x_emb).unsqueeze(0).expand(B, -1, -1)
        scores   = (torch.bmm(Q, self.attn_k(g_enc).transpose(1, 2))
                    * self.attn_scale)
        attn_out = torch.bmm(torch.softmax(scores, dim=-1), self.attn_v(g_enc))

        h = torch.cat([
            attn_out,
            global_ctx.unsqueeze(1).expand(-1, M, -1),
            x_emb.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1)
        h = self.trunk(self.trunk_in_proj(h))
        return self.head(h).squeeze(-1)                      # [B, M]


import torch_geometric.nn as pyg_nn

# ---------------------------------------------------------------------------
# Graph Neural Network (GNN / MeshGraphNet)
# ---------------------------------------------------------------------------

class GraphBlock(pyg_nn.MessagePassing):
    """A single message-passing block."""
    def __init__(self, hidden_dim: int):
        # aggr='mean' is highly stable for continuous physical fields
        super().__init__(aggr='mean') 
        
        # Edge MLP: Computes the "message" from source node to target node
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Node MLP: Updates the node state using the aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index):
        # 1. Propagate calls message() and aggregates the results across edges
        agg_msg = self.propagate(edge_index, x=x)
        # 2. Update node features
        out = self.node_mlp(torch.cat([x, agg_msg], dim=-1))
        return x + out  # Residual connection

    def message(self, x_i, x_j):
        # x_i is the target node, x_j is the source node
        return self.edge_mlp(torch.cat([x_i, x_j], dim=-1))


class MeshGraphNet(nn.Module):
    """
    GNN that builds the graph internally on the fly.
    Requires zero changes to the standard DataLoader or training loop!
    """
    def __init__(self, n_params: int, grid_res: int = 15, 
                 hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()
        self.grid_res = grid_res
        
        # 1. Generate the static edge connectivity for the 3D grid once
        base_edge_index = self._create_3d_grid_edges(grid_res)
        self.register_buffer("base_edge_index", base_edge_index)
        
        # 2. Encoders and Decoders
        # Input to node is: [x, y, z] + [p1, p2, ..., p48]
        self.node_encoder = nn.Sequential(
            nn.Linear(n_params + 3, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.processor = nn.ModuleList([
            GraphBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Map hidden dimension back to a single scalar (the physical field)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _create_3d_grid_edges(self, res: int) -> torch.Tensor:
        """Creates the 6-way connectivity list for the benchmark grid."""
        edges = []
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    curr = i * (res**2) + j * res + k
                    # Connect valid neighbors
                    if i > 0: edges.append([curr, (i - 1) * (res**2) + j * res + k])
                    if i < res - 1: edges.append([curr, (i + 1) * (res**2) + j * res + k])
                    if j > 0: edges.append([curr, i * (res**2) + (j - 1) * res + k])
                    if j < res - 1: edges.append([curr, i * (res**2) + (j + 1) * res + k])
                    if k > 0: edges.append([curr, i * (res**2) + j * res + (k - 1)])
                    if k < res - 1: edges.append([curr, i * (res**2) + j * res + (k + 1)])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, p, x_grid=None):
        B = p.shape[0]
        M = x_grid.shape[0]
        
        # 1. Expand parameters and coords to every node
        p_exp = p.unsqueeze(1).expand(-1, M, -1)     # [B, M, 48]
        x_exp = x_grid.unsqueeze(0).expand(B, -1, -1) # [B, M, 3]
        
        # 2. Flatten into a massive single batch of nodes: [B*M, 51]
        node_features = torch.cat([x_exp, p_exp], dim=-1).reshape(B * M, -1)
        
        # 3. Build the batch edge_index (shift indices for each graph in the batch)
        # We cache this to avoid recreating it at every single training step
        if hasattr(self, '_cached_batch_size') and self._cached_batch_size == B:
            batch_edge_index = self._cached_edge_index
        else:
            offsets = (torch.arange(B, device=p.device) * M).view(1, 1, B)
            # Shape math: [2, E, 1] + [1, 1, B] -> [2, E, B] -> [2, B*E]
            batch_edge_index = (self.base_edge_index.unsqueeze(2) + offsets).view(2, -1)
            self._cached_batch_size = B
            self._cached_edge_index = batch_edge_index

        # 4. GNN Forward Pass
        h = self.node_encoder(node_features)
        for block in self.processor:
            # --- NEW CODE: Gradient Checkpointing ---
            # use_reentrant=False is the modern PyTorch standard for safety
            h = checkpoint(block, h, batch_edge_index, use_reentrant=False)
            #h = block(h, batch_edge_index)
        out = self.decoder(h)  # [B*M, 1]
        
        # 5. Reshape back to [B, M] to match your exact loss function expectations!
        return out.view(B, M)