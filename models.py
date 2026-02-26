"""
Model zoo for the Gaussian benchmark.

Uniform forward signature
-------------------------
    forward(p, x_grid) -> Tensor[B, M]

    p      : [B, n_params]      parameter vector
    x_grid : [M, spatial_dim]   spatial coordinates (1-D or 3-D)

Grid-free models (FFNN, ResFFNN, DirectFieldNet, CNNDecoder) ignore x_grid
but accept it so callers never need to branch on model type.

Benchmark compatibility
-----------------------
    1-D and 3-D : FFNN, ResFFNN, DirectFieldNet, DeepONet, EfficientCoordinateNet
    1-D only    : CNNDecoder       (hardcoded convolutional topology)
                  DeepSetsCoordinateNet  (encodes (A, center, sigma) triplets;
                                          3-D inputs have no center in P)
"""

import torch
import torch.nn as nn


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
