import torch
import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[256, 256, 256]):
        super().__init__()
        layers = []
        curr_dim = in_dim

        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            # SiLU is generally better than ReLU for physics
            layers.append(nn.SiLU())
            curr_dim = h

        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)  # Skip connection


class ResFFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        # Project to hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim, num_feats, sigma=1.0):
        super().__init__()
        # Random Gaussian matrix for projecting coordinates
        self.register_buffer("B", torch.randn(in_dim, num_feats) * sigma)

    def forward(self, x):
        # x: [Batch, dim]
        # Project: x @ B -> [Batch, num_feats]
        # Output: [sin(proj), cos(proj)] -> [Batch, 2 * num_feats]
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DeepONet(nn.Module):
    def __init__(self, n_params, hidden_dim=128, latent_dim=128):
        super().__init__()

        # --- BRANCH NET (Processes Parameters) ---
        # Input: The 30 Gaussian params (or 24 Currents later)
        self.branch = nn.Sequential(
            nn.Linear(n_params, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, latent_dim)  # Output size: P
        )

        # --- TRUNK NET (Processes Coordinates) ---
        # Input: Spatial coordinate x (dim=1) or (x,y) (dim=2)
        # We use Fourier Features to help it learn sharp changes
        self.fourier = FourierFeatureEncoding(in_dim=1, num_feats=128, sigma=50.0)

        # Trunk input size is 2 * num_feats (sin + cos)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)  # Output size: P
        )

        # Bias for the final scalar output
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params, x):
        # params: [Batch, n_params]
        # x:      [M_points, 1] OR [Batch, M_points, 1]

        # 1. Branch Output: [Batch, P]
        B = self.branch(params)

        # 2. Trunk Output: [M_points, P] or [Batch, M_points, P]
        if x.dim() == 2:  # Shared grid for the whole batch [M, 1]
            x_encoded = self.fourier(x)
            # [M, P]
            T = self.trunk(x_encoded)
            # Combine using batch matrix-vector multiplication
            # [Batch, P] @ [P, M] -> [Batch, M]
            output = torch.matmul(B, T.t()) + self.bias

        else:  # Unique grid per sample [Batch, M, 1]
            batch_size, m_points, _ = x.shape
            x_flat = x.reshape(-1, 1)
            x_encoded = self.fourier(x_flat)
            T_flat = self.trunk(x_encoded)
            # [Batch, M, P]
            T = T_flat.view(batch_size, m_points, -1)
            # [Batch, 1, P]
            B_expanded = B.unsqueeze(1)
            output = torch.sum(B_expanded * T, dim=-1) + self.bias

        return output
