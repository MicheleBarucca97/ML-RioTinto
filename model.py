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

class CorrectionNet(nn.Module):
    def __init__(self, n_params, m_points):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, 64),
            nn.SiLU(),
            nn.Linear(64, m_points)
        )

    def forward(self, p):
        return self.net(p)

class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim, num_feats, sigma=10.0):
        super().__init__()
        # Random Gaussian matrix for projecting coordinates
        self.register_buffer("B", torch.randn(in_dim, num_feats) * sigma)

    def forward(self, x):
        # x: [Batch, dim] or [M, dim]
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class DeepONet(nn.Module):
    def __init__(self, n_params, hidden_dim=512, latent_dim=128, num_fourier_feats=128):
        super().__init__()
        
        # --- BRANCH NET (The "Set-based" logic) ---
        # Instead of Linear(30, ...), we use Linear(3, ...)
        # This forces the model to learn the logic of ONE Gaussian and apply it to all.
        self.branch_local = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # --- TRUNK NET (Spatial structure) ---
        self.fourier = FourierFeatureEncoding(in_dim=1, num_feats=num_fourier_feats, sigma=30.0)
        trunk_in_dim = 2 * num_fourier_feats
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.bias = nn.Parameter(torch.zeros(1))

        # IMPORTANT: Initialize weights smaller so the SUM doesn't explode
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)

    def forward(self, params, x):
        # 1. Reshape params: [Batch, 30] -> [Batch, 10, 3]
        B_size = params.shape[0]
        params = params.view(B_size, 10, 3) 
        
        # 2. Branch: Process each Gaussian, then Sum (Superposition Principle)
        # Output of branch_local: [Batch, 10, latent_dim]
        branch_latents = self.branch_local(params)
        # Summing over the 10 Gaussians: [Batch, latent_dim]
        B = torch.sum(branch_latents, dim=1)

        # 3. Trunk: Process spatial grid
        # x is [M, 1] -> [M, 2*num_feats] -> [M, latent_dim]
        x_encoded = self.fourier(x)
        T = self.trunk(x_encoded)

        # 4. Dot Product (Operator Mapping)
        # [Batch, latent_dim] @ [latent_dim, M] -> [Batch, M]
        output = torch.matmul(B, T.t()) + self.bias
        return output
