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

class FourierEncoding(nn.Module):
    def __init__(self, in_dim=1, num_feats=128, sigma=3.):
        super().__init__()
        B = torch.randn(in_dim, num_feats) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# -------------------------------------------------
# DeepONet operator
# -------------------------------------------------
class DeepONet(nn.Module):
    def __init__(self, m_sensors=100, spatial_dim=1, hidden_dim=256, latent_dim=128):
        super().__init__()
        
        # --- BRANCH NETWORK ---
        # Takes the 100 discrete sensor readings of the input function
        self.branch = nn.Sequential(
            nn.Linear(m_sensors, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # --- TRUNK NETWORK (Fourier-Embedded) ---
        num_fourier_feats = 64 # Yields an output dim of 128 (64 sin + 64 cos)
        # We use a high sigma (~20.0) to capture the sharp 0.02 width of your Gaussians
        self.fourier = FourierEncoding(spatial_dim, num_fourier_feats, sigma=20.0)
        
        self.trunk = nn.Sequential(
            # Input is now the 128D Fourier feature vector, NOT the 1D raw coordinate
            nn.Linear(num_fourier_feats * 2, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, zeta):
        """
        u_sensors: [Batch, m_sensors] (The function readings)
        zeta: [M, spatial_dim] (The spatial grid to evaluate on)
        """
        B = self.branch(u_sensors)  
        
        # Pass spatial coordinates through the Fourier feature map first
        zeta_emb = self.fourier(zeta)
        T = self.trunk(zeta_emb)        
        
        return torch.matmul(B, T.t()) + self.bias