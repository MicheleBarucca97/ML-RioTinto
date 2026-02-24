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
    

class CoordinateNet(nn.Module):
    def __init__(self, n_params, hidden_dim=256, layers=4):
        super().__init__()
        
        # Encode continuous spatial coordinates [M, 1] -> [M, 128]
        self.coord_encoder = FourierEncoding(1, 64, sigma=5.0) 
        
        # Encode scalar parameters [Batch, n_params] -> [Batch, 128]
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 128),
            nn.SiLU()
        )
        
        # Trunk (Combined processing)
        self.net = nn.ModuleList()
        input_dim = 256 # 128 (coords) + 128 (params)
        
        for _ in range(layers):
            self.net.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU()
            ))
            input_dim = hidden_dim

        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, p, x_grid):
        """
        p: [Batch, n_params]
        x_grid: [M, 1] (or [n_points, 1] during sub-sampling)
        """
        B_size = p.shape[0]
        M = x_grid.shape[0]

        # Expand inputs so they can be concatenated
        # p_expanded: [Batch, M, 128]
        p_expanded = p.unsqueeze(1).expand(-1, M, -1)
        p_emb = self.param_encoder(p_expanded)
        
        # x_expanded: [Batch, M, 128]
        x_expanded = x_grid.unsqueeze(0).expand(B_size, -1, -1)
        x_emb = self.coord_encoder(x_expanded)
        
        # Concatenate along the feature dimension -> [Batch, M, 256]
        h = torch.cat([p_emb, x_emb], dim=-1)
        
        # Forward pass through the MLP
        for layer in self.net:
            h = layer(h)
            
        # Output: [Batch, M]
        return self.final(h).squeeze(-1)
    
class DeepSetsCoordinateNet_old(nn.Module):
    def __init__(self, n_gaussians=10, hidden_dim=256, layers=5): # Added a layer
        super().__init__()
        self.n_gaussians = n_gaussians
        
        self.coord_encoder = FourierEncoding(1, 64, sigma=10.0) 
        
        # WIDENED ENCODER: Now outputs 256 instead of 128
        self.param_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU()
        )
        
        self.net = nn.ModuleList()
        
        # input_dim = 256 (sum params) + 256 (max params) + 128 (coords) = 640
        input_dim = 640 
        
        for _ in range(layers):
            self.net.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU()
            ))
            input_dim = hidden_dim

        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, p, x_grid):
        B_size = p.shape[0]
        M = x_grid.shape[0]

        # Reshape [Batch, 30] -> [Batch, 10, 3]
        p_reshaped = p.view(B_size, self.n_gaussians, 3)
        
        # Encode each Gaussian: [Batch, 10, 256]
        p_encoded = self.param_encoder(p_reshaped)
        
        # --- DUAL AGGREGATION ---
        p_sum = torch.sum(p_encoded, dim=1)       # [Batch, 256]
        p_max = torch.max(p_encoded, dim=1)[0]    # [Batch, 256]
        
        # Combine them: [Batch, 512]
        p_agg = torch.cat([p_sum, p_max], dim=-1)
        
        # Expand for grid: [Batch, M, 512]
        p_expanded = p_agg.unsqueeze(1).expand(-1, M, -1)

        # Encode coordinates: [Batch, M, 128]
        x_expanded = x_grid.unsqueeze(0).expand(B_size, -1, -1)
        x_emb = self.coord_encoder(x_expanded)
        
        # Concatenate everything: [Batch, M, 640]
        h = torch.cat([p_expanded, x_emb], dim=-1)
        
        for layer in self.net:
            h = layer(h)
            
        return self.final(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> Act -> Linear + skip."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# Improved DeepSets Coordinate Net
# ---------------------------------------------------------------------------
class DeepSetsCoordinateNet(nn.Module):
    """
    Key improvements over the original:

    1. Higher Fourier sigma (50 vs 10) — captures sharp Gaussians with s=0.02
    2. Per-Gaussian cross-attention with the query coordinate — lets the net
       focus on the Gaussians that matter at each x.
    3. Pre-norm residual blocks (LayerNorm + skip) — much more stable training
    4. Output head trained on NORMALISED targets — re-scaled at inference.
       (normalization is handled in the training loop, not here)
    5. Wider per-Gaussian encoder (256-D) so each Gaussian is richly encoded.
    """

    def __init__(self, n_gaussians=10, hidden_dim=256, n_layers=6,
                 fourier_feats=128, fourier_sigma=50.0):
        super().__init__()
        self.n_gaussians = n_gaussians
        coord_dim = fourier_feats * 2   # sin + cos

        # --- Per-Gaussian encoder: (A, c, log10_s) -> 256-D ---
        self.param_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Coordinate encoder ---
        self.coord_encoder = FourierEncoding(1, fourier_feats, fourier_sigma)

        # --- Cross-attention: coord queries each Gaussian ---
        # Q from coord, K/V from Gaussian encodings
        self.attn_q = nn.Linear(coord_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale = hidden_dim ** -0.5

        # --- Global aggregation (sum + max stays but AFTER attention) ---
        # After attention pool: [B, M, hidden_dim]
        # We also keep the sum/max of raw encodings as a global bias signal
        self.global_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # --- MLP trunk: input = attn_out + global_context + coord ---
        trunk_in = hidden_dim + hidden_dim + coord_dim  # 256 + 256 + 256 = 768
        self.trunk_in_proj = nn.Linear(trunk_in, hidden_dim)

        self.trunk = nn.Sequential(
            *[ResidualMLP(hidden_dim) for _ in range(n_layers)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
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
        """
        p:      [B, n_gaussians * 3]   normalised parameters
        x_grid: [M, 1]                 spatial coordinates in [-1, 1]
        returns [B, M]
        """
        B = p.shape[0]
        M = x_grid.shape[0]

        # ---- 1. Encode each Gaussian ----
        p_reshaped = p.view(B, self.n_gaussians, 3)          # [B, G, 3]
        g_enc = self.param_encoder(p_reshaped)                # [B, G, D]

        # ---- 2. Global context (permutation-invariant) ----
        g_sum = g_enc.sum(dim=1)                              # [B, D]
        g_max = g_enc.max(dim=1).values                       # [B, D]
        global_ctx = self.global_proj(
            torch.cat([g_sum, g_max], dim=-1)
        )                                                      # [B, D]

        # ---- 3. Encode coordinates ----
        x_emb = self.coord_encoder(x_grid)                    # [M, coord_dim]

        # ---- 4. Cross-attention: each x_point attends over all Gaussians ----
        # Q: [M, D], K: [B, G, D], V: [B, G, D]
        Q = self.attn_q(x_emb)                                # [M, D]
        K = self.attn_k(g_enc)                                # [B, G, D]
        V = self.attn_v(g_enc)                                 # [B, G, D]

        # scores: [B, M, G]
        Q_exp = Q.unsqueeze(0).expand(B, -1, -1)              # [B, M, D]
        scores = torch.bmm(Q_exp, K.transpose(1, 2)) * self.attn_scale
        attn_w = torch.softmax(scores, dim=-1)                # [B, M, G]
        attn_out = torch.bmm(attn_w, V)                       # [B, M, D]

        # ---- 5. Assemble trunk input ----
        x_emb_exp = x_emb.unsqueeze(0).expand(B, -1, -1)     # [B, M, coord_dim]
        global_exp = global_ctx.unsqueeze(1).expand(-1, M, -1) # [B, M, D]

        h = torch.cat([attn_out, global_exp, x_emb_exp], dim=-1)  # [B, M, trunk_in]
        h = self.trunk_in_proj(h)                             # [B, M, D]

        # ---- 6. Residual MLP trunk ----
        h = self.trunk(h)                                     # [B, M, D]

        # ---- 7. Output ----
        return self.head(h).squeeze(-1)                        # [B, M]
    

import torch
import torch.nn as nn

class MultiScaleFourierEncoding(nn.Module):
    """
    Creates deterministic frequencies: 2^0, 2^1, ..., 2^{num_frequencies-1}
    This ensures the network captures both low and high-frequency details reliably.
    """
    def __init__(self, in_dim=1, num_frequencies=10):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        # Calculate output dimension: original + (sin and cos for each frequency)
        self.out_dim = in_dim + (in_dim * 2 * num_frequencies)
        
        # Create frequencies: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        # x shape: [Batch, M, in_dim] or [M, in_dim]
        scaled_x = x.unsqueeze(-1) * self.freqs # [..., in_dim, num_frequencies]
        scaled_x = scaled_x.view(*x.shape[:-1], -1) # Flatten last two dims
        
        proj = 2 * torch.pi * scaled_x
        # Concatenate original x with sin and cos projections
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class EfficientCoordinateNet(nn.Module):
    # Notice we changed num_freqs to 6 here!
    def __init__(self, n_params=30, spatial_dim=1, hidden_dim=256, num_freqs=6):
        super().__init__()
        
        self.coord_encoder = MultiScaleFourierEncoding(spatial_dim, num_freqs)
        coord_emb_dim = self.coord_encoder.out_dim
        
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )
        
        input_dim = 128 + coord_emb_dim
        
        # Concatenation avoids the dot-product gradient trap
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, p, x_grid):
        B = p.shape[0]
        M = x_grid.shape[0]

        p_emb = self.param_encoder(p)
        p_expanded = p_emb.unsqueeze(1).expand(-1, M, -1)
        
        x_emb = self.coord_encoder(x_grid)
        x_expanded = x_emb.unsqueeze(0).expand(B, -1, -1)
        
        h = torch.cat([p_expanded, x_expanded], dim=-1)
        
        return self.net(h).squeeze(-1)
    

class DeepONet(nn.Module):
    def __init__(self, n_params=30, spatial_dim=1, latent_dim=128, num_freqs=6):
        super().__init__()
        
        # --- 1. Branch Network (Processes Parameters) ---
        # Takes the 30 parameters and creates a 128D "blueprint" of the function
        self.branch = nn.Sequential(
            nn.Linear(n_params, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
        # --- 2. Trunk Network (Processes Coordinates) ---
        # num_freqs=6 gives frequencies [1, 2, 4, 8, 16, 32]. Perfect for s=0.02.
        self.coord_encoder = MultiScaleFourierEncoding(spatial_dim, num_freqs)
        
        self.trunk = nn.Sequential(
            nn.Linear(self.coord_encoder.out_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, p, x_grid):
        """
        p: [Batch, n_params]
        x_grid: [M, spatial_dim]
        """
        # 1. Get the parameter blueprint: [Batch, latent_dim]
        branch_out = self.branch(p)
        
        # 2. Get the coordinate embeddings: [M, latent_dim]
        x_emb = self.coord_encoder(x_grid)
        trunk_out = self.trunk(x_emb)
        
        # 3. The true DeepONet dot product
        # Multiply [Batch, latent_dim] by [latent_dim, M] -> yields [Batch, M]
        out = torch.matmul(branch_out, trunk_out.t()) + self.bias
        
        return out

class DirectFieldNet(nn.Module):
    def __init__(self, in_dim=30):
        super().__init__()
        
        # Project 30 parameters to 64 channels across 40 spatial blocks
        self.fc = nn.Linear(in_dim, 64 * 40)
        
        # Upsample 40 -> 2560 using Transposed Convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=4), 
            nn.GELU(),
            # Shape: [Batch, 32, 160]
            
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=4),
            nn.GELU(),
            # Shape: [Batch, 16, 640]
            
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=4),
            # Final Shape: [Batch, 1, 2560]
        )

    def forward(self, p, x_grid=None):
        # 1. Project to low-res latent space
        h = self.fc(p)
        
        # 2. Reshape to 1D spatial channels [Batch, Channels, SpatialLength]
        h = h.view(-1, 64, 40) 
        
        # 3. Decode and squeeze out the channel dimension
        out = self.decoder(h)  
        return out.squeeze(1)
    
class CNNDecoder(nn.Module):
    def __init__(self, in_dim=30):
        super().__init__()
        
        # Project 30 parameters to 64 channels across 40 spatial blocks
        self.fc = nn.Linear(in_dim, 64 * 40)
        
        # Upsample 40 -> 2560 using Transposed Convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=4), 
            nn.GELU(),
            # Shape: [Batch, 32, 160]
            
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=4),
            nn.GELU(),
            # Shape: [Batch, 16, 640]
            
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=4),
            # Final Shape: [Batch, 1, 2560]
        )

    def forward(self, p, x_grid=None):
        # 1. Project to low-res latent space
        h = self.fc(p)
        
        # 2. Reshape to 1D spatial channels [Batch, Channels, SpatialLength]
        h = h.view(-1, 64, 40) 
        
        # 3. Decode and squeeze out the channel dimension
        out = self.decoder(h)  
        return out.squeeze(1)
