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
