import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import yaml

# Imports from our files
from dataset import GaussianDataset
from model import DeepONet, ResFFNN, FFNN
from utils import set_seed

# --- Define the Relative L2 Loss function ---
class RelL2Loss(nn.Module):
    def forward(self, pred, target):
        # Calculate L2 norm across the spatial dimension (dim=1)
        num = torch.norm(pred - target, p=2, dim=1)
        den = torch.norm(target, p=2, dim=1)
        # We use mean of the ratio for the batch loss
        return torch.mean(num / (den + 1e-7))

def weighted_mse_loss(pred, target):
    # Weight the loss: points where target > 0.1 get 10x more importance
    weights = torch.where(target > 0.1, 10.0, 1.0)
    return torch.mean(weights * (pred - target) ** 2)

def main():
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Extract parameters from yaml for cleaner code below
    # (Matches the structure of the uploaded config.yaml)
    seed = cfg["seed"]
    h5_path = cfg["data"]["h5_path"]
    batch_size = cfg["training"]["batch_size"]
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    epochs = cfg["training"]["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    # 2. Prepare Data
    print(f"Loading data from {h5_path}...")
    train_ds = GaussianDataset(h5_path, split="train")
    val_ds = GaussianDataset(h5_path, split="val")

    if cfg["data"].get("use_subset", False):
        num_train_samples = cfg["data"]["samples"]["train"]
        
        # Use a local generator for reproducibility
        rng = np.random.default_rng(seed)
        indices = np.arange(len(train_ds))
        rng.shuffle(indices)
        
        # Wrap the dataset in a Subset
        train_indices = indices[:num_train_samples]
        train_ds = Subset(train_ds, train_indices)
        print(f"Using a subset of {num_train_samples} training samples.")
    else:
        print(f"Using full training set: {len(train_ds)} samples.")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4)

    # 3. Initialize Model
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    n_params = cfg["data"]["n_gaussians"] * 3
    m_points = cfg["data"]["M"]

    if model_type == "FFNN":
        model = FFNN(in_dim=n_params, out_dim=m_points, 
                    hidden_dims=[model_cfg["hidden_dim"]]*3)
    elif model_type == "ResFFNN":
        model = ResFFNN(in_dim=n_params, out_dim=m_points, 
                        hidden_dim=model_cfg["hidden_dim"], 
                        num_blocks=model_cfg["num_blocks"])
    elif model_type == "DeepONet":
        model = DeepONet(n_params=n_params, 
                        hidden_dim=model_cfg["hidden_dim"], 
                        latent_dim=model_cfg["latent_dim"])
        # Pre-create the spatial grid based on config
        # Shape: [M, 1]
        x_grid = torch.linspace(cfg["data"]["x_min"],
                                cfg["data"]["x_max"],
                                cfg["data"]["M"]).to(device).unsqueeze(-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # 4. Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = weighted_mse_loss

    best_val_loss = float("inf")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "best_model.pth")

    # 5. Training Loop
    print(f"Starting training on {device} for {epochs} epochs...")
    # Early stopping parameters
    patience = 15
    counter = 0
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False)

        for p, u in loop:
            p, u = p.to(device), u.to(device)

            optimizer.zero_grad()
            if model_type == "DeepONet":
                preds = model(p, x_grid)
            else:
                preds = model(p)
            loss = criterion(preds, u)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p, u in val_loader:
                p, u = p.to(device), u.to(device)
                preds = model(p, x_grid) if model_type == "DeepONet" else model(p)
                loss = criterion(preds, u)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- STEP THE SCHEDULER ---
        # The scheduler needs to see the validation loss to decide if it should drop the LR
        scheduler.step(avg_val_loss)
        # --- LOGGING ---
        # It's helpful to see the current LR in your logs
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.6f} | LR={current_lr:.2e}")

        # --- LOGGING & SAVING ---
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f} | "
              f"Val Loss={avg_val_loss:.6f}")

        # Inside the loop, after validation
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0 # Reset counter
            print(f"  >>> New best model saved! ({best_val_loss:.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


if __name__ == "__main__":
    main()
