import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import yaml
import matplotlib.pyplot as plt
# Imports from our files
from dataset import GaussianDataset
from model import DeepONet, ResFFNN, FFNN, CorrectionNet, EfficientCoordinateNet, DirectFieldNet
from utils import set_seed

def analytic_gaussians(params, x, p_mean, p_std):
    # params is normalized from the DataLoader. 
    # Denormalize to get real A, c, and log10_s
    params = params * p_std.to(params.device) + p_mean.to(params.device)
    
    # params: [B, 30], x: [M,1]
    B, _ = params.shape
    params = params.view(B, -1, 3)  # [B,10,3]

    A = params[..., 0].unsqueeze(-1)
    c = params[..., 1].unsqueeze(-1)
    s = (10**params[..., 2] + 1e-9).unsqueeze(-1)

    # x: [M, 1] -> [1, 1, M]
    x_reshaped = x.view(1, 1, -1)

    return torch.sum(
        A * torch.exp(-0.5 * ((x_reshaped - c) / s) ** 2),
        dim=1
    )  # [B,M]

def main():
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cpu" if torch.cuda.is_available() else "cpu"
    # Extract parameters from yaml for cleaner code below
    # (Matches the structure of the uploaded config.yaml)
    seed = cfg["seed"]
    h5_path = cfg["data"]["h5_path"]
    batch_size = cfg["training"]["batch_size"]
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    epochs = cfg["training"]["epochs"]
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    n_params = cfg["data"]["n_gaussians"] * 3
    M = cfg["data"]["M"]
    # Number of sampled spatial points per batch
    # (critical hyperparameter)
    n_points = min(512, M)

    set_seed(seed)

    # --- Prepare Data ---
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

    # Pre-create the spatial grid based on config
    # Shape: [M, 1]
    x_grid = torch.linspace(cfg["data"]["x_min"],
                            cfg["data"]["x_max"],
                            M, dtype=torch.float32
                            ).view(-1, 1).to(device)

    # --- Model ---
    m_sensors = 100  # For DeepONet, we use 100 sensors to read the input function
    # Fixed locations where the Branch network "looks" at the input function
    x_sensors = torch.linspace(cfg["data"]["x_min"], cfg["data"]["x_max"], m_sensors).view(-1, 1).to(device)
    if model_type == "FFNN":
        model = FFNN(in_dim=n_params, out_dim=M, 
                    hidden_dims=[model_cfg["hidden_dim"]]*3)
    elif model_type == "ResFFNN":
        model = ResFFNN(in_dim=n_params, out_dim=M, 
                        hidden_dim=model_cfg["hidden_dim"], 
                        num_blocks=model_cfg["num_blocks"])
    elif model_type == "CorrectionNet":
        model = CorrectionNet(n_params=n_params, m_points=M)
    elif model_type == "DeepONet":
        model = DeepONet()
    elif model_type == "EfficientCoordinateNet":
        model = EfficientCoordinateNet()
    elif model_type == "DirectFieldNet":
        model = DirectFieldNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "best_model.pth")

    # 5. Training Loop
    print(f"Starting training on {device} for {epochs} epochs...")
    # Early stopping parameters
    patience = 30
    counter = 0
    # Get stats once to pass to the analytic function
    p_mean = train_ds.dataset.p_mean if isinstance(train_ds, Subset) else train_ds.p_mean
    p_std = train_ds.dataset.p_std if isinstance(train_ds, Subset) else train_ds.p_std

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
                x = x_grid.clone().detach().requires_grad_(True)
                # --- FORWARD PASS ---
                preds = model(p, x)       # [Batch, n_points]

                # --- LOSS COMPUTATION ---
                grad_pred = torch.autograd.grad(
                outputs=preds,
                inputs=x,                  # Differentiate with respect to 'x'
                grad_outputs=torch.ones_like(preds),
                create_graph=True,         # Crucial for backpropagating through the derivative
                retain_graph=True
                )[0] 
                
                # (The rest of your loss computation remains exactly the same)
                dx = x[1, 0] - x[0, 0]
                grad_true_fd = (u[:, 1:] - u[:, :-1]) / dx
                
                # Align shapes (since finite diff loses one point)
                grad_pred_aligned = grad_pred.squeeze(-1).unsqueeze(0).expand(p.shape[0], -1) 
                grad_pred_aligned = grad_pred_aligned[:, :-1] 
                
                loss_u = criterion(preds, u)
                loss_grad = criterion(grad_pred_aligned, grad_true_fd)
                
                loss = loss_u + 0.1 * loss_grad
            elif model_type == "EfficientCoordinateNet":
                x = x_grid.clone().detach().requires_grad_(True)
                # --- FORWARD PASS ---
                preds = model(p, x)       # [Batch, n_points]

                # --- LOSS COMPUTATION ---
                grad_pred = torch.autograd.grad(
                outputs=preds,
                inputs=x,                  # Differentiate with respect to 'x'
                grad_outputs=torch.ones_like(preds),
                create_graph=True,         # Crucial for backpropagating through the derivative
                retain_graph=True
                )[0] 
                
                # (The rest of your loss computation remains exactly the same)
                dx = x[1, 0] - x[0, 0]
                grad_true_fd = (u[:, 1:] - u[:, :-1]) / dx
                
                # Align shapes (since finite diff loses one point)
                grad_pred_aligned = grad_pred.squeeze(-1).unsqueeze(0).expand(p.shape[0], -1) 
                grad_pred_aligned = grad_pred_aligned[:, :-1] 
                
                # 1. Peak-Weighted Data Loss
                # Multiply the squared error by a weight that is higher where 'u' is large
                weight = 1.0 + 5.0 * torch.abs(u) 
                loss_u = torch.mean(weight * (preds - u)**2)
                loss_grad = criterion(grad_pred_aligned, grad_true_fd)
                
                loss = loss_u
            else:
                # Fallback for standard MLPs that output the full 2560 grid
                if model_type == "CorrectionNet":
                    preds = analytic_gaussians(p, x_grid, p_mean, p_std) + model(p)
                else:
                    preds = model(p)
                    
                loss = criterion(preds, u)

            # Backpropagation
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_rel_l2 = 0.0
        with torch.no_grad():
            for p, u in val_loader:
                p, u = p.to(device), u.to(device)
                
                if model_type == "DeepONet":
                    preds = model(p, x_grid)
                elif model_type == "EfficientCoordinateNet":
                    preds = model(p, x_grid)
                else:
                    if model_type == "CorrectionNet":
                        preds = analytic_gaussians(p, x_grid, p_mean, p_std) + model(p)
                    else:
                        preds = model(p)
                error = torch.norm(preds - u, dim=1) / (torch.norm(u, dim=1) + 1e-6)
                val_rel_l2 += error.mean().item()

                # --- NEW PLOTTING CODE (Run only once per epoch) ---
                if epoch % 5 == 0 and p.shape[0] == batch_size: # Just to plot occasionally
                    plt.figure(figsize=(8, 4))
                    # Plot the first sample in the batch
                    plt.plot(x_grid.cpu().numpy(), u[0].cpu().numpy(), label="Ground Truth (u)", color='blue')
                    plt.plot(x_grid.cpu().numpy(), preds[0].cpu().numpy(), label="Prediction", color='red', linestyle='dashed')
                    plt.title(f"Epoch {epoch+1} - Val Rel L2: {error[0].item():.4f}")
                    plt.legend()
                    plt.savefig(f"val_plot_epoch_{epoch+1}.png")
                    plt.close()
        
        avg_val_rel_l2 = val_rel_l2 / len(val_loader)
        # --- STEP THE SCHEDULER ---
        # The scheduler needs to see the validation loss to decide if it should drop the LR
        scheduler.step(avg_val_rel_l2)
        # --- LOGGING ---
        # It's helpful to see the current LR in your logs
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_rel_l2:.6f} | LR={current_lr:.2e}")

        # Inside the loop, after validation
        if avg_val_rel_l2 < best_val_loss:
            best_val_loss = avg_val_rel_l2
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
