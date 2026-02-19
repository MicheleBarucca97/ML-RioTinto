import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your classes
from dataset import GaussianDataset
from model import DeepONet, ResFFNN, FFNN, CorrectionNet

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

def evaluate(config_path, model_path, n_plots=5):
    # 1. Load Configuration
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Prepare Data (Test Split)
    # We use the dataset class to handle loading + input normalization
    # automatically
    h5_path = cfg["data"]["h5_path"]
    test_ds = GaussianDataset(h5_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=2)

    # 3. Initialize Model
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    n_params = cfg["data"]["n_gaussians"] * 3
    m_points = cfg["data"]["M"]

    # Pre-create the spatial grid based on config
    # Shape: [M, 1]
    x_grid = torch.linspace(cfg["data"]["x_min"],
                                cfg["data"]["x_max"],
                                cfg["data"]["M"], dtype=torch.float32).view(-1, 1).to(device)
    m_sensors = 100  # For DeepONet, we use 100 sensors to read the input function
    # Fixed locations where the Branch network "looks" at the input function
    x_sensors = torch.linspace(cfg["data"]["x_min"], cfg["data"]["x_max"], m_sensors).view(-1, 1).to(device)
    
    if model_type == "FFNN":
        model = FFNN(in_dim=n_params, out_dim=m_points, 
                    hidden_dims=[model_cfg["hidden_dim"]]*3)
    elif model_type == "ResFFNN":
        model = ResFFNN(in_dim=n_params, out_dim=m_points, 
                        hidden_dim=model_cfg["hidden_dim"], 
                        num_blocks=model_cfg["num_blocks"])
    elif model_type == "CorrectionNet":
        model = CorrectionNet(n_params=n_params, m_points=m_points)
    elif model_type == "DeepONet":
        model = DeepONet(m_sensors=m_sensors)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # 4. Load Weights
    # In classical PyTorch, we load the "state_dict"
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    all_preds = []
    all_targets = []

    # Retrieve normalization stats from the dataset object
    # They are tensors on CPU, we move them to the device for calculation
    p_mean = test_ds.p_mean.to(device)
    p_std = test_ds.p_std.to(device)

    print("Running inference...")
    with torch.no_grad():
        for p, u in test_loader:
            p = p.to(device)
            u = u.to(device)  # This is the normalized target
            
            # Generate predictions based on model type
            if model_type == "DeepONet":
                # DeepONet requires sensor readings and the evaluation grid
                u_sensors = analytic_gaussians(p, x_sensors, p_mean, p_std)
                preds = model(u_sensors, x_grid)
            elif model_type == "CorrectionNet":
                # CorrectionNet adds its output to the analytical base
                preds = analytic_gaussians(p, x_grid, p_mean, p_std) + model(p)
            else:
                # Standard models (FFNN, ResFFNN) just take the parameters
                preds = model(p)
            
            targets_real = u 

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets_real.cpu().numpy())

    # Concatenate all batches into big arrays
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # 6. Metrics
    # Calculate RMSE
    mse = np.mean((preds - targets) ** 2, axis=1)
    rmse = np.sqrt(mse)
    diff_norm = np.linalg.norm(preds - targets, axis=1)
    target_norm = np.linalg.norm(targets, axis=1)
    rel_l2 = diff_norm / (target_norm + 1e-12)

    print("=" * 40)
    print(f"Results on {len(preds)} test samples:")
    print(f"Mean RMSE: {rmse.mean():.6f}")
    print(f"Std RMSE:  {rmse.std():.6f}")
    print(f"Mean Rel L2 Error: {rel_l2.mean():.6f}")
    print("=" * 40)

    # 7. Plotting
    x_grid_plot = x_grid.cpu().numpy().flatten()

    # Select random indices
    indices = np.random.choice(len(preds), n_plots, replace=False)

    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3*n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(x_grid_plot, targets[idx], label="Ground Truth", color="black",
                alpha=0.7, linewidth=2)
        ax.plot(x_grid_plot, preds[idx], label="Prediction", color="red",
                linestyle="--", linewidth=2)

        ax.set_title(f"Sample {idx} | RMSE: {rmse[idx]:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    # Default points to where train.py saves the model
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to .pth model file")
    args = parser.parse_args()

    evaluate(args.config, args.model)
