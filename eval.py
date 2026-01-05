import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your classes
from dataset import GaussianDataset
from model import FFNN, ResFFNN


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
    # Must match the architecture in train.py exactly
    in_dim = cfg["data"]["n_gaussians"] * 3
    out_dim = cfg["data"]["M"]

    model = ResFFNN(in_dim=in_dim, out_dim=out_dim,
                    hidden_dim=256, num_blocks=3).to(device)

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
    u_mean = test_ds.u_mean.to(device)
    u_std = test_ds.u_std.to(device)

    print("Running inference...")
    with torch.no_grad():
        for p, u in test_loader:
            p = p.to(device)
            u = u.to(device)  # This is the normalized target

            # Predict
            preds_norm = model(p)

            # Denormalize: Real = Norm * Std + Mean
            preds_real = preds_norm * u_std + u_mean
            targets_real = u * u_std + u_mean

            all_preds.append(preds_real.cpu().numpy())
            all_targets.append(targets_real.cpu().numpy())

    # Concatenate all batches into big arrays
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # 6. Metrics
    # Calculate RMSE
    mse = np.mean((preds - targets) ** 2, axis=1)
    rmse = np.sqrt(mse)

    print("=" * 40)
    print(f"Results on {len(preds)} test samples:")
    print(f"Mean RMSE: {rmse.mean():.6f}")
    print(f"Std RMSE:  {rmse.std():.6f}")
    print("=" * 40)

    # 7. Plotting
    # Load x_grid manually just for plotting (it's in the HDF5 metadata)
    import h5py
    with h5py.File(h5_path, "r") as f:
        x_grid = f["x_grid"][:]

    # Select random indices
    indices = np.random.choice(len(preds), n_plots, replace=False)

    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3*n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(x_grid, targets[idx], label="Ground Truth", color="black",
                alpha=0.7, linewidth=2)
        ax.plot(x_grid, preds[idx], label="Prediction", color="red",
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
