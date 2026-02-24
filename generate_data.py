import h5py
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt


def params_to_function(params, x_grid, n_gaussians):
    """Maps (N, 30) parameters -> (N, 256) function values."""
    N = params.shape[0]
    M = x_grid.shape[0]
    u = np.zeros((N, M), dtype=np.float32)

    # Vectorized computation is much faster than loops
    # params: [A1, c1, s1, A2, c2, s2 ...]
    for i in range(n_gaussians):
        A = params[:, 3*i + 0][:, None]  # Shape (N, 1)
        c = params[:, 3*i + 1][:, None]
        # Convert log10_s back to linear s for the Gaussian formula
        s_log = params[:, 3*i + 2][:, None]
        s = 10**s_log + 1e-9

        # Gaussian formula: A * exp(-0.5 * ((x-c)/s)^2)
        u += A * np.exp(-0.5 * ((x_grid[None, :] - c) / s) ** 2)
    return u


def generate():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])
    M = cfg["data"]["M"]
    n_gaussians = cfg["data"]["n_gaussians"]
    x_min = cfg["data"]["x_min"]
    x_max = cfg["data"]["x_max"]
    x_grid = np.linspace(x_min, x_max, M).astype(np.float32)

    total_samples = (cfg["data"]["samples"]["train"] +
                     cfg["data"]["samples"]["val"] +
                     cfg["data"]["samples"]["test"])

    # Generate Parameters (Input)
    print("Generating parameters...")
    # A ~ U[-1, 1], c ~ U[min, max], s ~ U[0.02, 0.25]
    A = np.random.uniform(-1, 1, (total_samples, n_gaussians))
    s = np.random.uniform(0.02, 0.25, (total_samples, n_gaussians))
    s_log = np.log10(s) # This turns 0.02 into ~ -1.7 and 0.25 into ~ -0.6

    # --- THE CRITICAL FIX: Fixed Centers ---
    # Create 10 evenly spaced centers across the domain (e.g., -0.9, -0.7 ... 0.9)
    # This mirrors your 24 fixed anodes perfectly.
    fixed_centers = np.linspace(x_min + 0.1, x_max - 0.1, n_gaussians)
    
    # Broadcast these fixed centers to all samples.
    # Shape becomes (total_samples, 10) where every row is identical.
    c = np.tile(fixed_centers, (total_samples, 1))
    
    # Interleave parameters: [A1, c1, s1, A2, c2, s2...]
    P = np.stack([A, c, s_log], axis=2) \
        .reshape(total_samples, -1) \
        .astype(np.float32)

    # Generate Functions (Target)
    print("Computing functions (ground truth)...")
    U = params_to_function(P, x_grid, n_gaussians)

    for i in range(5):
        plt.plot(x_grid, U[i], label=f"Sample {i} Function")
        plt.show()

    # Compute Normalization Stats (using only Train split conceptually)
    n_train = cfg["data"]["samples"]["train"]
    p_mean, p_std = P[:n_train].mean(0), P[:n_train].std(0) + 1e-9
    u_mean, u_std = U[:n_train].mean(0), U[:n_train].std(0) + 1e-9

    # Save to HDF5
    os.makedirs(os.path.dirname(cfg["data"]["h5_path"]), exist_ok=True)
    with h5py.File(cfg["data"]["h5_path"], "w") as f:
        # Save X Grid
        f.create_dataset("x_grid", data=x_grid)

        # Save Stats
        stats = f.create_group("stats")
        stats.create_dataset("p_mean", data=p_mean)
        stats.create_dataset("p_std", data=p_std)
        stats.create_dataset("u_mean", data=u_mean)
        stats.create_dataset("u_std", data=u_std)

        # Create Splits
        indices = {
            "train": (0, n_train),
            "val": (n_train, n_train + cfg["data"]["samples"]["val"]),
            "test": (n_train + cfg["data"]["samples"]["val"], total_samples)
        }

        for split, (start, end) in indices.items():
            print(f"Saving {split} split...")
            g = f.create_group(split)
            # Use GZIP compression to save disk space
            g.create_dataset("P", data=P[start:end], compression="gzip")
            g.create_dataset("U", data=U[start:end], compression="gzip")

    print(f"Dataset saved to {cfg['data']['h5_path']}")


if __name__ == "__main__":
    generate()
