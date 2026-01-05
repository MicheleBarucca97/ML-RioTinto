import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml  # <--- Added this

# Imports from our files
from dataset import GaussianDataset
from model import FFNN, ResFFNN


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
    epochs = cfg["training"]["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set Seed for reproducibility
    torch.manual_seed(seed)

    # 2. Prepare Data
    print(f"Loading data from {h5_path}...")
    train_ds = GaussianDataset(h5_path, split="train")
    val_ds = GaussianDataset(h5_path, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4)

    # 3. Initialize Model
    # We can also pull architecture params from config if we want
    n_gaussians = cfg["data"]["n_gaussians"]
    m_points = cfg["data"]["M"]

    model = ResFFNN(in_dim=n_gaussians * 3, out_dim=m_points,
                    hidden_dim=256, num_blocks=3).to(device)

    # 4. Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "best_model.pth")

    # 5. Training Loop
    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False)

        for p, u in loop:
            p, u = p.to(device), u.to(device)

            optimizer.zero_grad()
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
                preds = model(p)
                loss = criterion(preds, u)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- LOGGING & SAVING ---
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f} | "
              f"Val Loss={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New best model saved! ({best_val_loss:.6f})")


if __name__ == "__main__":
    main()
