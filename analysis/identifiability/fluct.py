
import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import h5py, numpy as np, torch, yaml
from torch.utils.data import DataLoader
from dataset import GaussianDataset
from evaluate import run_inference
from utils import build_model, load_x_grid, resolve_checkpoint

print("%-10s %8s %10s %10s %10s %10s" % (
      "field", "k", "evalR2", "1-eps^2", "R2_run", "|mean_j c|"))
for name in ["midacd", "full3d", "interface"]:
    cfg = yaml.safe_load(open("configs/config_alucell_%s.yaml" % name))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = GaussianDataset(cfg["data"]["h5_path"], split="test", normalize=False)
    ld = DataLoader(ds, batch_size=16, shuffle=False)
    xg = load_x_grid(cfg, dev)
    m = build_model(cfg).to(dev)
    m.load_state_dict(torch.load(resolve_checkpoint(cfg), map_location=dev,
                                 weights_only=True))
    m.eval()
    P, T = run_inference(m, ld, xg, dev)          # [N, k] POD coefficients
    N, k = T.shape

    # exactly what evaluate.py does in model-output space
    ss_res = ((T - P) ** 2).sum(1)
    ss_tot = ((T - T.mean(1, keepdims=True)) ** 2).sum(1)   # mean over MODES
    r2_eval = (1 - ss_res / (ss_tot + 1e-12)).mean()

    eps = np.linalg.norm(T - P, axis=1) / (np.linalg.norm(T, axis=1) + 1e-12)
    one_minus_eps2 = (1 - eps ** 2).mean()

    # baseline = mean over RUNS, pooled (what compare_baselines reports)
    r2_run = 1 - ((T - P) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()

    # how far the within-run mode-mean is from zero, relative to the coeff scale
    rel_mean = (np.abs(T.mean(1)) / (np.linalg.norm(T, axis=1) / np.sqrt(k))).mean()
    print("%-10s %8d %10.4f %10.4f %10.4f %10.4f" % (
        name, k, r2_eval, one_minus_eps2, r2_run, rel_mean))
