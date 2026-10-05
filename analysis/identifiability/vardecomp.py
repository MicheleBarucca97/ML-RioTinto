
import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from dataset import (GaussianDataset, load_reconstruction_context,
                     load_test_modes, reconstruct_field)
from evaluate import compute_metrics, run_inference
from utils import build_model, load_x_grid, resolve_checkpoint

for name in ["midacd", "full3d", "interface"]:
    cfg = yaml.safe_load(open("configs/config_alucell_%s.yaml" % name))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h5 = cfg["data"]["h5_path"]
    ds = GaussianDataset(h5, split="test", normalize=False)
    ld = DataLoader(ds, batch_size=16, shuffle=False)
    xg = load_x_grid(cfg, dev)
    m = build_model(cfg).to(dev)
    m.load_state_dict(torch.load(resolve_checkpoint(cfg), map_location=dev,
                                 weights_only=True))
    m.eval()
    P, T = run_inference(m, ld, xg, dev)
    ctx = load_reconstruction_context(h5)
    rp, rt = reconstruct_field(P, ctx), reconstruct_field(T, ctx)
    fs = ctx.get("field_shape") or ()
    nc = int(fs[1]) if len(fs) > 1 else 1
    met = compute_metrics(rp, rt, n_comp=nc)
    modes = np.array(load_test_modes(h5))

    print("\n=== %s ===" % name)
    for key in ("rmse", "rel_l2"):
        x = met[key]
        xbar = x.mean()
        ss_tot = ((x - xbar) ** 2).sum()
        print("  %-7s mean=%.4e  std=%.4e  (std/mean=%.2f)"
              % (key, xbar, x.std(), x.std() / xbar))
        rows = []
        for g in sorted(set(modes)):
            sel = modes == g
            ss_g = ((x[sel] - xbar) ** 2).sum()
            rows.append((100 * ss_g / ss_tot, g, sel.sum(), x[sel].mean()))
        for share, g, n, mu in sorted(rows, reverse=True):
            print("      %-9s n=%3d  mean=%.3e   share of total variance %5.1f%%"
                  % (g, n, mu, share))
        keep = modes != "weak"
        print("      -> excluding weak: mean=%.4e  std=%.4e  (std/mean=%.2f)"
              % (x[keep].mean(), x[keep].std(), x[keep].std() / x[keep].mean()))
