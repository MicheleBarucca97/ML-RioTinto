#!/usr/bin/env python3
"""Per-sample reconstructed-field figures for the two planar Alucell targets.

evaluate.py already draws truth / prediction / error triptychs for the mid-ACD
velocity and the interface, but it stacks every sample into one tall PNG, which
is unusable as a slide or a thesis figure.  This writes ONE file per sample, in
the same spirit as plot_alucell_3d.py, and picks the samples by campaign regime
so the figure shown is the one being argued about -- the design point, or the
stress case -- rather than whichever index a random seed produced.

    python plot_alucell_2d.py --config configs/config_alucell_midacd.yaml
    python plot_alucell_2d.py --config configs/config_alucell_interface.yaml \
        --regimes gaussian weak --per-regime 1

Interface figures default to the perturbation about the reference run: the
absolute elevation is a nearly flat sheet on which truth and prediction are
visually identical, and the quantity actually learned is the departure from it.
Pass --absolute to plot the elevation itself.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import yaml
from torch.utils.data import DataLoader

from dataset import (GaussianDataset, load_reconstruction_context,
                     load_test_modes, reconstruct_field)
from evaluate import compute_metrics, run_inference
from utils import build_model, load_x_grid, resolve_checkpoint, run_tag


def _select(modes, wanted, per_regime, n_total, rel=None, worst=False):
    """Pick up to `per_regime` sample indices for each requested regime.

    With worst=True the highest-error member of each regime is chosen instead
    of the first.  A stress-case figure picked by array order is an arbitrary
    draw; picked by error it is the claim being made.
    """
    if not modes:
        return list(range(min(per_regime, n_total)))
    chosen = []
    for regime in wanted:
        hits = [i for i, m in enumerate(modes) if m == regime]
        if not hits:
            print(f"  [skip] no test sample of regime '{regime}'")
            continue
        if worst and rel is not None:
            hits = sorted(hits, key=lambda i: -rel[i])
        chosen += hits[:per_regime]
    return chosen


def _panel(ax, tri, field, title, cmap, vmin, vmax, label="", xlabel=False,
           stream=None, extend="neither"):
    """One field view.

    Gouraud shading rather than filled contours: the mid-ACD velocity carries
    fine spanwise banding, and 40 contour levels over it produce a moire that
    is an artefact of the contouring, not of the data.
    """
    sc = ax.tripcolor(tri, field, shading="gouraud", cmap=cmap,
                      vmin=vmin, vmax=vmax, rasterized=True)
    cax = make_axes_locatable(ax).append_axes("right", size="1.1%", pad=0.10)
    cb = plt.colorbar(sc, cax=cax, extend=extend)
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=10)

    if stream is not None:
        xg, yg, ug, vg = stream
        ax.streamplot(xg, yg, ug, vg, color="w", linewidth=0.75,
                      density=(2.4, 0.85), arrowsize=0.8)

    ax.set_title(title, fontsize=13, pad=5)
    ax.set_aspect("equal")
    ax.set_adjustable("box")
    ax.set_xlim(tri.x.min(), tri.x.max())
    ax.set_ylim(tri.y.min(), tri.y.max())
    ax.set_ylabel("y (m)", fontsize=11)
    if xlabel:
        ax.set_xlabel("x (m)", fontsize=11)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=10)


def _stream_grid(coords, ux, uy, nx=420):
    """Resample the in-plane velocity onto a regular grid for streamplot.

    streamplot needs a rectilinear grid; the mesh is unstructured, so the two
    in-plane components are interpolated once per panel.
    """
    x, y = coords[:, 0], coords[:, 1]
    ny = max(int(nx * (y.max() - y.min()) / (x.max() - x.min())), 12)
    xg = np.linspace(x.min(), x.max(), nx)
    yg = np.linspace(y.min(), y.max(), ny)
    XG, YG = np.meshgrid(xg, yg)
    ug = griddata((x, y), ux, (XG, YG), method="linear")
    vg = griddata((x, y), uy, (XG, YG), method="linear")
    return xg, yg, np.nan_to_num(ug), np.nan_to_num(vg)


def main(config_path, regimes, per_regime, outdir, absolute, model_path, worst,
         figw, dpi):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5_path = cfg["data"]["h5_path"]
    model_path = model_path or resolve_checkpoint(cfg)

    test_ds = GaussianDataset(h5_path, split="test",
                              normalize=cfg["data"].get("normalize", False))
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    x_grid = load_x_grid(cfg, device)
    coords = x_grid.cpu().numpy()

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    preds, targets = run_inference(model, loader, x_grid, device)
    ctx = load_reconstruction_context(h5_path)
    recon_p = reconstruct_field(preds, ctx)
    recon_t = reconstruct_field(targets, ctx)
    metrics = compute_metrics(recon_p, recon_t)
    modes = load_test_modes(h5_path)

    M_grid = coords.shape[0]
    is_vector = recon_p.shape[1] == M_grid * 3
    u_ref = ctx["u_ref"]

    tag = run_tag(cfg)
    stem = "midacd" if is_vector else "interface"
    os.makedirs(outdir, exist_ok=True)

    indices = _select(modes, regimes, per_regime, len(recon_p),
                      rel=metrics['rel_l2'], worst=worst)
    print(f"{tag}: writing {len(indices)} figures to {outdir}/")

    for idx in indices:
        regime = modes[idx] if modes else "unknown"

        if is_vector:
            vt = recon_t[idx].reshape(M_grid, 3) * 1e3      # mm/s reads better
            vp = recon_p[idx].reshape(M_grid, 3) * 1e3      # than 5 decimals of m/s
            t = np.linalg.norm(vt, axis=1)
            p = np.linalg.norm(vp, axis=1)
            err = np.linalg.norm(vt - vp, axis=1)
            unit, errunit, cmap = "|u| (mm/s)", "|error| (mm/s)", "magma"
            sym = False
            st_t = _stream_grid(coords, vt[:, 0], vt[:, 1])
            st_p = _stream_grid(coords, vp[:, 0], vp[:, 1])
        else:
            t, p = recon_t[idx], recon_p[idx]
            if not absolute and u_ref is not None:
                t, p = t - u_ref, p - u_ref
                unit = r"$\Delta h$ (mm)"
                sym = True
            else:
                unit, sym = "h (mm)", False
            t, p = t * 1e3, p * 1e3
            err = np.abs(t - p)
            errunit = "|error| (mm)"
            cmap = "RdBu_r" if sym else "cividis"
            st_t = st_p = None

        if sym:
            lim = max(abs(t).max(), abs(p).max())
            vmin, vmax = -lim, lim
            extend = "neither"
        elif is_vector:
            # A handful of nodes at the cell ends run an order of magnitude
            # above the bulk.  On a full-range scale they flatten everything
            # else to one colour, so the scale is cut at the 99th percentile
            # and the bar is marked as extended.
            vmin = 0.0
            vmax = float(np.percentile(np.concatenate([t, p]), 99.0))
            extend = "max"
        else:
            vmin = min(t.min(), p.min())
            vmax = max(t.max(), p.max())
            extend = "neither"
        evmax = float(np.percentile(err, 99.0)) if is_vector else float(err.max())

        tri = Triangulation(coords[:, 0], coords[:, 1])

        # The cell is ~17 m x 3 m.  Side-by-side panels at equal aspect leave
        # most of the canvas blank, so the three views are stacked instead.
        fig, axes = plt.subplots(3, 1, figsize=(figw, figw * 0.68))
        _panel(axes[0], tri, t, "Reference solver", cmap, vmin, vmax, unit,
               stream=st_t, extend=extend)
        _panel(axes[1], tri, p, "POD-ResNet surrogate", cmap, vmin, vmax, unit,
               stream=st_p, extend=extend)
        _panel(axes[2], tri, err, "Pointwise error", "hot_r", 0.0,
               max(evmax, 1e-12), errunit, xlabel=True,
               extend="max" if is_vector else "neither")

        fig.suptitle(
            f"{stem} — sample {idx} ({regime})   "
            f"$\\varepsilon_{{rel}}$ = {metrics['rel_l2'][idx]:.4f}   "
            f"$R^2$ = {metrics['r2'][idx]:.4f}",
            fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # The absolute-elevation interface figure is a different view of the
        # same run, not a replacement for the perturbation one; keep both.
        suffix = "_full" if (absolute and not is_vector) else ""
        out = os.path.join(outdir, f"eval2d_{stem}{suffix}_{regime}_s{idx}.png")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  saved → {out}   ({regime}, rel-L2 {metrics['rel_l2'][idx]:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--regimes", nargs="+",
                    default=["gaussian", "weak", "dead"],
                    help="campaign regimes to illustrate")
    ap.add_argument("--per-regime", type=int, default=1)
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--worst", action="store_true",
                    help="pick the highest-error sample of each regime")
    ap.add_argument("--absolute", action="store_true",
                    help="interface: plot elevation, not the perturbation")
    ap.add_argument("--figwidth", type=float, default=14.0,
                    help="figure width in inches")
    ap.add_argument("--dpi", type=int, default=180)
    a = ap.parse_args()
    main(a.config, a.regimes, a.per_regime, a.outdir, a.absolute, a.model,
         a.worst, a.figwidth, a.dpi)
