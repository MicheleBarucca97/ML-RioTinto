#!/usr/bin/env python3
"""Reconstructed-field figures for the full 3-D velocity surrogate.

plot_alucell_3d.py renders PyVista glyphs, which look nothing like the figures
for the other two targets and are hard to read in print.  This slices the same
tetrahedral field and renders the slices in matplotlib in exactly the style of
plot_alucell_2d.py -- speed as colour, streamlines in white, solver / surrogate
/ error stacked -- so the three surrogates can be compared by eye.

PyVista does the slicing because the mesh is a genuine tet mesh with no layer
structure (116k distinct z values over 0.33 m), so a slab-and-project shortcut
would smear the field; slice() interpolates on the elements.

    python plot_alucell_3d_slices.py --config configs/config_alucell_full3d.yaml
    python plot_alucell_3d_slices.py --config configs/config_alucell_full3d.yaml \
        --regimes weak --worst
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
import pyvista as pv
import torch
import yaml
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from torch.utils.data import DataLoader

from dataset import (GaussianDataset, load_reconstruction_context,
                     load_test_modes, reconstruct_field)
from evaluate import compute_metrics, run_inference
from plot_alucell_3d import build_tet_grid, load_fluid_mesh
from utils import build_model, load_x_grid, resolve_checkpoint


def _select(modes, wanted, per_regime, n_total, rel=None, worst=False):
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


def _stream_grid(xy, ux, uy, nx=420, aspect=1.0):
    x, y = xy[:, 0], xy[:, 1]
    span_x, span_y = x.max() - x.min(), y.max() - y.min()
    ny = max(int(nx * (span_y / span_x) * aspect), 14)
    xg = np.linspace(x.min(), x.max(), nx)
    yg = np.linspace(y.min(), y.max(), ny)
    XG, YG = np.meshgrid(xg, yg)
    ug = griddata((x, y), ux, (XG, YG), method="linear")
    vg = griddata((x, y), uy, (XG, YG), method="linear")
    return xg, yg, np.nan_to_num(ug), np.nan_to_num(vg)


def _panel(ax, tri, field, title, cmap, vmin, vmax, label, xlabel, ylabel,
           stream=None, extend="neither", aspect=1.0, density=(2.4, 0.85)):
    sc = ax.tripcolor(tri, field, shading="gouraud", cmap=cmap,
                      vmin=vmin, vmax=vmax, rasterized=True)
    cax = make_axes_locatable(ax).append_axes("right", size="1.1%", pad=0.10)
    cb = plt.colorbar(sc, cax=cax, extend=extend)
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=10)
    if stream is not None:
        xg, yg, ug, vg = stream
        ax.streamplot(xg, yg, ug, vg, color="w", linewidth=0.75,
                      density=density, arrowsize=0.8)
    ax.set_title(title, fontsize=13, pad=5)
    ax.set_aspect(aspect)
    ax.set_adjustable("box")
    ax.set_xlim(tri.x.min(), tri.x.max())
    ax.set_ylim(tri.y.min(), tri.y.max())
    ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel("x (m)", fontsize=11)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=10)


def _slice_fields(grid, vt, vp, normal, origin):
    """Slice solver and surrogate on the same plane; return 2-D coords + data."""
    grid.point_data["v_t"] = vt
    grid.point_data["v_p"] = vp
    sl = grid.slice(normal=normal, origin=origin)
    if sl.n_points == 0:
        return None
    pts = sl.points
    if normal == "z":
        xy = pts[:, [0, 1]]
        comp = [0, 1]
        ylabel = "y (m)"
    else:                                   # normal 'y' -> an x-z plane
        xy = pts[:, [0, 2]]
        comp = [0, 2]
        ylabel = "z (m)"
    return xy, sl.point_data["v_t"], sl.point_data["v_p"], comp, ylabel


def main(a):
    cfg = yaml.safe_load(open(a.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5 = cfg["data"]["h5_path"]

    ds = GaussianDataset(h5, split="test",
                         normalize=cfg["data"].get("normalize", False))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    x_grid = load_x_grid(cfg, device)
    coords = x_grid.cpu().numpy().astype(np.float64)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(a.model or resolve_checkpoint(cfg),
                                     map_location=device, weights_only=True))
    model.eval()

    preds, targets = run_inference(model, loader, x_grid, device)
    ctx = load_reconstruction_context(h5)
    recon_p, recon_t = reconstruct_field(preds, ctx), reconstruct_field(targets, ctx)
    metrics = compute_metrics(recon_p, recon_t)
    modes = load_test_modes(h5)

    info = load_fluid_mesh(h5)
    if info is None:
        raise SystemExit("no fluid_elems in the prepared file; re-run prepare_alucell.py")
    grid = build_tet_grid(coords, info["fluid_elems"], info.get("fluid_refs"))
    M = coords.shape[0]
    zmin, zmax = coords[:, 2].min(), coords[:, 2].max()
    print(f"fluid mesh: {M} nodes, {info['fluid_elems'].shape[0]} tets, "
          f"z in [{zmin:.3f}, {zmax:.3f}]")

    os.makedirs(a.outdir, exist_ok=True)
    indices = _select(modes, a.regimes, a.per_regime, len(recon_p),
                      rel=metrics["rel_l2"], worst=a.worst)

    for idx in indices:
        regime = modes[idx] if modes else "unknown"
        vt = recon_t[idx].reshape(M, 3) * 1e3          # mm/s
        vp = recon_p[idx].reshape(M, 3) * 1e3

        views = [("z", (0.0, 0.0, a.zlevel), f"z = {a.zlevel:.3f} m (ACD gap)", 1.0,
                  f"z{int(round(a.zlevel*1000)):03d}"),
                 ("y", (0.0, 0.0, 0.0), "mid-plane y = 0", a.zexag, "mid")]

        for normal, origin, what, aspect, tag in views:
            out = _slice_fields(grid, vt, vp, normal, origin)
            if out is None:
                print(f"  [skip] empty slice {what}")
                continue
            xy, st, sp, comp, ylabel = out

            t = np.linalg.norm(st, axis=1)
            p = np.linalg.norm(sp, axis=1)
            err = np.linalg.norm(st - sp, axis=1)

            vmax = float(np.percentile(np.concatenate([t, p]), 99.0))
            evmax = float(np.percentile(err, 99.0))
            tri = Triangulation(xy[:, 0], xy[:, 1])
            str_t = _stream_grid(xy, st[:, comp[0]], st[:, comp[1]], aspect=aspect)
            str_p = _stream_grid(xy, sp[:, comp[0]], sp[:, comp[1]], aspect=aspect)
            dens = (2.4, 0.85) if normal == "z" else (2.4, 0.7)

            fig, axes = plt.subplots(3, 1, figsize=(a.figwidth, a.figwidth * 0.68))
            _panel(axes[0], tri, t, "Reference solver", "magma", 0.0, vmax,
                   "|u| (mm/s)", False, ylabel, str_t, "max", aspect, dens)
            _panel(axes[1], tri, p, "POD-ResNet surrogate", "magma", 0.0, vmax,
                   "|u| (mm/s)", False, ylabel, str_p, "max", aspect, dens)
            _panel(axes[2], tri, err, "Pointwise error", "hot_r", 0.0,
                   max(evmax, 1e-12), "|error| (mm/s)", True, ylabel, None,
                   "max", aspect, dens)

            extra = (f"   (vertical scale $\\times${a.zexag:g})"
                     if normal == "y" else "")
            fig.suptitle(
                f"full 3-D velocity, {what}{extra} — sample {idx} ({regime})   "
                f"$\\varepsilon_{{rel}}$ = {metrics['rel_l2'][idx]:.4f}   "
                f"$R^2$ = {metrics['r2'][idx]:.4f}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fn = os.path.join(a.outdir, f"eval3d_{tag}_{regime}_s{idx}.png")
            fig.savefig(fn, dpi=a.dpi)
            plt.close(fig)
            print(f"  saved → {fn}   ({regime}, rel-L2 {metrics['rel_l2'][idx]:.4f}, "
                  f"|u|max {t.max():.1f} errmax {err.max():.2f} mm/s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--regimes", nargs="+", default=["gaussian", "weak"])
    ap.add_argument("--per-regime", type=int, default=1)
    ap.add_argument("--worst", action="store_true")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--zlevel", type=float, default=0.17,
                    help="height of the horizontal slice.  z is the vertical "
                         "coordinate.  Measured on this campaign: the metal-bath "
                         "interface sits at z = 0.080-0.167 m (mean 0.149) and the "
                         "mid-ACD surface at z = 0.101-0.183 m (mean 0.166), so the "
                         "ACD gap is the band z ~ 0.15-0.18 and 0.17 cuts through it. "
                         "Note the mid-ACD surface is NOT flat: it follows the "
                         "interface about 17 mm above it.")
    ap.add_argument("--zexag", type=float, default=8.0,
                    help="vertical exaggeration of the mid-plane slice")
    ap.add_argument("--figwidth", type=float, default=14.0)
    ap.add_argument("--dpi", type=int, default=180)
    main(ap.parse_args())
