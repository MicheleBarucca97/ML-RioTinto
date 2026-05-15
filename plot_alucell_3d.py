"""
3D plotting for Alucell velocity field on the FLUID mesh.

After prepare_alucell.py filters to fluid-only nodes, the HDF5 stores:
    reconstruction/fluid_node_ids   (M_fluid,)   global node indices
    reconstruction/fluid_elems      (Ne_fluid, 4) local tet connectivity
    reconstruction/fluid_refs       (Ne_fluid,)   material IDs per tet

This module builds a proper PyVista UnstructuredGrid from the tet
connectivity, slices it cleanly, and renders glyphs scaled to the
actual velocity magnitude in the cell.

Requirements:
    pip install pyvista[all] h5py
"""

from __future__ import annotations

import h5py
import numpy as np
import pyvista as pv

# ── Off-screen setup ──
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def load_fluid_mesh(h5_path: str):
    """Load the fluid mesh info stored by prepare_alucell.py.

    Returns dict with keys: fluid_elems, fluid_refs, fluid_node_ids,
    M_total, or None if not available.
    """
    with h5py.File(h5_path, "r") as f:
        recon = f.get("reconstruction")
        if recon is None:
            return None

        info = {}
        if "fluid_elems" in recon:
            info["fluid_elems"] = recon["fluid_elems"][:].astype(np.int32)
        if "fluid_refs" in recon:
            info["fluid_refs"] = recon["fluid_refs"][:].astype(np.int32)
        if "fluid_node_ids" in recon:
            info["fluid_node_ids"] = recon["fluid_node_ids"][:].astype(np.int32)
        if "M_total" in recon.attrs:
            info["M_total"] = int(recon.attrs["M_total"])

    return info if "fluid_elems" in info else None


def build_tet_grid(coords: np.ndarray, elems: np.ndarray,
                   refs: np.ndarray | None = None) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid from tetrahedral connectivity.

    Parameters
    ----------
    coords : (M, 3) float   node positions
    elems  : (Ne, 4) int    0-indexed tet connectivity (LOCAL fluid indices)
    refs   : (Ne,) int      material IDs per element (optional)
    """
    Ne = elems.shape[0]
    cells = np.hstack([np.full((Ne, 1), 4, dtype=np.int64),
                       elems.astype(np.int64)]).ravel()
    cell_types = np.full(Ne, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, coords.astype(np.float64))

    if refs is not None:
        grid.cell_data["material"] = refs

    return grid


# ---------------------------------------------------------------------------
# Glyph helpers
# ---------------------------------------------------------------------------

def _auto_glyph_factor(points_or_grid, mag_key="velocity_mag",
                       arrow_fraction=0.03):
    """Scale arrows so mean arrow length ≈ arrow_fraction × domain diagonal.

    This produces readable arrows regardless of velocity units or cell size.
    """
    bounds = np.array(points_or_grid.bounds)
    domain_diag = np.linalg.norm(bounds[1::2] - bounds[::2])
    target_length = arrow_fraction * domain_diag
    mean_vel = float(points_or_grid.point_data[mag_key].mean())
    if mean_vel < 1e-12:
        return 1.0
    return target_length / mean_vel


def _make_glyphs(mesh_slice, vel_key="velocity", mag_key="velocity_mag",
                 glyph_factor=None, max_arrows=800):
    """Build glyph arrows from a sliced mesh, subsampling for readability."""
    n = mesh_slice.n_points
    if n == 0:
        return pv.PolyData()

    # Ensure we have point data (slice of UnstructuredGrid gives cell data)
    if vel_key not in mesh_slice.point_data and vel_key in mesh_slice.cell_data:
        mesh_slice = mesh_slice.cell_data_to_point_data()

    if vel_key not in mesh_slice.point_data:
        return pv.PolyData()

    # Subsample to keep the plot legible
    if n > max_arrows:
        ids = np.random.default_rng(0).choice(n, max_arrows, replace=False)
        pts = pv.PolyData(mesh_slice.points[ids])
        pts[vel_key] = mesh_slice.point_data[vel_key][ids]
        pts[mag_key] = mesh_slice.point_data[mag_key][ids]
    else:
        pts = pv.PolyData(mesh_slice.points.copy())
        pts[vel_key] = mesh_slice.point_data[vel_key].copy()
        pts[mag_key] = mesh_slice.point_data[mag_key].copy()

    # Drop near-zero arrows (1% of max)
    mag = pts[mag_key]
    cutoff = 0.01 * mag.max() if mag.max() > 0 else 1e-12
    keep = mag > cutoff
    if keep.sum() == 0:
        return pv.PolyData()
    pts = pts.extract_points(keep)

    if glyph_factor is None:
        glyph_factor = _auto_glyph_factor(pts, mag_key)

    return pts.glyph(orient=vel_key, scale=mag_key,
                     factor=glyph_factor, geom=pv.Arrow())


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_alucell_3d(preds, targets, x_grid_tensor, metrics, indices, cfg,
                    recon_preds=None, recon_targets=None,
                    n_slices: int = 3, save_prefix: str = "eval3d"):
    """
    Parameters
    ----------
    preds, targets : (N_test, M_out)    raw model output
    x_grid_tensor  : (M_fluid, 3)       fluid node coordinates
    metrics        : dict               from compute_metrics
    indices        : array-like         which test samples to plot
    cfg            : dict               config dict
    recon_preds / recon_targets : (N, M_field) reconstructed physical fields
    n_slices       : number of horizontal z-slices
    save_prefix    : filename prefix
    """
    coords = x_grid_tensor.cpu().numpy() if hasattr(x_grid_tensor, 'cpu') \
        else np.asarray(x_grid_tensor)
    M = coords.shape[0]

    plot_preds   = recon_preds if recon_preds is not None else preds
    plot_targets = recon_targets if recon_targets is not None else targets

    M_out = plot_preds.shape[1]
    is_vector = (M_out == M * 3)
    if not is_vector:
        print(f"[plot_alucell_3d] M_out={M_out} != M*3={M*3}. Skipping.")
        return

    # ── Try loading fluid mesh connectivity from prepared HDF5 ──
    h5_path = cfg["data"]["h5_path"]
    mesh_info = load_fluid_mesh(h5_path)
    has_tets = (mesh_info is not None)

    if has_tets:
        base_grid = build_tet_grid(coords, mesh_info["fluid_elems"],
                                   mesh_info.get("fluid_refs"))
        print(f"  Fluid mesh: {base_grid.n_points} nodes, "
              f"{base_grid.n_cells} tets")
    else:
        print("  No fluid mesh in HDF5 — using point cloud fallback.")
        base_grid = None

    # ── Slice z-levels (avoid very top/bottom of domain) ──
    z_all = coords[:, 2]
    z_min, z_max = z_all.min(), z_all.max()
    z_pad = 0.08 * (z_max - z_min)
    z_levels = np.linspace(z_min + z_pad, z_max - z_pad, n_slices)

    for sample_idx in indices:
        vel_true = plot_targets[sample_idx].reshape(M, 3)
        vel_pred = plot_preds[sample_idx].reshape(M, 3)
        vel_err  = vel_true - vel_pred
        err_mag  = np.linalg.norm(vel_err, axis=1)

        rel_l2 = metrics["rel_l2"][sample_idx]

        # ── Attach velocity to grids ──
        if has_tets:
            grid_t = base_grid.copy()
            grid_p = base_grid.copy()
            grid_e = base_grid.copy()
        else:
            grid_t = pv.PolyData(coords.astype(np.float64))
            grid_p = pv.PolyData(coords.astype(np.float64))
            grid_e = pv.PolyData(coords.astype(np.float64))

        grid_t.point_data["velocity"] = vel_true
        grid_t.point_data["velocity_mag"] = np.linalg.norm(vel_true, axis=1)
        grid_p.point_data["velocity"] = vel_pred
        grid_p.point_data["velocity_mag"] = np.linalg.norm(vel_pred, axis=1)
        grid_e.point_data["velocity"] = vel_err
        grid_e.point_data["velocity_mag"] = err_mag
        grid_e.point_data["error_mag"] = err_mag

        # ── One glyph factor for all panels (from truth) ──
        gf = _auto_glyph_factor(grid_t, arrow_fraction=0.03)

        # ── Horizontal slice plots ──
        for iz, zl in enumerate(z_levels):
            if has_tets:
                sl_t = grid_t.slice(normal="z", origin=(0, 0, zl))
                sl_p = grid_p.slice(normal="z", origin=(0, 0, zl))
                sl_e = grid_e.slice(normal="z", origin=(0, 0, zl))
            else:
                for g in (grid_t, grid_p, grid_e):
                    g.point_data["z_coord"] = g.points[:, 2]
                tol = max(0.005, 0.005 * (z_max - z_min))
                sl_t = grid_t.threshold([zl - tol, zl + tol], scalars="z_coord")
                sl_p = grid_p.threshold([zl - tol, zl + tol], scalars="z_coord")
                sl_e = grid_e.threshold([zl - tol, zl + tol], scalars="z_coord")

            if sl_t.n_points == 0:
                print(f"  WARNING: empty slice at z={zl:.4f}, skipping.")
                continue

            # Shared color scale
            def _get_mag(sl, key="velocity_mag"):
                if key in sl.point_data:
                    return sl.point_data[key]
                if key in sl.cell_data:
                    return sl.cell_data[key]
                return np.array([0.0])

            vmin = min(_get_mag(sl_t).min(), _get_mag(sl_p).min())
            vmax = max(_get_mag(sl_t).max(), _get_mag(sl_p).max())

            glyph_t = _make_glyphs(sl_t, glyph_factor=gf)
            glyph_p = _make_glyphs(sl_p, glyph_factor=gf)

            # ── 3-panel render ──
            pl = pv.Plotter(shape=(1, 3), off_screen=True,
                            window_size=[2400, 700])

            pl.subplot(0, 0)
            pl.add_mesh(sl_t, scalars="velocity_mag",
                        cmap="coolwarm", clim=[vmin, vmax],
                        show_edges=False,
                        scalar_bar_args={"title": "|u| (m/s)", "fmt": "%.3f"})
            if glyph_t.n_points > 0:
                pl.add_mesh(glyph_t, color="black", opacity=0.5)
            pl.add_text(f"Truth  z={zl:.3f} m", font_size=10)
            pl.view_xy()

            pl.subplot(0, 1)
            pl.add_mesh(sl_p, scalars="velocity_mag",
                        cmap="coolwarm", clim=[vmin, vmax],
                        show_edges=False,
                        scalar_bar_args={"title": "|u| (m/s)", "fmt": "%.3f"})
            if glyph_p.n_points > 0:
                pl.add_mesh(glyph_p, color="black", opacity=0.5)
            pl.add_text(f"Prediction  Rel-L2={rel_l2:.4f}", font_size=10)
            pl.view_xy()

            pl.subplot(0, 2)
            pl.add_mesh(sl_e, scalars="error_mag", cmap="hot_r",
                        show_edges=False,
                        scalar_bar_args={"title": "|error| (m/s)", "fmt":"%.4f"})
            pl.add_text("Pointwise error", font_size=10)
            pl.view_xy()

            fname = f"{save_prefix}_sample{sample_idx}_z{iz}.png"
            pl.screenshot(fname)
            pl.close()
            print(f"  Saved → {fname}")

        # ── Full 3D glyph overview ──
        n_gl = min(3000, M)
        rng = np.random.default_rng(42)
        ids = rng.choice(M, n_gl, replace=False)

        sub = pv.PolyData(coords[ids].astype(np.float64))
        sub["velocity"] = vel_pred[ids]
        sub["velocity_mag"] = np.linalg.norm(vel_pred[ids], axis=1)

        # Drop low-velocity arrows for clarity
        mag = sub["velocity_mag"]
        cutoff = 0.05 * mag.max() if mag.max() > 0 else 0
        active = mag > cutoff
        if active.sum() > 10:
            sub = sub.extract_points(active)

        gf_3d = _auto_glyph_factor(sub, arrow_fraction=0.025)
        glyphs_3d = sub.glyph(orient="velocity", scale="velocity_mag",
                              factor=gf_3d, geom=pv.Arrow())

        pl3 = pv.Plotter(off_screen=True, window_size=[1400, 1000])
        pl3.add_mesh(glyphs_3d, scalars="velocity_mag", cmap="coolwarm",
                     scalar_bar_args={"title": "|u| (m/s)", "fmt": "%.3f"})

        # Translucent domain outline
        if has_tets:
            outline = base_grid.extract_surface()
            pl3.add_mesh(outline, color="grey", opacity=0.08,
                         show_edges=False)

        pl3.add_text(f"Sample {sample_idx} — 3D prediction   "
                     f"Rel-L2={rel_l2:.4f}", font_size=10)
        pl3.camera.elevation = 25
        pl3.camera.azimuth = 40
        pl3.camera.zoom(1.2)

        fname = f"{save_prefix}_sample{sample_idx}_3d.png"
        pl3.screenshot(fname)
        pl3.close()
        print(f"  Saved → {fname}")

    print(f"\nAll 3D evaluation plots saved with prefix '{save_prefix}_'")