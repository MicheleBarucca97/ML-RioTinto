"""
prepare_alucell.py — Prepare training data from Alucell simulations.

Two orthogonal flags control the dataset:

  --mapping    WHAT to extract (which physical field)
  --pod        HOW to reduce it (optional offline POD → coefficient target)
  --delta      Subtract reference solution before anything else

Mappings  (--mapping)
--------
  velocity_midacd   I[24] → fiber-averaged velocity on mid-ACD plane   [M_mid × 3]
  velocity_full     I[24] → full 3D velocity field (all nodes)         [M_full × 3]
  interface         I[24] → interface displacement h(x,y)              [M_int]

Reduction  (--pod)
---------
  Without --pod:  The raw (possibly delta-subtracted) field is stored.
                  Use POD_MLP at training time (POD is done inside the model)
                  or ResFFNN for a direct mapping (only feasible if M is small).

  With --pod:     POD is computed in this script; only the k scalar
                  coefficients are stored as the training target.
                  Use ResFFNN at training time (do NOT use POD_MLP — it would
                  apply a second POD on top of the coefficients).
                  The basis V, mean, and explained variance are saved under
                  reconstruction/ for field recovery at evaluation time.

  The coupled-POD approach is used for vector fields: all 3 velocity
  components are flattened into a single vector of length M×3 before SVD.
  Each mode therefore captures correlated patterns across all components,
  which respects the physical coupling (continuity, Lorentz force).

Delta-learning  (--delta)
--------------
  Subtracts the reference field u_ref (auto-detected or specified via
  --reference-run) from every snapshot BEFORE optional POD.  The model
  trains on the perturbation Δu = u − u_ref.

  Reconstruction at evaluation time:
    Without POD:  u_pred = u_ref + Δu_pred
    With POD:     u_pred = u_ref + u_pod_mean + V · c_pred

Usage examples
--------------
  # Mid-ACD velocity, delta, let POD_MLP handle reduction
  python prepare_alucell.py --master ../ml_alu_data/master_dataset.h5 --manifest ../ml_alu_data/manifest.csv --mapping velocity_midacd --delta --output data/midacd_delta.h5

  # Full 3D velocity, delta + offline POD → small coefficient target
  python prepare_alucell.py \\
      --master ml_alu_data/master_dataset.h5 --manifest ml_alu_data/manifest.csv \\
      --mapping velocity_full --delta --pod --n-modes 50 \\
      --output data/full3d_pod_delta.h5

  # Interface, delta, no POD (POD_MLP at training time)
  python prepare_alucell.py \\
      --master ml_alu_data/master_dataset.h5 --manifest ml_alu_data/manifest.csv \\
      --mapping interface --delta \\
      --output data/interface_delta.h5

Output HDF5 schema
-------------------
  meta/                       attrs: mapping, delta_learning, pod_reduction,
                                     reference_run, n_params, M_out, M_field,
                                     field_shape, input_norm
  x_grid          [M_grid, d] spatial coords (for plotting)
  stats/                      p_mean, p_std, u_mean, u_std (from train split)
  reconstruction/             u_ref     [M_field]    (if delta)
                              V         [M_field, k] (if pod)
                              u_pod_mean [M_field]   (if pod)
                              evr       [k]          (if pod)
                              cumevr    [k]          (if pod)
  {train,val,test}/P          [N, 24]
  {train,val,test}/U          [N, M_out]   (M_out = k if pod, else M_field)
  {train,val,test}/modes      [N] string
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys

import h5py
import numpy as np
from scipy.interpolate import griddata


# ======================================================================
# Constants
# ======================================================================

N_ANODES = 24
I_TOTAL  = 490_000.0
I_MEAN   = I_TOTAL / N_ANODES


# ======================================================================
# Helpers
# ======================================================================

def _get_run_names(master, dead_thresh, exclude_dead):
    names = []
    for key in sorted(master.keys()):
        grp = master[key]
        if not isinstance(grp, h5py.Group):
            continue
        if "input/currents" not in grp:
            continue
        if exclude_dead:
            curr = grp["input/currents"][:].ravel()
            if np.any(curr < dead_thresh):
                continue
        names.append(key)
    return names


def _find_reference_run(master, run_names, user_ref):
    if user_ref is not None:
        if user_ref not in master:
            sys.exit(f"Reference run '{user_ref}' not found in HDF5.")
        return user_ref
    best_name, best_dist = None, np.inf
    target = np.full(N_ANODES, I_MEAN)
    for name in run_names:
        curr = master[name]["input/currents"][:].ravel()[:N_ANODES]
        dist = float(np.linalg.norm(curr - target))
        if dist < best_dist:
            best_dist, best_name = dist, name
    print(f"  Auto-detected reference: {best_name}  "
          f"(dist from uniform = {best_dist:.1f} A)")
    return best_name


def _check_consistent_grid(master, run_names, grid_path):
    ref = master[run_names[0]][grid_path][:].astype(np.float64)
    for name in run_names[1:min(6, len(run_names))]:
        other = master[name][grid_path][:].astype(np.float64)
        
        # Only check X and Y coordinates ([:, :2]) for strict consistency.
        # The Z coordinate fluctuates slightly due to the conformal mapping 
        # to the interface and the ALE volume conservation constraint.
        if other.shape != ref.shape or not np.allclose(other[:, :2], ref[:, :2], atol=1e-5):
            return ref, False
    return ref, True


# ======================================================================
# Field extractors — return (P, U_raw, x_grid, u_ref, field_shape)
#
#   P:           [N, 24]       anode currents (raw amperes)
#   U_raw:       [N, M_field]  field values (possibly delta-subtracted)
#   x_grid:      [M_grid, d]   spatial coords for plotting
#   u_ref:       [M_field] or None
#   field_shape: tuple e.g. (M_nodes, 3) for vector fields
# ======================================================================

def extract_velocity_midacd(master, run_names, ref_name, do_delta, **kw):
    field_path = "fields_midacd/vitesse"
    grid_path  = "fields_midacd/nodes"

    ref_grid, ok = _check_consistent_grid(master, run_names, grid_path)
    if not ok:
        raise ValueError("Mid-ACD grid is inconsistent across runs.")
    x_grid = ref_grid[:, :2].astype(np.float32) if ref_grid.shape[1] == 3 \
        else ref_grid.astype(np.float32)

    M = x_grid.shape[0]
    N = len(run_names)
    field_shape = (M, 3)
    M_field = M * 3

    P = np.empty((N, N_ANODES), dtype=np.float32)
    U = np.empty((N, M_field),  dtype=np.float32)

    for i, name in enumerate(run_names):
        grp = master[name]
        P[i] = grp["input/currents"][:].astype(np.float32).ravel()[:N_ANODES]
        U[i] = grp[field_path][:].astype(np.float32).ravel()

    u_ref = None
    if do_delta:
        u_ref = master[ref_name][field_path][:].astype(np.float32).ravel()
        U -= u_ref[None, :]

    print(f"  velocity_midacd: P{P.shape}, U{U.shape}, grid{x_grid.shape}")
    return P, U, x_grid, u_ref, field_shape


def extract_velocity_full(master, run_names, ref_name, do_delta, **kw):
    """Extract the 3D velocity field on FLUID nodes only (bath + aluminium).
 
    The cuveb mesh contains the entire cell (solids + fluids). We use
    cuveb_refs and the material IDs ref_alu / ref_ele (stored by
    ml_export.cpp in /scalars) to restrict to fluid elements, then
    collect the unique node set.
 
    This avoids training on trivially-zero solid-domain DOFs and makes
    the POD significantly more efficient.
    """
    field_path = "fields_full/vitesse"
    grid_path  = "mesh/cuveb_nodes"
    elem_path  = "mesh/cuveb_elems"
    refs_path  = "mesh/cuveb_refs"
 
    # ── Identify fluid nodes from the first run ──
    grp0 = master[run_names[0]]
 
    # Material reference IDs
    scalars = grp0["scalars"]
    ref_ele = int(scalars.attrs["ref_ele"])   # bath / electrolyte
    ref_alu = int(scalars.attrs["ref_alu"])   # liquid aluminium
 
    elems = grp0[elem_path][:].astype(np.int32)   # (Ne, 4) 0-indexed
    refs  = grp0[refs_path][:].ravel()
 
    fluid_elem_mask = (refs == ref_ele) | (refs == ref_alu)
    fluid_node_ids  = np.unique(elems[fluid_elem_mask].ravel())  # sorted
    fluid_node_ids  = fluid_node_ids.astype(np.int32)
 
    M_total = grp0[grid_path].shape[0]
    M_fluid = len(fluid_node_ids)
    n_comp  = 3
    field_shape = (M_fluid, n_comp)
    M_field = M_fluid * n_comp
    N = len(run_names)
 
    print(f"  Fluid-node filter: {M_fluid} / {M_total} nodes "
          f"({100 * M_fluid / M_total:.1f}% of cuveb mesh)")
    print(f"  Material refs: bath={ref_ele}, aluminium={ref_alu}")
    print(f"  Loading {N} fluid-only snapshots ({M_field} DOFs)…")
 
    # Reference grid: fluid nodes only
    x_grid_ref = grp0[grid_path][:].astype(np.float32)[fluid_node_ids]
 
    P = np.empty((N, N_ANODES), dtype=np.float32)
    U = np.empty((N, M_field),  dtype=np.float32)
 
    for i, name in enumerate(run_names):
        grp = master[name]
        P[i] = grp["input/currents"][:].astype(np.float32).ravel()[:N_ANODES]
        vel  = grp[field_path][()].astype(np.float32)   # (M_total, 3)
        U[i] = vel[fluid_node_ids].ravel()               # (M_fluid * 3,)
 
    u_ref = None
    if do_delta:
        vel_ref = master[ref_name][field_path][()].astype(np.float32)
        u_ref = vel_ref[fluid_node_ids].ravel()
        U -= u_ref[None, :]
 
    print(f"  velocity_full (fluid only): P{P.shape}, U{U.shape}, "
          f"grid{x_grid_ref.shape}")
 
    # Pack fluid_node_ids into kw so write_h5 can store it.
    # We piggyback on the **kw mechanism or return it via a side channel.
    # Cleanest: store in a module-level variable that write_h5 reads.
    extract_velocity_full._fluid_node_ids = fluid_node_ids
    extract_velocity_full._M_total = M_total
 
    # Also store the fluid element connectivity for proper mesh plotting
    # Remap element node IDs: old_global → new_local
    global_to_local = np.full(M_total, -1, dtype=np.int32)
    global_to_local[fluid_node_ids] = np.arange(M_fluid, dtype=np.int32)
    fluid_elems_global = elems[fluid_elem_mask]          # (Ne_fluid, 4)
    fluid_elems_local  = global_to_local[fluid_elems_global]
    fluid_refs = refs[fluid_elem_mask]
 
    extract_velocity_full._fluid_elems = fluid_elems_local
    extract_velocity_full._fluid_refs  = fluid_refs
 
    return P, U, x_grid_ref, u_ref, field_shape


def extract_interface(master, run_names, ref_name, do_delta,
                      grid_res=64, **kw):
    # We no longer need field_path = "fields_interface/h"
    int_node_path = "mesh/interface_nodes"

    # Bounding box
    x_all, y_all = [], []
    for name in run_names:
        if int_node_path not in master[name]:
            continue
        nodes = master[name][int_node_path][:].astype(np.float64)
        x_all.append(nodes[:, 0])
        y_all.append(nodes[:, 1])
    x_all, y_all = np.concatenate(x_all), np.concatenate(y_all)

    dx = (x_all.max() - x_all.min()) * 0.02
    dy = (y_all.max() - y_all.min()) * 0.02
    x_lin = np.linspace(x_all.min() + dx, x_all.max() - dx, grid_res)
    y_lin = np.linspace(y_all.min() + dy, y_all.max() - dy, grid_res)
    xg, yg = np.meshgrid(x_lin, y_lin, indexing='ij')
    xy_target = np.stack([xg.ravel(), yg.ravel()], axis=1)
    M = xy_target.shape[0]
    x_grid = xy_target.astype(np.float32)
    field_shape = (M, 1)

    def _interp(nodes_2d, target_vals, xy):
        val = griddata(nodes_2d, target_vals, xy, method='linear')
        nans = np.isnan(val)
        if nans.any():
            val[nans] = griddata(nodes_2d, target_vals, xy[nans], method='nearest')
        return val

    N = len(run_names)
    P = np.empty((N, N_ANODES), dtype=np.float32)
    U = np.empty((N, M),        dtype=np.float32)
    valid = 0
    for name in run_names:
        grp = master[name]
        if int_node_path not in grp:
            continue
        nodes  = grp[int_node_path][:].astype(np.float64)
        
        # THE FIX: Extract the absolute Z coordinate of the interface mesh
        z_vals = nodes[:, 2]
        
        P[valid] = grp["input/currents"][:].astype(np.float32).ravel()[:N_ANODES]
        U[valid] = _interp(nodes[:, :2], z_vals, xy_target).astype(np.float32)
        valid += 1
    P, U = P[:valid], U[:valid]

    u_ref = None
    if do_delta:
        rg = master[ref_name]
        ref_nodes = rg[int_node_path][:].astype(np.float64)
        
        # Extract the absolute Z coordinate of the REFERENCE run
        ref_z = ref_nodes[:, 2]
        u_ref = _interp(ref_nodes[:, :2], ref_z, xy_target).astype(np.float32)
        
        # U -= u_ref now naturally computes the total cumulative physical \Delta Z
        U -= u_ref[None, :]

    print(f"  interface: P{P.shape}, U{U.shape}, grid{x_grid.shape}")
    return P, U, x_grid, u_ref, field_shape


EXTRACTORS = {
    "velocity_midacd": extract_velocity_midacd,
    "velocity_full":   extract_velocity_full,
    "interface":       extract_interface,
}


# ======================================================================
# Offline POD (optional reduction step)
# ======================================================================

def apply_pod(U_raw, n_modes):
    """Coupled POD on snapshot matrix U_raw [N, M_field].

    For vector fields (M_field = M_nodes × 3), the SVD operates on the
    concatenated vector.  Each mode φ_k ∈ R^{M_field} therefore captures
    correlated patterns across all three velocity components — this is the
    standard "vector POD" used in fluid mechanics.

    Returns:
        C:          [N, k]         coefficient matrix (training target)
        pod_info:   dict with V [M_field, k], u_pod_mean [M_field],
                    evr [k], cumevr [k]
    """
    N, M = U_raw.shape
    k = min(n_modes, N - 1, M)

    u_mean = U_raw.mean(axis=0)
    Uc = U_raw - u_mean

    print(f"  Computing coupled POD ({k} modes on {M}-dim field)…")

    # Gram-matrix trick: N×N eigendecomposition (O(N²M) vs O(NM²))
    G = (Uc @ Uc.T) / max(N - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(G.astype(np.float64))
    idx = np.argsort(eigvals)[::-1][:k]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx].astype(np.float32)

    # Recover right singular vectors (POD modes)
    V = Uc.T @ eigvecs                           # [M, k]
    norms = np.linalg.norm(V, axis=0, keepdims=True).clip(min=1e-12)
    V = V / norms                                 # orthonormal

    # Project snapshots → coefficients
    C = Uc @ V                                    # [N, k]

    evr = eigvals / max(eigvals.sum(), 1e-30)
    cumevr = np.cumsum(evr)
    eff_rank = 1.0 / max(np.sum((evr / evr.sum()) ** 2), 1e-30)

    def _n_for(thr):
        i = np.searchsorted(cumevr, thr)
        return int(i) + 1 if i < len(cumevr) else len(cumevr)

    print(f"    Eff. rank = {eff_rank:.1f}")
    print(f"    90% → {_n_for(0.90)} modes,  "
          f"95% → {_n_for(0.95)},  99% → {_n_for(0.99)}")

    pod_info = {
        "V":          V.astype(np.float32),
        "u_pod_mean": u_mean.astype(np.float32),
        "evr":        evr.astype(np.float32),
        "cumevr":     cumevr.astype(np.float32),
    }
    return C.astype(np.float32), pod_info


# ======================================================================
# Input normalization
# ======================================================================

def normalize_currents(P, method="deviation"):
    info = {"method": method}
    if method == "none":
        info["p_mean"] = np.zeros(N_ANODES, dtype=np.float32)
        info["p_std"]  = np.ones(N_ANODES,  dtype=np.float32)
        return P.copy(), info
    if method == "deviation":
        I_per_sample = P.mean(axis=1, keepdims=True)
        P_norm = ((P - I_per_sample) / I_per_sample).astype(np.float32)
        info["p_mean"] = np.zeros(N_ANODES, dtype=np.float32)
        info["p_std"]  = np.ones(N_ANODES,  dtype=np.float32)
        info["I_mean_global"] = float(P.mean())
        return P_norm, info
    p_mean = P.mean(axis=0).astype(np.float32)
    p_std  = (P.std(axis=0) + 1e-9).astype(np.float32)
    P_norm = ((P - p_mean) / p_std).astype(np.float32)
    info["p_mean"], info["p_std"] = p_mean, p_std
    return P_norm, info


# ======================================================================
# Stratified splitting
# ======================================================================

def _load_manifest_modes(path):
    f2m = {}
    with open(path, "r") as f:
        for row in _csv.DictReader(f):
            f2m[row["folder"]] = row["mode"]
    return f2m


def _infer_mode(currents, dead_thresh=500.0):
    mu = currents.mean()
    N = len(currents)
    if np.any(currents < dead_thresh):
        return "dead"
    I_MIN, I_MAX = 16_400.0, 24_400.0
    at_lo = np.sum(currents < I_MIN + 100)
    at_hi = np.sum(currents > I_MAX - 100)
    if (at_lo == 1 and at_hi == 0) or (at_hi == 1 and at_lo == 0):
        return "single"
    x = np.arange(N, dtype=np.float64)
    slope, _ = np.polyfit(x, currents, 1)
    pred = slope * x + (mu - slope * (N - 1) / 2)
    ss_res = np.sum((currents - pred) ** 2)
    ss_tot = np.sum((currents - mu) ** 2)
    if ss_tot > 1e-10 and (1 - ss_res / ss_tot) > 0.7 and abs(slope) > 100:
        return "gradient"
    below = currents < (mu - 1500)
    for i in range(N - 1):
        if below[i] and below[i + 1]:
            return "cluster"
    if np.any((currents > dead_thresh) & (currents < 0.5 * mu)):
        return "weak"
    return "gaussian"


def stratified_split(N, run_names, ratios, manifest_path=None,
                     P=None, dead_thresh=500.0, seed=42):
    rng = np.random.default_rng(seed)
    if manifest_path:
        f2m = _load_manifest_modes(manifest_path)
        modes = [f2m.get(n, "unknown") for n in run_names]
    elif P is not None:
        modes = [_infer_mode(P[i], dead_thresh) for i in range(N)]
    else:
        modes = ["unknown"] * N

    groups: dict[str, list[int]] = {}
    for i, m in enumerate(modes):
        groups.setdefault(m, []).append(i)

    train_all, val_all, test_all = [], [], []
    print("\n  Stratified split:")
    for mode in sorted(groups):
        idx = np.array(groups[mode])
        n = len(idx)
        rng.shuffle(idx)
        if n < 3:
            train_all.extend(idx)
            print(f"    {mode:12s}: {n:4d} → all train")
            continue
        nt = max(1, int(n * ratios[0]))
        nv = max(1, int(n * ratios[1]))
        ne = max(1, n - nt - nv)
        nt = n - nv - ne
        train_all.extend(idx[:nt])
        val_all.extend(idx[nt:nt + nv])
        test_all.extend(idx[nt + nv:])
        print(f"    {mode:12s}: {n:4d} → train={nt}, val={nv}, test={ne}")

    return np.sort(train_all), np.sort(val_all), np.sort(test_all), modes


def _write_fluid_mesh_info(recon_group):
    """Call this inside write_h5 after creating the reconstruction/ group.
 
    Stores the fluid-node index array and mesh connectivity so that
    evaluation/plotting can map predicted DOFs back to the full mesh.
    """
    fluid_node_ids = getattr(extract_velocity_full, '_fluid_node_ids', None)
    M_total        = getattr(extract_velocity_full, '_M_total', None)
    fluid_elems    = getattr(extract_velocity_full, '_fluid_elems', None)
    fluid_refs     = getattr(extract_velocity_full, '_fluid_refs', None)
 
    if fluid_node_ids is not None:
        recon_group.create_dataset("fluid_node_ids", data=fluid_node_ids)
        recon_group.attrs["M_total"] = M_total
        print(f"  Stored fluid_node_ids ({len(fluid_node_ids)}) and "
              f"M_total={M_total}")
 
    if fluid_elems is not None:
        recon_group.create_dataset("fluid_elems", data=fluid_elems,
                                   compression="gzip")
        recon_group.create_dataset("fluid_refs", data=fluid_refs)
        print(f"  Stored fluid mesh: {fluid_elems.shape[0]} tets")


# ======================================================================
# HDF5 writer
# ======================================================================

def write_h5(path, P, U, x_grid, train_idx, val_idx, test_idx,
             norm_info, mapping, modes, u_ref, field_shape,
             delta_learning, pod_reduction, reference_run,
             M_field, pod_info=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    U_train = U[train_idx]
    u_mean  = U_train.mean(axis=0).astype(np.float32)
    u_std   = (U_train.std(axis=0) + 1e-9).astype(np.float32)

    with h5py.File(path, "w") as f:
        m = f.create_group("meta")
        m.attrs["mapping"]        = mapping
        m.attrs["delta_learning"] = bool(delta_learning)
        m.attrs["pod_reduction"]  = bool(pod_reduction)
        m.attrs["reference_run"]  = reference_run or ""
        m.attrs["n_params"]       = int(P.shape[1])
        m.attrs["M_out"]          = int(U.shape[1])
        m.attrs["M_field"]        = int(M_field)
        m.attrs["field_shape"]    = list(field_shape)
        m.attrs["input_norm"]     = norm_info["method"]

        f.create_dataset("x_grid", data=x_grid)

        s = f.create_group("stats")
        s.create_dataset("p_mean", data=norm_info["p_mean"])
        s.create_dataset("p_std",  data=norm_info["p_std"])
        s.create_dataset("u_mean", data=u_mean)
        s.create_dataset("u_std",  data=u_std)

        r = f.create_group("reconstruction")
        _write_fluid_mesh_info(r)
        if u_ref is not None:
            r.create_dataset("u_ref", data=u_ref, compression="gzip")
        if pod_info is not None:
            r.create_dataset("V",          data=pod_info["V"],
                             compression="gzip")
            r.create_dataset("u_pod_mean", data=pod_info["u_pod_mean"])
            r.create_dataset("evr",        data=pod_info["evr"])
            r.create_dataset("cumevr",     data=pod_info["cumevr"])

        dt = h5py.string_dtype()
        for name, idx in [("train", train_idx), ("val", val_idx),
                          ("test", test_idx)]:
            g = f.create_group(name)
            g.create_dataset("P", data=P[idx], compression="gzip")
            g.create_dataset("U", data=U[idx], compression="gzip")
            g.create_dataset("modes",
                             data=[modes[i] for i in idx], dtype=dt)
            print(f"  {name}: {len(idx)} samples")

    print(f"\nSaved → {path}")
    print(f"  mapping: {mapping},  delta: {delta_learning},  pod: {pod_reduction}")
    print(f"  P: [{P.shape[1]}],  U(target): [{U.shape[1]}],  "
          f"M_field: {M_field},  grid: {x_grid.shape}")


# ======================================================================
# CLI
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Prepare Alucell data for ML.")
    ap.add_argument("--master",        required=True)
    ap.add_argument("--manifest",      default=None)
    ap.add_argument("--mapping",       required=True,
                    choices=list(EXTRACTORS.keys()))
    ap.add_argument("--output",        required=True)

    # Orthogonal flags
    ap.add_argument("--delta",         action="store_true",
                    help="Subtract reference solution (delta-learning)")
    ap.add_argument("--pod",           action="store_true",
                    help="Apply offline POD — store coefficients as target. "
                         "Use ResFFNN at training time (not POD_MLP).")
    ap.add_argument("--reference-run", default=None)

    # POD options
    ap.add_argument("--n-modes",       type=int, default=50,
                    help="Number of POD modes if --pod is set (default: 50)")

    # Filtering
    ap.add_argument("--dead-threshold", type=float, default=500.0)
    ap.add_argument("--include-dead",  action="store_true")

    # Split & normalization
    ap.add_argument("--split",         nargs=3, type=float,
                    default=[0.75, 0.15, 0.10])
    ap.add_argument("--input-norm",    default="deviation",
                    choices=["standard", "deviation", "none"])
    ap.add_argument("--interface-res", type=int, default=64)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    ratios = tuple(args.split)
    assert abs(sum(ratios) - 1.0) < 1e-6

    print(f"Master    : {args.master}")
    print(f"Mapping   : {args.mapping}")
    print(f"Delta     : {args.delta}")
    print(f"POD       : {args.pod}" +
          (f" ({args.n_modes} modes)" if args.pod else ""))
    print(f"Ref run   : {args.reference_run or '(auto)'}")
    print(f"Split     : {ratios}\n")

    master    = h5py.File(args.master, "r")
    run_names = _get_run_names(master, args.dead_threshold,
                               exclude_dead=not args.include_dead)
    print(f"Runs: {len(run_names)}")

    ref_name = _find_reference_run(master, run_names, args.reference_run)

    # ── Extract raw field ──
    extractor = EXTRACTORS[args.mapping]
    P, U_raw, x_grid, u_ref, field_shape = extractor(
        master, run_names, ref_name, args.delta,
        grid_res=args.interface_res)
    master.close()

    M_field = U_raw.shape[1]

    # ── Optional POD reduction ──
    pod_info = None
    if args.pod:
        U_target, pod_info = apply_pod(U_raw, args.n_modes)
    else:
        U_target = U_raw
        if M_field > 50_000:
            print(f"\n  [NOTE] M_field = {M_field:,} is large. Without --pod,")
            print(f"  training will load full fields per batch. Consider using")
            print(f"  --pod for faster training, or POD_MLP which does POD")
            print(f"  internally (but still loads full fields).\n")

    # ── Normalize inputs ──
    P_norm, norm_info = normalize_currents(P, method=args.input_norm)

    # ── Assign campaign modes ──
    if args.manifest:
        f2m   = _load_manifest_modes(args.manifest)
        modes = [f2m.get(n, "unknown") for n in run_names[:len(P)]]
    else:
        modes = [_infer_mode(P[i], args.dead_threshold) for i in range(len(P))]

    # ── Stratified split ──
    train_idx, val_idx, test_idx, modes = stratified_split(
        len(P_norm), run_names[:len(P_norm)], ratios,
        manifest_path=args.manifest, P=P,
        dead_thresh=args.dead_threshold, seed=args.seed)

    # ── Write ──
    write_h5(args.output, P_norm, U_target, x_grid,
             train_idx, val_idx, test_idx,
             norm_info, args.mapping, modes,
             u_ref, field_shape, args.delta, args.pod, ref_name,
             M_field, pod_info)

    # ── Sanity ──
    print(f"\n--- Sanity ---")
    print(f"  P range: [{P_norm.min():.3f}, {P_norm.max():.3f}]")
    print(f"  U range: [{U_target.min():.6f}, {U_target.max():.6f}]")
    print(f"  U train std: {U_target[train_idx].std():.6f}")


if __name__ == "__main__":
    main()