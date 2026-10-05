"""
aggregate_coverage.py — Build the master dataset and analyse campaign coverage.

Usage:
    python aggregate_coverage.py --dir runs/ --dead-threshold 500
    python aggregate_coverage.py --dir runs/ --skip-phase1     # analysis only

Phase 1 (WRITE):  Validate → Merge → Postprocess → master_dataset.h5
Phase 2 (READ):   Input coverage · Field POD · Predictability · EM forcing ·
                  Flow topology · Convergence quality

Every Phase 2 statistic is introduced by a comment stating the one question it
answers and, where relevant, the question it does NOT answer.  The header of
the Phase 2 section states the four distinctions needed to read the numbers
without apparent contradictions (design vs response, energy vs truncation,
marginal vs multivariate, magnitude vs shape).

Outputs:
    master_dataset.h5          per-run groups with postprocessed fields
    scalars.parquet/.csv       one row per run, all scalar diagnostics
    coverage_report.txt        annotated text summary
    coverage_full.png          six panels, one per question
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# Rejection tolerance on the pointwise kinematic residual max|u.n| (m/s).
# u.n = 0 is imposed exactly through a Lagrange multiplier, so what is left is
# the pointwise residual of a weakly-imposed constraint.  Shared with the
# Phase 2 report so the two can never drift apart.
KINEMATIC_RESIDUAL_TOL = 1e-2


# ======================================================================
# PHASE 1: VALIDATION, AGGREGATION, POSTPROCESSING
# ======================================================================

# ── Validation ────────────────────────────────────────────────────────

def validate_run(folder: Path) -> tuple[bool, str]:
    """Check solver convergence AND physical plausibility.

    Level 1: Alucell STATUS + CSV residuals (numerical convergence)
    Level 2: Physical sanity checks on run.h5 (inverted elements,
             unphysical velocity, extreme interface, max-iter detection)
    """
    # ── Level 1: STATUS file ──
    status_file = folder / "STATUS"
    if not status_file.exists():
        return False, "Missing STATUS"
    if "SUCCESS" not in status_file.read_text().upper():
        return False, "STATUS != SUCCESS"

    control_dir = folder / "control"
    if not control_dir.exists():
        return False, "Missing control folder"

    try:
        # Velocity residual — must be below threshold AND not increasing
        df_vel = pd.read_csv(control_dir / "CONTROL_VELOCITY.csv",
                             skipinitialspace=True)
        col_vel = [c for c in df_vel.columns if "Velocity-rel-diff-L2" in c]
        if not col_vel:
            return False, "Velocity CSV missing expected column"
        final_vel = df_vel[col_vel[0]].iloc[-1]
        if final_vel > 1e-3:
            return False, f"Velocity non-converged (res={final_vel:.1e})"
        # Monotonicity: residual should not grow over last 5 steps
        if len(df_vel) >= 5 and final_vel > 5e-4:
            last5 = df_vel[col_vel[0]].iloc[-5:].values
            if last5[-1] > last5[0] * 1.1:
                return False, f"Velocity residual increasing ({last5[0]:.1e}→{last5[-1]:.1e})"

        # Interface residual
        df_h = pd.read_csv(control_dir / "CONTROL_INTERFACE_H.csv",
                           skipinitialspace=True)
        col_h = [c for c in df_h.columns if "Height-rel-diff-L2" in c]
        if not col_h:
            return False, "Interface CSV missing expected column"
        final_h = df_h[col_h[0]].iloc[-1]
        if final_h > 1e-3:
            return False, f"Interface non-converged (res={final_h:.1e})"

        # Control point stability — tightened to 0.1 mm
        df_def = pd.read_csv(control_dir / "CONTROL_INTERFACE_DEFORMATION.csv",
                             skipinitialspace=True)
        p_cols = [f'P{i:02d}_h (mm)' for i in range(1, 11)]
        found = [c for c in p_cols if c in df_def.columns]
        if len(found) < 5:
            return False, f"Control points missing ({len(found)}/10 found)"
        max_drift = df_def[found].tail(2).diff().abs().iloc[-1].max()
        if max_drift > 0.1:
            return False, f"Control points drifting ({max_drift:.2e} mm)"

    except Exception as e:
        return False, f"CSV error: {e}"

    # ── Level 2: Physical sanity checks on run.h5 ──
    h5_path = folder / "run.h5"
    if not h5_path.exists():
        return False, "Missing run.h5"

    try:
        with h5py.File(h5_path, "r") as f:
            # Currents must exist
            if "input/currents" not in f:
                return False, "run.h5 missing /input/currents"

            if "scalars" in f:
                s = f["scalars"]

                # NOTE: to detect max-iteration runs, store max_iter as an
                # HDF5 attribute in ml_export.cpp and check statk == max_iter.
                # A hardcoded list of common values is too fragile.
                # The CSV residual checks below already catch non-convergence.

                # Kinematic residual: u·n should be near zero at steady state
                if "kinematic_residual_max" in s.attrs:
                    kr = float(s.attrs["kinematic_residual_max"])
                    if kr > KINEMATIC_RESIDUAL_TOL:
                        return False, f"Interface not stationary (kin_res={kr:.1e})"

                # 2. NEW: Mass conservation / Well-posedness check
                # For an incompressible flow in a closed box, net flux across a plane must be 0.
                planes_to_check = [
                    "mid_acd", "plane_X", "plane_minus_X", "plane_Y", "plane_minus_Y"
                ]
                for plane in planes_to_check:
                    net_key = f"net_flux_{plane}"
                    abs_key = f"half_abs_flux_{plane}"
                    
                    if net_key in s.attrs and abs_key in s.attrs:
                        net_f = float(s.attrs[net_key])
                        abs_f = float(s.attrs[abs_key])
                        
                        # Only check if there is actual flow happening
                        if abs_f > 1e-8:
                            # Normalized leakage: Net Mass Flux / Circulating Mass Flux
                            leakage_ratio = abs(net_f) / abs_f
                            # Threshold: If > 5% of the flow is "disappearing", reject the run
                            if leakage_ratio > 0.1:
                                return False, f"Mass leak on {plane} (leakage={leakage_ratio*100:.1f}%)"

                # Inverted elements from ALE mesh deformation
                if "mesh_min_jacobian" in s.attrs:
                    mj = float(s.attrs["mesh_min_jacobian"])
                    if mj <= 0:
                        return False, f"Inverted element (min_jac={mj:.1e})"

            # Velocity: NaN/Inf and unphysical magnitudes
            if "fields_full/vitesse" in f:
                v = f["fields_full/vitesse"][()]
                if not np.all(np.isfinite(v)):
                    return False, "NaN/Inf in velocity"
                v_max = float(np.linalg.norm(v, axis=1).max())
                if v_max > 10.0:
                    return False, f"Unphysical velocity ({v_max:.1f} m/s)"

            # Interface deformation check.
            # IMPORTANT: use fields_interface/h (the displacement from the
            # reference plane), NOT mesh/interface_nodes z-coordinates.
            # The absolute z includes the base geometry height which can
            # easily exceed 100 mm even for a perfectly valid simulation.
            if "fields_interface/h" in f:
                h_def = f["fields_interface/h"][()].ravel()
                if h_def.size > 0:
                    h_ptp = float(np.max(np.abs(h_def)))
                    # Max displacement > 150 mm means the interface likely
                    # touches an anode (ACD is typically ~50 mm)
                    if h_ptp > 0.15:
                        return False, f"Extreme deformation ({h_ptp*1000:.0f} mm)"

    except Exception as e:
        return False, f"HDF5 check error: {e}"

    return True, "Valid"

# ── Fiber projection ─────────────────────────────────────────────────

def _project_fiber_1d(x, z, o, oz, zdn, zup, z_target):
    """Project a set of (x, z) points along diverging fibers to z_target."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    both_above = (z >= zup) & (z_target >= zup)
    z_ini = np.where(z > zup, zup, z)
    z_fin = min(z_target, zup)
    active = (z_ini >= zdn) & ~both_above
    denom = z_ini - oz
    safe = np.abs(denom) > 1e-14
    ratio = np.where(safe, (z_fin - oz) / np.where(safe, denom, 1.0), 1.0)
    x_out = x.copy()
    x_out = np.where(active & (x > o), o + (x - o) * ratio, x_out)
    x_out = np.where(active & (x < -o), -o + (x + o) * ratio, x_out)
    return x_out


def _fiber_vertical_average(bath_nodes, bath_vel, midacd_xy,
                            ox, oy, oz, zdn, zup, z_target):
    """Average velocity along vertical fibers, then interpolate to mid-ACD grid."""
    x, y, z = bath_nodes[:, 0], bath_nodes[:, 1], bath_nodes[:, 2]
    x_proj = _project_fiber_1d(x, z, ox, oz, zdn, zup, z_target)
    y_proj = _project_fiber_1d(y, z, oy, oz, zdn, zup, z_target)

    # Group nodes by projected (x,y) position = one fiber each
    key = np.stack([np.round(x_proj, 6), np.round(y_proj, 6)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)

    v_avg = np.zeros((len(uniq), 3), dtype=np.float64)
    for fid in range(len(uniq)):
        mask = inv == fid
        z_f = z[mask]
        v_f = bath_vel[mask]
        order = np.argsort(z_f)
        z_f, v_f = z_f[order], v_f[order]
        if len(z_f) == 1:
            v_avg[fid] = v_f[0]
            continue
        dz = np.diff(z_f)
        h = dz.sum()
        if h > 1e-14:
            v_avg[fid] = (0.5 * (v_f[:-1] + v_f[1:]) * dz[:, None]).sum(0) / h
        else:
            v_avg[fid] = v_f.mean(axis=0)

    # Interpolate fiber averages onto the structured mid-ACD grid
    v_midacd = np.zeros((len(midacd_xy), 3), dtype=np.float32)
    for k in range(3):
        vk = griddata(uniq, v_avg[:, k], midacd_xy, method='linear')
        vk_nn = griddata(uniq, v_avg[:, k], midacd_xy, method='nearest')
        v_midacd[:, k] = np.where(np.isnan(vk), vk_nn, vk)

    return v_midacd, uniq.astype(np.float32), v_avg.astype(np.float32)


# ── Per-run postprocessing ────────────────────────────────────────────

def _to_element_space(field: np.ndarray, elems: np.ndarray) -> np.ndarray | None:
    """Return *field* as one value per element.

    The exported fields do NOT all live in the same space.  Verified against
    the Alucell macros (alu-data/macros), where a table is created on
    cuveb_nodes for a P1 field and on cuveb_baryc for a P0 one:

        vitesse    P1  (Nn, 3)   eval( cuveb_nodes ; cuveb_vitesse )
        potentiel  P1  (Nn,)     the finite-element unknown
        induction  P1  (Nn, 3)   eval( cuveb_nodes ; cuveb_induction )
        courant    P0  (Ne, 3)   eval( cuveb_baryc ; cuveb_courant )
                                 j = sigma*grad(V) with V in P1
        forces     P0  (Ne, 3)   cforctet, per tetrahedron
        cuveb_refs P0  (Ne,)     material tag per element

    The macros name the native field plainly and give the PROJECTION a
    suffix — cuveb_courantN is the nodal projection of the P0 current,
    cuveb_induction_E the element projection of the P1 induction — so the
    unsuffixed names exported by ml_export.cpp are the ones listed above.
    (Note the comment block at the top of ml_export.cpp lists every field as
    (Nn, ...), which is wrong for courant and forces.)

    A P0 field is returned untouched; a P1 field is averaged over the four
    vertices of each tetrahedron.  For a P1 field on a simplex that average
    IS the value at the barycentre, so the projection is exact rather than
    approximate, and it is the same operation as Alucell's own `baryc`.

    Returns None when the field matches neither count, so callers can skip a
    statistic rather than compute a wrong one.
    """
    if elems is None:
        return None
    n_elems = elems.shape[0]
    n_nodes = int(elems.max()) + 1

    if field.shape[0] == n_elems:
        return field                            # already P0
    if field.shape[0] >= n_nodes:
        return field[elems].mean(axis=1)        # nodal -> element average
    return None


def _material_values(field: np.ndarray, elems: np.ndarray,
                     refs: np.ndarray, ref_id: int):
    """Values of *field* restricted to the material tagged *ref_id*.

    cuveb_refs tags ELEMENTS, so a P0 field (courant, forces) can be masked
    with it directly, while a P1 field (vitesse, induction) must be projected
    onto the elements first.  Which one applies is decided from the array
    lengths rather than assumed, so the same call site works for both — and
    would still work if a future export changed a field's space.

    The field-length test comes FIRST: for a P0 field, refs, elems and the
    field all have length Ne, and averaging over elems in that case would
    index a per-element array with node numbers and silently return garbage.

    Returns None when no convention matches, so callers can skip the
    statistic instead of crashing the whole run.
    """
    if field.shape[0] == refs.shape[0]:
        values = field                          # same space as the tags
    elif elems is not None and refs.shape[0] == elems.shape[0]:
        values = _to_element_space(field, elems)
        if values is None:
            return None
    else:
        return None

    mask = (refs == ref_id)
    return values[mask] if mask.any() else None


def _safe_stat(arr, fn, default=np.nan):
    if arr.size == 0:
        return default
    try:
        return float(fn(arr))
    except Exception:
        return default


def postprocess_run(grp: h5py.Group) -> dict:
    """Compute derived fields and metrics for one run group inside master.h5.

    Material reference IDs (ref_alu, ref_ele) are read from the /scalars
    group, where ml_export.cpp writes them from ns3d_matalu and ns3d_matele.
    If they are missing, the run cannot be postprocessed.

    Modifies the HDF5 group in-place (adds fields_midacd/vitesse and
    derived_metrics). Returns a dict of scalar metrics.
    """
    metrics = {}

    # ── Read mesh and fields ──
    nodes = grp['mesh/cuveb_nodes'][:].astype(np.float64)
    elems = grp['mesh/cuveb_elems'][:]               # (Ne, 4), 0-indexed
    refs = grp['mesh/cuveb_refs'][:].ravel()          # .ravel() for (Ne,1)→(Ne,)
    velocity = grp['fields_full/vitesse'][:].astype(np.float64)

    scalars = grp.get('scalars')
    if scalars is None:
        raise ValueError("No /scalars group — ml_export may not have completed")

    def get_attr(name, required=False):
        if name not in scalars.attrs:
            if required:
                raise ValueError(f"Required scalar attr '{name}' missing from /scalars")
            return None
        return float(scalars.attrs[name])

    # ── Material refs from HDF5 (written by ml_export.cpp) ──
    ref_ele = int(get_attr('ref_ele', required=True))  # bath/electrolyte
    ref_alu = int(get_attr('ref_alu', required=True))  # metal/aluminium

    # ── Global velocity stats ──
    vmag = np.linalg.norm(velocity, axis=1)
    metrics['u_max_full'] = _safe_stat(vmag, np.max)
    metrics['u_mean_full'] = _safe_stat(vmag, np.mean)
    metrics['u_rms_full'] = _safe_stat(vmag, lambda v: np.sqrt(np.mean(v**2)))

    # Velocity magnitude restricted to each material.
    for label, ref_id in [("alu", ref_alu), ("bath", ref_ele)]:
        vals = _material_values(vmag, elems, refs, ref_id)
        if vals is not None and vals.size:
            metrics[f'u_max_{label}'] = float(vals.max())
            metrics[f'u_mean_{label}'] = float(vals.mean())

    # ── Interface stats ──
    if "mesh/interface_nodes" in grp:
        z_int = grp["mesh/interface_nodes"][:, 2]
        metrics["eta_max"] = _safe_stat(z_int, np.max)
        metrics["eta_min"] = _safe_stat(z_int, np.min)
        metrics["eta_ptp"] = float(z_int.max() - z_int.min()) if z_int.size else np.nan
        metrics["eta_std"] = _safe_stat(z_int, np.std)

    # ── Horizontal current fraction (metal only — MHD instability proxy) ──
    if "fields_full/courant" in grp:
        j_all = grp["fields_full/courant"][:].astype(np.float32)
        j_met = _material_values(j_all, elems, refs, ref_alu)
        if j_met is not None and j_met.size:
            j_horiz = np.linalg.norm(j_met[:, :2], axis=1)
            jz = np.abs(j_met[:, 2]).sum()
            metrics['j_horiz_frac_alu'] = float(j_horiz.sum() / max(jz, 1e-30))
            metrics['j_horiz_max_alu'] = float(j_horiz.max())

    # ── Body force stats (ρg + J×B, NOT just Lorentz) ──
    if "fields_full/forces" in grp:
        F = grp["fields_full/forces"][:].astype(np.float32)
        Fmag = np.linalg.norm(F, axis=1)
        metrics['f_body_max'] = _safe_stat(Fmag, np.max)
        metrics['f_body_mean'] = _safe_stat(Fmag, np.mean)

    # ── Fiber-averaged velocity on mid-ACD plane ──
    ox = get_attr('fiber_ox')
    oy = get_attr('fiber_oy')
    oz = get_attr('fiber_oz')  # Note: ml_export writes this as geom_z0
    zdn = get_attr('fiber_zdn')
    zup = get_attr('fiber_zup')

    if None not in (ox, oy, oz, zdn, zup) and 'fields_midacd/nodes' in grp:
        midacd_nodes = grp['fields_midacd/nodes'][:].astype(np.float64)
        bath_mask = (refs == ref_ele)
        bath_node_ids = np.unique(elems[bath_mask].ravel())

        z_midacd = float(midacd_nodes[:, 2].mean())
        v_midacd, f_xy, f_v = _fiber_vertical_average(
            nodes[bath_node_ids], velocity[bath_node_ids],
            midacd_nodes[:, :2], ox, oy, oz, zdn, zup, z_midacd)

        # Write fiber-averaged velocity into the HDF5
        midacd_grp = grp['fields_midacd']
        if 'vitesse' in midacd_grp:
            del midacd_grp['vitesse']
        midacd_grp.create_dataset('vitesse', data=v_midacd.astype(np.float32))

        vmag_mid = np.linalg.norm(v_midacd, axis=1)
        metrics["u_max_midacd"] = _safe_stat(vmag_mid, np.max)
        metrics["u_mean_midacd"] = _safe_stat(vmag_mid, np.mean)
        metrics["u_rms_midacd"] = _safe_stat(vmag_mid, lambda v: np.sqrt(np.mean(v**2)))

        # Store fiber data for debugging
        pp = grp.require_group('postprocess')
        if 'fiber_avg' in pp:
            del pp['fiber_avg']
        fa = pp.create_group('fiber_avg')
        fa.create_dataset('xy', data=f_xy)
        fa.create_dataset('v', data=f_v)

    # ── Collect all C++ scalar attrs ──
    if scalars:
        for k, v in scalars.attrs.items():
            try:
                metrics[f"attr_{k}"] = float(np.asarray(v).item())
            except Exception:
                pass

    # ── Save Python-derived metrics to HDF5 ──
    dm = grp.require_group('derived_metrics')
    for k, v in metrics.items():
        if not k.startswith("attr_"):
            dm.attrs[k] = v

    return metrics


# ── Phase 1 main ──────────────────────────────────────────────────────

# -- Master layout: shared connectivity + compression ------------------------
# Every run in a campaign is solved on the same mesh, deformed by ALE.  The
# node COORDINATES therefore differ run to run, but the connectivity and the
# element/node reference tags are byte-identical in all of them (verified on
# stat_0002/0500/1000/1400).  Copying them per run costs 46 MB x N -- 74 GB on
# a 1600-run campaign, about 40% of the master, for one mesh repeated.
#
# They are written once and hard-linked into every other run, so every existing
# read path still resolves unchanged -- master[run]["mesh/cuveb_elems"] returns
# the same array -- while the bytes are stored once.  Bulk arrays additionally
# get gzip-1 + shuffle: roughly 2x on float32 fields for a few percent of the
# write time, and gzip ships with the base HDF5 library, so h5dump and ParaView
# still read the result.
SHARED_DATASETS = (
    "mesh/cuveb_elems", "mesh/cuveb_refs",
    "mesh/fluid_bord_elems", "mesh/fluid_bord_refs", "mesh/fluid_bord_nodes",
    "mesh/interface_elems", "mesh/interface_refs_cuveb",
    "fields_midacd/elems",
)


def _copy_run(src, master, name, link_src):
    """Copy one run.h5 into `master` under `name`.

    `link_src` maps a SHARED_DATASETS path to the first run's dataset; every
    later run hard-links to it instead of storing its own identical copy.
    """
    grp = master.create_group(name)
    grp.attrs.update(src.attrs)

    def visit(obj, path):
        for key in obj:
            item = obj[key]
            sub = f"{path}/{key}" if path else key
            if isinstance(item, h5py.Group):
                g = grp.create_group(sub)
                g.attrs.update(item.attrs)
                visit(item, sub)
            elif sub in SHARED_DATASETS and sub in link_src:
                grp[sub] = link_src[sub]              # hard link, no bytes
            else:
                kw = {}
                if item.size >= 4096:                 # chunking below is a loss
                    kw = dict(compression="gzip", compression_opts=1,
                              shuffle=True, chunks=True)
                d = grp.create_dataset(sub, data=item[...], **kw)
                d.attrs.update(item.attrs)
                if sub in SHARED_DATASETS:
                    link_src[sub] = d                 # first run holds the data

    visit(src, "")


def phase1_aggregate(out_root: Path) -> tuple[Path, list[str]]:
    """Validate, merge, postprocess. Returns (master_path, valid_run_names).
    
    Material reference IDs are read from each run's /scalars group
    (written by ml_export.cpp from ns3d_matalu and ns3d_matele).
    """
    manifest = out_root / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"Manifest not found at {manifest}")

    rows = list(csv.DictReader(manifest.open()))
    valid_folders, rejected = [], {}
    for r in rows:
        folder = out_root / r["folder"]
        ok, reason = validate_run(folder)
        if ok:
            valid_folders.append(folder)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    total = len(rows)
    accepted = len(valid_folders)
    print(f"Validation: {accepted}/{total} accepted")
    if rejected:
        for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
            print(f"  rejected: {count:4d}  {reason}")

    if not valid_folders:
        sys.exit("No valid runs to aggregate.")

    master_path = out_root / "master_dataset.h5"
    print(f"\nBuilding {master_path} ...")

    run_names = []
    pp_failures = 0
    link_src = {}          # shared dataset path -> first run's copy
    with h5py.File(master_path, 'w') as master:
        for i, folder in enumerate(valid_folders):
            name = folder.name
            with h5py.File(folder / "run.h5", 'r') as src:
                _copy_run(src, master, name, link_src)
            try:
                postprocess_run(master[name])
                run_names.append(name)
            except Exception as e:
                pp_failures += 1
                print(f"  [FAIL] {name}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  ... processed {i+1}/{accepted}")

    n_rejected = total - accepted
    print(f"\n{'='*55}")
    print(f"Phase 1 Summary")
    print(f"{'='*55}")
    print(f"  Total simulations : {total}")
    print(f"  Accepted (valid)  : {accepted}")
    print(f"  Rejected (invalid): {n_rejected}")
    if rejected:
        for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
            print(f"    {count:4d}  {reason}")
    print(f"  Postprocess OK    : {len(run_names)}")
    if pp_failures:
        print(f"  Postprocess FAIL  : {pp_failures}")
    print(f"  Final in master   : {len(run_names)}")
    print(f"{'='*55}")
    return master_path, run_names

# ======================================================================
# PHASE 2: COVERAGE ANALYSIS (read-only on master)
# ======================================================================
#
# Every statistic below answers exactly ONE question, and each is introduced
# by a comment saying which question it answers and — where it has been
# misread in the past — which question it does NOT answer.
#
# Four distinctions organise the whole section.  Most apparent contradictions
# between these numbers dissolve once the right one is applied.
#
#   1. DESIGN vs RESPONSE.
#      Input-space statistics describe the campaign we chose to run.  Field
#      statistics describe how the cell answered.  A high input rank together
#      with a low field rank is the ideal outcome, not a warning: it means the
#      sampling budget was not wasted AND the physics is compressible.  They
#      are not meant to agree.
#
#   2. ENERGY vs TRUNCATION.
#      Effective rank (participation ratio) says how many modes dominate the
#      energy.  n90/n95/n99 say how many modes are needed before truncation is
#      safe.  A spectrum with ~4 dominant modes and a long slow tail gives
#      eff_rank = 4.2 and n99 = 28 simultaneously.  Use n95/n99 to choose the
#      POD truncation k; eff_rank is a diversity headline only.
#
#   3. MARGINAL vs MULTIVARIATE.
#      With 24 inputs, even a purely LINEAR map c = A*I gives each output a
#      merely moderate correlation with any SINGLE anode, because the response
#      is spread over many of them.  Marginal per-anode correlation — Pearson
#      or Spearman — therefore cannot establish that the mapping is nonlinear.
#      Only a multivariate fit can.  That is what linear_predictability()
#      does, and it is why the old per-anode Pearson/Spearman comparison was
#      removed from this file.
#
#   4. MAGNITUDE vs SHAPE.
#      A coefficient of variation on a scalar integral (kinetic energy, |B|)
#      measures how much the MAGNITUDE moved.  Two flows with identical
#      kinetic energy can have completely different structure, so a low CV is
#      not evidence of low diversity.  The POD spectrum is the shape
#      measurement; CV is the magnitude measurement.  Report both.
#
# A note on the campaign mix.  If most runs come from the "gaussian" generator
# (small perturbations about uniform current), pooled CVs are dominated by
# that bulk and largely measure the generator's sigma rather than the physics.
# Every dispersion statistic here is therefore ALSO reported per campaign
# mode, so the extreme regimes (dead / single / weak) can be read separately.


#: The three mappings the surrogate has to learn, and therefore the three
#: fields every field-level statistic is run on:
#:      I -> velocity on the mid-ACD plane
#:      I -> full 3D velocity field
#:      I -> interface displacement
#: They are separate regression problems and need not behave alike, so none of
#: them is treated as a stand-in for the others.
DEFAULT_FIELDS = ("fields_midacd/vitesse",
                  "fields_full/vitesse",
                  "fields_interface/h")


# ── Campaign modes ────────────────────────────────────────────────────

def _load_modes(out_root: Path, run_names: list[str]) -> dict[str, str]:
    """Map run folder name -> campaign generation mode, from manifest.csv.

    Used to stratify every dispersion statistic.  Falls back to a single
    "all" group when the manifest is absent, so the analysis still runs.
    """
    manifest = out_root / "manifest.csv"
    if not manifest.exists():
        print("  [note] no manifest.csv — dispersion will not be stratified")
        return {n: "all" for n in run_names}

    f2m = {}
    with manifest.open() as f:
        for row in csv.DictReader(f):
            f2m[row["folder"]] = row["mode"]
    missing = [n for n in run_names if n not in f2m]
    if missing:
        print(f"  [note] {len(missing)} run(s) absent from manifest → mode 'unknown'")
    return {n: f2m.get(n, "unknown") for n in run_names}


# ── Generic helpers ───────────────────────────────────────────────────

def _standardize(X: np.ndarray):
    """Z-score each column. Returns (X_std, mean, std)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-15, 1.0, sd)
    return (X - mu) / sd, mu.ravel(), sd.ravel()


def _pca_variance(X: np.ndarray) -> np.ndarray:
    """Explained variance ratio from the SVD of centred X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2 or Xc.shape[1] == 0:
        return np.array([1.0])
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2) / max(Xc.shape[0] - 1, 1)
    total = var.sum()
    return var / total if total > 0 else np.zeros_like(var)


def _effective_rank(evr: np.ndarray) -> float:
    """Participation ratio of a variance distribution: 1 / sum(p_i^2).

    Reads as "how many components carry most of the energy".  It is dominated
    by the LARGEST eigenvalues and is deliberately insensitive to the tail —
    see distinction 2 in the section header.
    """
    p = evr / max(evr.sum(), 1e-30)
    return float(1.0 / max(np.sum(p ** 2), 1e-30))


def _dispersion(values: np.ndarray, modes: list[str]) -> dict:
    """Spread of one scalar quantity, pooled and per campaign mode.

    CV = 100 * std / |mean| is only interpretable when the mean is the right
    reference scale for the quantity.  It is NOT when the mean is dominated by
    a constant that cannot vary — the classic trap here being the total body
    force rho*g + jxB, whose mean is ~99.5% gravity, so that a perfectly
    healthy 10% variation of the Lorentz part shows up as a CV of 0.07%.
    See em_forcing_variance(), which measures the Lorentz force on its own.
    """
    values = np.asarray(values, dtype=np.float64)

    def _stats(v):
        if v.size == 0:
            return None
        mu = float(v.mean())
        return {
            "mean": mu, "std": float(v.std()),
            "cv": float(100.0 * v.std() / abs(mu)) if abs(mu) > 1e-30 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size),
        }

    out = {"overall": _stats(values), "by_mode": {}}
    modes = np.asarray(modes)
    for mode in sorted(set(modes.tolist())):
        s = _stats(values[modes == mode])
        if s is not None:
            out["by_mode"][mode] = s
    return out


def _log_dispersion(values: np.ndarray, modes: list[str],
                    tol: float | None = None) -> dict:
    """Spread of a positive quantity that ranges over several decades.

    CV = std/mean is the wrong summary for such a quantity: a sample spanning
    1e-10 to 1e-5 has a mean pinned by its largest members and a standard
    deviation of the same size, giving a CV of many hundred percent that says
    nothing except "this is a log-scale variable".  Median and percentiles do
    not have that failure mode.

    When *tol* is given (the value the run would have been rejected on), the
    worst-case margin is reported too, which is what one actually wants to
    know about a convergence residual.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    pos = v[v > 0]

    out = {
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        "n": int(v.size),
        "decades": (float(np.log10(pos.max() / pos.min()))
                    if pos.size and pos.min() > 0 else float("nan")),
        "geo_mean": float(np.exp(np.log(pos).mean())) if pos.size else float("nan"),
    }
    if tol is not None and out["max"] > 0:
        out["tol"] = tol
        out["worst_margin"] = tol / out["max"]

    modes = np.asarray(modes)[np.isfinite(np.asarray(values, dtype=np.float64))]
    out["by_mode"] = {}
    for mode in sorted(set(modes.tolist())):
        vm = v[modes == mode]
        if vm.size:
            out["by_mode"][mode] = {"median": float(np.median(vm)),
                                    "max": float(vm.max()), "n": int(vm.size)}
    return out


# ── 2A. Linear predictability — the multivariate coupling test ────────
#
# QUESTION ANSWERED
#   "How much of each output can a LINEAR function of the 24 anode currents
#    already explain, and how much is left for a nonlinear model to earn?"
#
# WHY THIS REPLACED THE PER-ANODE CORRELATIONS
#   The previous version of this file reported, for each output, the maximum
#   |Pearson| and |Spearman| over the 24 anodes, and read a Spearman-minus-
#   Pearson gap as evidence of nonlinear physics.  That inference does not
#   hold.  A purely linear map spreads each output over many anodes, so the
#   maximum MARGINAL correlation is moderate (0.3-0.6) no matter how linear
#   the map is; and a Spearman/Pearson gap only says that a single-variable
#   relationship is curved, not that the multivariate map I -> y is nonlinear.
#
# WHAT IT DOES ESTABLISH
#   r2_lin is the honest linear baseline, measured on held-out runs.  Any
#   surrogate must beat it to be worth its parameters.  r2_quad adds pairwise
#   products of the currents (ridge-regularised) as a cheap nonlinearity
#   probe, so that r2_quad - r2_lin is a defensible "nonlinear gap" — the one
#   the Spearman-minus-Pearson gap was standing in for.
#
# WHAT IT DOES NOT ESTABLISH
#   A low r2_lin does not by itself mean "nonlinear": it can equally mean the
#   output is numerical noise, unrelated to the currents.  Read it next to the
#   variance the target carries (for POD modes, next to its explained-variance
#   share) before concluding anything.

def _pairwise_features(Z: np.ndarray) -> np.ndarray:
    """[N, d] -> [N, d(d+1)/2]: all products z_i*z_j, i <= j (squares included).

    Only the pairwise block: the linear terms are handled separately so the
    quadratic model can be fitted as a correction ON TOP of the linear fit.
    """
    d = Z.shape[1]
    iu = np.triu_indices(d)
    return (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]


def _ridge_fit(Z: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge coefficients for centred/standardised Z, Y (no intercept term)."""
    G = Z.T @ Z + alpha * np.eye(Z.shape[1])
    return np.linalg.solve(G, Z.T @ Y)


def _hierarchical_ridge(D_lin: np.ndarray, D_quad: np.ndarray,
                        y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit [linear | quadratic] jointly, penalising the quadratic block only.

    Fitting the quadratic terms to the residual of a FIXED linear model caps
    what they can achieve: where the linear fit is actively harmful (a target
    with no linear signal still fits ~d/n of the noise, giving a negative
    held-out R^2), the correction inherits that damage and cannot remove it,
    because the linear coefficients are frozen.  Estimating both blocks
    together lets the linear part be revised once the quadratic terms explain
    the structure it was mistaking for signal.

    The penalty is applied to the quadratic block alone, so alpha -> infinity
    recovers the plain linear fit rather than shrinking everything towards
    zero.  The linear block still carries a token ridge, scaled to the Gram
    matrix, because the currents sum to a constant and the block is therefore
    exactly rank-deficient.
    """
    D = np.hstack([D_lin, D_quad])
    n_lin, n_quad = D_lin.shape[1], D_quad.shape[1]
    lam0 = 1e-6 * max(len(D), 1)
    P = np.concatenate([np.full(n_lin, lam0), np.full(n_quad, alpha)])
    return np.linalg.solve(D.T @ D + np.diag(P), D.T @ y)


def _r2_columns(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Held-out R^2 per column, against each column's own test-set mean."""
    ss_res = ((Y_true - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-30)


def linear_predictability(X: np.ndarray, Y: np.ndarray,
                          names: list[str] | None = None,
                          test_frac: float = 0.25,
                          quadratic: bool = True,
                          seed: int = 0) -> dict:
    """Held-out linear (and quadratic) predictability of Y from X.

    Args:
        X:      [N, d]  anode currents.
        Y:      [N, m]  targets (POD coefficients, or scalar diagnostics).
        names:  optional column names for Y.
        quadratic: also fit pairwise products as a nonlinearity probe.

    Returns dict with r2_lin, r2_quad, gap (all [m]), the dominant input per
    target, and the sizes of the split actually used.
    """
    N, d = X.shape
    m = Y.shape[1]
    names = names or [str(i) for i in range(m)]

    # Too few runs for a trustworthy held-out estimate.
    n_test = int(round(N * test_frac))
    if N < 30 or n_test < 5:
        return {"error": f"only {N} runs — too few for a held-out fit"}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    te, tr = perm[:n_test], perm[n_test:]

    # Standardise on the TRAIN split only, then apply to both.
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
    mu_y, sd_y = Y[tr].mean(0), Y[tr].std(0)
    sd_x = np.where(sd_x < 1e-15, 1.0, sd_x)
    sd_y = np.where(sd_y < 1e-15, 1.0, sd_y)
    Zx = (X - mu_x) / sd_x
    Zy = (Y - mu_y) / sd_y

    # --- Linear: least squares with the constraint direction truncated.
    # The currents sum to a fixed total, so X is exactly rank-deficient: 23 of
    # 24 directions are populated and the 24th sits at numerical-noise level
    # (singular value ~1e-6 of the largest).  rcond=None uses a machine-epsilon
    # cutoff, which KEEPS that direction and inverts it — predictions barely
    # change, because the data has no component along it, but the coefficients
    # blow up along the null direction and every target then appears to be
    # "driven" by whichever anode that vector happens to weight most.
    # An explicit cutoff discards it and fixes the gauge, which is what makes
    # the dominant-anode column below meaningful.
    _RCOND = 1e-6
    beta, *_ = np.linalg.lstsq(Zx[tr], Zy[tr], rcond=_RCOND)
    r2_lin = _r2_columns(Zy[te], Zx[te] @ beta)

    # Sum of squares of each target on the SCORED runs, in physical units.
    # Zy was standardised column-wise, so the physical spread is recovered by
    # sd_y**2.  These are the weights that make the variance-weighted mean of
    # the per-mode R^2 equal the pooled R^2 of the reconstructed field exactly;
    # the POD's explained-variance ratio is the same quantity estimated on the
    # fitting set instead, and agrees only to about 1e-3.
    sst_test = ((Zy[te] - Zy[te].mean(0)) ** 2).sum(axis=0) * sd_y ** 2

    # Which anode carries the largest standardised weight for each target.
    # This is a REGRESSION coefficient, not a marginal correlation: it is the
    # influence of that anode with the others held fixed, which is what one
    # actually means by "this mode is driven by anode k".  It is only defined
    # modulo the sum constraint, which the truncation above pins down.
    dominant = np.argmax(np.abs(beta), axis=0)

    # --- Quadratic: pairwise products fitted to the LINEAR MODEL'S RESIDUAL.
    # Two reasons for the nesting rather than one ridge over [linear|quadratic]:
    #   * a single ridge penalises the linear coefficients too, so a strongly
    #     regularised fit shrinks towards zero rather than towards the linear
    #     solution, and can score WORSE than linear on held-out data — which
    #     would make the "nonlinear gap" negative and meaningless;
    #   * fitted this way, a large alpha drives the correction to zero and
    #     recovers the linear fit exactly, so the gap measures what the
    #     pairwise terms add ON TOP of the best linear model, which is the
    #     quantity of interest.
    # The pairwise block is standardised on the train split as well: squared
    # features have non-zero mean, and the ridge carries no intercept.
    r2_quad = np.full(m, np.nan)
    if quadratic:
        Q = _pairwise_features(Zx)
        mu_q, sd_q = Q[tr].mean(0), Q[tr].std(0)
        sd_q = np.where(sd_q < 1e-15, 1.0, sd_q)
        Q = (Q - mu_q) / sd_q

        # Inner split of the training set to pick the ridge strength; the test
        # split stays untouched so the reported R^2 remains out-of-sample.
        n_val = max(5, int(0.2 * len(tr)))
        va, tr2 = tr[:n_val], tr[n_val:]
        if len(tr2) > 10:
            b_in, *_ = np.linalg.lstsq(Zx[tr2], Zy[tr2], rcond=_RCOND)
            resid_in = Zy[tr2] - Zx[tr2] @ b_in

            # Two things are chosen PER TARGET here, and both matter.
            #
            # SCREENING.  Plain ridge over all d(d+1)/2 products cannot
            # recover a SPARSE nonlinearity: ridge shrinks every coefficient
            # by the same factor and never concentrates weight, so one true
            # product among ~300 collinear features is buried by the other
            # 299.  Sparsity is the physically likely case — an interaction
            # between two NEIGHBOURING anodes is one specific product — so
            # each target first keeps only the products most correlated with
            # its own linear residual, and the ridge is fitted on those.
            # Screening uses training rows only; the test split is untouched.
            #
            # ALPHA.  Chosen per target rather than once on the mean score.
            # Most modes are already at R^2 ~ 0.99 from the linear part and
            # are only degraded by a correction; a single shared alpha is
            # decided by that majority and switches the quadratic term off
            # for the few modes that genuinely need it — exactly the
            # interesting ones.  The largest alpha leaves the correction
            # negligible, so a target with no quadratic signal falls back to
            # its linear fit.
            alphas = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e6, 1e8)
            n_keep = int(min(40, max(5, len(tr) // 8)))

            b_in, *_ = np.linalg.lstsq(Zx[tr2], Zy[tr2], rcond=_RCOND)
            resid_in = Zy[tr2] - Zx[tr2] @ b_in
            resid_tr = Zy[tr] - Zx[tr] @ beta
            lin_va = Zx[va] @ b_in

            score_in = Q[tr2].T @ resid_in / max(len(tr2), 1)
            score_tr = Q[tr].T @ resid_tr / max(len(tr), 1)

            # Two candidate feature sets, chosen per target on the validation
            # split.  Screening recovers a SPARSE nonlinearity (one product of
            # two neighbouring anodes) that plain ridge buries among 300
            # collinear features; the full basis recovers a DENSE one (a full
            # quadratic form, as any energy-like quantity is) that screening
            # would truncate.  Which case applies cannot be known in advance,
            # so both are offered and the data decides.
            n_feat = Q.shape[1]
            candidates = [n_keep] + ([n_feat] if n_feat > n_keep else [])

            pred_te = np.empty((len(te), m))
            for k in range(m):
                order_in = np.argsort(np.abs(score_in[:, k]))[::-1]
                order_tr = np.argsort(np.abs(score_tr[:, k]))[::-1]

                best_cfg, best = (candidates[0], alphas[-1]), -np.inf
                for n_sel in candidates:
                    sel_in = order_in[:n_sel]
                    D_va = np.hstack([Zx[va], Q[np.ix_(va, sel_in)]])
                    for alpha in alphas:
                        W = _hierarchical_ridge(Zx[tr2], Q[np.ix_(tr2, sel_in)],
                                                Zy[np.ix_(tr2, [k])], alpha)
                        r2 = _r2_columns(Zy[np.ix_(va, [k])], D_va @ W)[0]
                        if np.isfinite(r2) and r2 > best:
                            best, best_cfg = r2, (n_sel, alpha)

                n_sel, alpha = best_cfg
                sel_tr = order_tr[:n_sel]
                W = _hierarchical_ridge(Zx[tr], Q[np.ix_(tr, sel_tr)],
                                        Zy[np.ix_(tr, [k])], alpha)
                D_te = np.hstack([Zx[te], Q[np.ix_(te, sel_tr)]])
                pred_te[:, k] = (D_te @ W)[:, 0]

            r2_quad = _r2_columns(Zy[te], pred_te)

    return {
        "names": list(names),
        "r2_lin": r2_lin,
        "r2_quad": r2_quad,
        "gap": r2_quad - r2_lin,
        "dominant_input": dominant,
        "sst_test": sst_test,
        "n_train": len(tr), "n_test": len(te),
    }


# ── Reading the master file ───────────────────────────────────────────

def _collect_scalars(master: h5py.File, exclude_dead: bool,
                     dead_thresh: float
                     ) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Read per-anode currents and every scalar diagnostic from all runs.

    Returns (X_inputs, df_scalars, run_names) for the selected subset, with
    the three kept strictly in the same row order.
    """
    inputs, rows, names = [], [], []
    n_skipped_no_currents = n_skipped_dead = n_errors = 0

    for name in sorted(master.keys()):
        grp = master[name]
        if not isinstance(grp, h5py.Group):
            continue
        if "input/currents" not in grp:
            n_skipped_no_currents += 1
            continue
        try:
            curr = grp["input/currents"][:].astype(np.float64).ravel()
            if exclude_dead and np.any(curr < dead_thresh):
                n_skipped_dead += 1
                continue
            inputs.append(curr)
            d = {"run": name}
            for sub, prefix in [("scalars", "attr_"), ("derived_metrics", "")]:
                if sub in grp:
                    for k, v in grp[sub].attrs.items():
                        try:
                            d[prefix + k] = float(np.asarray(v).item())
                        except Exception:
                            pass
            rows.append(d)
            names.append(name)
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  [collect ERROR] {name}: {type(e).__name__}: {e}")

    print(f"  [collect_scalars] kept={len(rows)}  "
          f"no_currents={n_skipped_no_currents}  dead={n_skipped_dead}  "
          f"errors={n_errors}")

    if not rows:
        sys.exit("No runs found in master for coverage analysis. "
                 "Check that /input/currents exists inside each run group.")

    X = np.vstack(inputs)
    df = pd.DataFrame(rows).set_index("run")
    return X, df, names


# ── 2B. Input-space coverage ──────────────────────────────────────────
#
# QUESTION ANSWERED
#   "Did the campaign explore the feasible operating space, or did it keep
#    re-running the same conditions?"
#
# This describes the DESIGN only.  It says nothing about how the cell
# responded, and it is not meant to agree with the field ranks below.
#
# IMPORTANT — the denominator is 23, not 24.  The generator enforces
# sum(I_k) = 490 kA exactly, so the input covariance has one exactly-zero
# eigenvalue and at most 23 directions can ever be populated.  Quoting an
# effective rank "out of 24" understates the coverage.

def input_space_coverage(X_in: np.ndarray, modes: list[str]) -> dict:
    evr = _pca_variance(X_in)
    n_dof = X_in.shape[1] - 1          # sum constraint removes one dof

    # Nearest-neighbour distance in standardised input space.
    # Near-zero values mean duplicated operating points; the median is a
    # compact measure of how finely the space is filled.
    Xs, _, _ = _standardize(X_in)
    d2 = (np.sum(Xs**2, axis=1, keepdims=True)
          + np.sum(Xs**2, axis=1, keepdims=True).T
          - 2 * Xs @ Xs.T)
    np.fill_diagonal(d2, np.inf)
    nn_dists = np.sqrt(np.maximum(d2.min(axis=1), 0.0))

    cum = np.cumsum(evr)
    return {
        "evr": evr,
        "n_dof": n_dof,
        "eff_rank": _effective_rank(evr),
        "dims_95": int(np.searchsorted(cum, 0.95)) + 1,
        "nn_dists": nn_dists,
        "nn_min": float(nn_dists.min()),
        "nn_median": float(np.median(nn_dists)),
        "by_mode": {m: int(np.sum(np.asarray(modes) == m))
                    for m in sorted(set(modes))},
    }


# ── 2C. Scalar diagnostics ────────────────────────────────────────────
#
# QUESTION ANSWERED
#   "Which of the scalar quantities the solver reports actually respond to the
#    anode currents, and how much of that response is linear?"
#
# Two failure modes are flagged:
#   * near-constant outputs (low CV), which carry no information for a
#     surrogate — but see the CV caveat in _dispersion();
#   * unpredictable outputs (low r2_lin AND low r2_quad), which either do not
#     depend on the currents or are numerical noise.
#
# NOTE ON kinematic_residual_*.  The kinematic condition u.n = 0 is imposed
# exactly at every iteration through a Lagrange multiplier, so a nonzero
# max|u.n| is the POINTWISE DISCRETISATION RESIDUAL of a constraint that holds
# in the weak sense — it is a convergence-quality diagnostic, not a physical
# "interface stability" observable.  Its correlation with the currents most
# plausibly says that harder runs converge less tightly.  It is reported below
# under convergence quality and deliberately kept out of any statement about
# nonlinear physics.  (It is used as a rejection criterion in Phase 1, which
# is the correct use of it.)

_CONVERGENCE_KEYS = ("attr_kinematic_residual_max", "attr_kinematic_residual_mean",
                     "attr_statk")


def _numeric_frame(df_out: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Numeric, non-constant columns, with NaNs filled by the column mean."""
    Y = df_out.select_dtypes(include=[np.number]).copy()
    Y = Y.loc[:, Y.notna().any() & (Y.nunique(dropna=True) > 1)]
    A = Y.to_numpy(dtype=np.float64, copy=True)
    nans = np.where(np.isnan(A))
    if nans[0].size:
        A[nans] = np.take(np.nanmean(A, axis=0), nans[1])
    return Y, A


def scalar_diagnostics(X_in: np.ndarray, df_out: pd.DataFrame,
                       modes: list[str], cv_thresh: float = 2.0,
                       r2_thresh: float = 0.1) -> dict:
    Y, A = _numeric_frame(df_out)
    cols = list(Y.columns)

    physics_cols = [c for c in cols if c not in _CONVERGENCE_KEYS]
    phys_idx = [cols.index(c) for c in physics_cols]

    pred = linear_predictability(X_in, A[:, phys_idx], names=physics_cols)

    # An output can be unpredictable for two very different reasons, and the
    # CV separates them: a near-constant quantity has nothing to predict (R^2
    # is then meaningless noise around zero), whereas one that varies but is
    # still unpredictable is either independent of the currents or numerical
    # noise.  The CV is therefore computed once and reported next to R^2.
    cv_of = {}
    for c in cols:
        v = Y[c].dropna().to_numpy(dtype=np.float64)
        if v.size >= 2 and abs(v.mean()) > 1e-30:
            cv_of[c] = 100.0 * v.std() / abs(v.mean())

    low_var = sorted(((c, float(cv)) for c, cv in cv_of.items()
                      if cv < cv_thresh), key=lambda t: t[1])

    unpredictable = []
    if "error" not in pred:
        for i, c in enumerate(pred["names"]):
            best = np.nanmax([pred["r2_lin"][i], pred["r2_quad"][i]])
            if best < r2_thresh:
                unpredictable.append((c, float(pred["r2_lin"][i]), float(best),
                                      cv_of.get(c, float("nan"))))

    # Convergence residuals span decades, so they get log-scale statistics.
    convergence = {}
    for c in _CONVERGENCE_KEYS:
        if c not in Y.columns:
            continue
        tol = KINEMATIC_RESIDUAL_TOL if "kinematic_residual" in c else None
        convergence[c] = _log_dispersion(Y[c].to_numpy(dtype=np.float64),
                                         modes, tol=tol)

    return {"pred": pred, "low_var": low_var, "unpredictable": unpredictable,
            "convergence": convergence, "columns": cols}


# ── 2D. Field POD — the SHAPE measurement ─────────────────────────────
#
# QUESTION ANSWERED
#   "How many spatial patterns are needed to represent the whole family of
#    computed flow fields, i.e. can the field be compressed enough for a
#    surrogate to predict a handful of coefficients instead of ~10^5 DOFs?"
#
# This is the diversity measurement that a coefficient of variation cannot
# make: two runs with identical kinetic energy still need separate POD modes
# if their circulation patterns differ.  Read eff_rank and n90/n95/n99 as the
# two different things they are (distinction 2 in the header): eff_rank is a
# headline, n95/n99 choose the truncation k.
#
# Coupled POD: for a vector field the three components are flattened into one
# vector before the SVD, so each mode captures correlated structure across
# components.  Because every snapshot satisfies the same linear constraints
# (discrete incompressibility, u.n = 0 on the walls), every mode — being a
# linear combination of snapshots — satisfies them too, and so does any
# reconstruction.  That holds exactly only insofar as the runs share a mesh;
# the interface moves between runs, so it degrades with interface deformation.

def field_pod(master: h5py.File, field_path: str,
              exclude_dead: bool, dead_thresh: float,
              max_modes: int = 50) -> dict:
    run_names = sorted(k for k in master.keys()
                       if isinstance(master[k], h5py.Group)
                       and field_path in master[k])
    if exclude_dead:
        keep = []
        for n in run_names:
            if "input/currents" in master[n]:
                curr = master[n]["input/currents"][:].ravel()
                if np.any(np.abs(curr) < dead_thresh):
                    continue
            keep.append(n)
        run_names = keep

    if len(run_names) < 10:
        return {"error": f"only {len(run_names)} runs have {field_path}"}

    M = len(master[run_names[0]][field_path][()].ravel())

    # Every snapshot must have the same number of DOFs.  The mesh topology is
    # fixed across runs (only node positions move), so a mismatch means a run
    # was exported from a different mesh — drop it rather than crash.
    keep = []
    for n in run_names:
        if master[n][field_path].size == M:
            keep.append(n)
    if len(keep) < len(run_names):
        print(f"  [WARN] {len(run_names) - len(keep)} run(s) dropped: "
              f"{field_path} has a different DOF count")
        run_names = keep
    if len(run_names) < 10:
        return {"error": f"only {len(run_names)} runs with a consistent {field_path}"}

    N = len(run_names)
    print(f"  Field POD: {N} snapshots x {M} DOFs")

    U = np.empty((N, M), dtype=np.float32)
    for i, n in enumerate(run_names):
        U[i, :] = master[n][field_path][()].astype(np.float32).ravel()

    u_mean = U.mean(axis=0)
    U -= u_mean                                   # centre in place to save RAM
    k = min(max_modes, N - 1, M)

    # Method of snapshots: eigendecompose the N x N Gram matrix rather than
    # the M x M covariance, since N << M here.
    G = (U @ U.T) / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(G.astype(np.float64))
    idx = np.argsort(eigvals)[::-1][:k]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx].astype(np.float32)

    V = U.T @ eigvecs
    V /= np.linalg.norm(V, axis=0, keepdims=True).clip(min=1e-12)
    coeffs = U @ V

    evr = eigvals / max(eigvals.sum(), 1e-30)
    cumevr = np.cumsum(evr)

    def _n_for(thr):
        i = np.searchsorted(cumevr, thr)
        return int(i) + 1 if i < len(cumevr) else len(cumevr)

    # Relative size of the fluctuation about the mean field: the honest
    # "how big is the perturbation" number, and the one that tells you whether
    # delta-learning against a reference run is worthwhile.  Note this is an
    # L2 ratio over the whole field, NOT a ratio of peak values — comparing a
    # perturbation against the field's MAXIMUM rather than its rms understates
    # it, typically by an order of magnitude.
    # Undefined when the mean field is itself ~0 (a field stored as a
    # deviation), so it is reported as NaN rather than as a huge ratio.
    mean_norm = float(np.linalg.norm(u_mean))
    rms_fluct = float(np.sqrt((U ** 2).sum(axis=1).mean()))
    fluct = rms_fluct / mean_norm if mean_norm > 1e-12 * max(rms_fluct, 1e-30) \
        else float("nan")

    return {
        "evr": evr, "cumevr": cumevr,
        "n90": _n_for(0.90), "n95": _n_for(0.95), "n99": _n_for(0.99),
        "eff_rank": _effective_rank(evr),
        "top3": float(cumevr[min(2, len(cumevr) - 1)]),
        "N": N, "M": M, "coeffs": coeffs, "run_names": run_names,
        "fluct_rel": fluct,
    }


# ── 2D-bis. How much does the mesh itself move? ───────────────────────
#
# QUESTION ANSWERED
#   "The POD is taken in NODE-INDEX space.  Node i is the same node of the
#    same mesh in every run, but it sits at a different place, because the
#    interface deforms and the mesh is deformed to follow it.  How much of
#    what the modes capture is the flow changing, and how much is the mesh
#    moving underneath it?"
#
# The mesh topology is fixed across the campaign and only the node positions
# vary, so the snapshot matrix is well defined; but a mode is a pattern of
# velocity values indexed by node, and if node i has moved by an amount
# comparable to the length scale over which the velocity varies, part of the
# apparent change in the field is a change of sampling point rather than a
# change of flow.  This is the ALE frame, and it is the right frame to work
# in here — but its contamination has to be bounded rather than assumed.
#
# The comparison to make is between the node displacement and the local
# element size: a displacement well below one element means the field is
# being resampled a small fraction of the distance over which it varies, and
# the POD is clean.  The absolute displacement is reported in millimetres
# too, because that is the number to quote next to an interface deformation.

def mesh_motion(master: h5py.File, exclude_dead: bool,
                dead_thresh: float) -> dict:
    """Run-to-run displacement of the mesh nodes, in one streaming pass."""
    out = {}
    for label, path in (("cell mesh", "mesh/cuveb_nodes"),
                        ("interface", "mesh/interface_nodes")):
        s1 = s2 = None
        n = 0
        n_elems = None
        for name in sorted(master.keys()):
            grp = master[name]
            if not isinstance(grp, h5py.Group) or path not in grp:
                continue
            if exclude_dead and "input/currents" in grp:
                if np.any(grp["input/currents"][:].ravel() < dead_thresh):
                    continue
            X = grp[path][()].astype(np.float64)
            if s1 is None:
                s1 = np.zeros_like(X)
                s2 = np.zeros(X.shape[0])
                if "mesh/cuveb_elems" in grp:
                    n_elems = grp["mesh/cuveb_elems"].shape[0]
            elif X.shape != s1.shape:
                continue
            s1 += X
            s2 += (X ** 2).sum(axis=1)
            n += 1

        if n < 2:
            continue

        mean = s1 / n
        # Per-node r.m.s. displacement about the mean position:
        #   E||x - xbar||^2 = E||x||^2 - ||xbar||^2
        var = np.maximum(s2 / n - (mean ** 2).sum(axis=1), 0.0)
        disp = np.sqrt(var)

        extent = mean.max(axis=0) - mean.min(axis=0)
        diag = float(np.linalg.norm(extent))
        rec = {"n": n, "mean_mm": float(1e3 * disp.mean()),
               "max_mm": float(1e3 * disp.max()),
               "diag_m": diag,
               "rel_domain": float(disp.mean() / max(diag, 1e-30))}

        # Mean element size, estimated from the bounding box and the element
        # count: the length scale the displacement has to be compared against.
        if n_elems and label == "cell mesh":
            h_elem = float((np.prod(extent) / n_elems) ** (1.0 / 3.0))
            rec["h_elem_mm"] = 1e3 * h_elem
            rec["disp_over_h"] = float(disp.mean() / max(h_elem, 1e-30))
        out[label] = rec
    return out


# ── 2E. POD modes vs anode currents ───────────────────────────────────
#
# QUESTION ANSWERED
#   "Are the POD coefficients actually predictable from the anode currents,
#    how much of that is linear, and where should the basis be truncated?"
#
# This replaces the old per-anode correlation table (see 2A for why).  It also
# gives a principled truncation rule: keep the modes the currents PREDICT, not
# merely the modes that carry variance.  A mode with variance but no
# predictability is noise as far as the surrogate is concerned — including it
# can only add error.

def pod_linear_model(pod: dict, master: h5py.File,
                     r2_thresh: float = 0.5,
                     min_evr: float = 1e-4) -> dict:
    if "error" in pod:
        return pod

    names = pod["run_names"]
    coeffs = pod["coeffs"]

    rows, keep = [], []
    for i, n in enumerate(names):
        if "input/currents" in master[n]:
            rows.append(master[n]["input/currents"][:].ravel())
            keep.append(i)
    if not rows:
        return {"error": "no currents found for the POD runs"}
    X = np.stack(rows).astype(np.float64)
    C = coeffs[keep]

    pred = linear_predictability(X, C.astype(np.float64),
                                 names=[f"mode{i}" for i in range(C.shape[1])])
    if "error" in pred:
        return pred

    best = np.fmax(pred["r2_lin"], np.nan_to_num(pred["r2_quad"], nan=-np.inf))

    # A mode is worth keeping only if it carries variance AND is predictable.
    # Predictability alone is not enough: a mode holding a negligible share of
    # the field can still be predicted almost perfectly (there is barely any
    # noise in it to get wrong), and letting such a mode extend the truncation
    # would add outputs that contribute nothing to the reconstruction.
    evr_all = pod["evr"][:len(best)]
    worth = evr_all >= min_evr
    ok = worth & (best >= r2_thresh)

    learnable = int(ok.sum())
    below = np.where(~ok)[0]
    k_suggested = int(below[0]) if below.size else len(best)

    # Variance-weighted R^2 — the headline number.  A plain mean over modes is
    # misleading: it gives a noise mode carrying 0.01% of the variance the same
    # weight as mode 0 carrying 40%, so it reads near zero even when the field
    # is almost perfectly predicted.  Weighting by explained variance gives the
    # fraction of the FIELD's variance that the fit reproduces, which is what
    # "how much of the flow is linear in the currents?" actually asks.
    sst = np.asarray(pred.get("sst_test", []), dtype=float)[:len(best)]
    if sst.size == len(best) and np.isfinite(sst).all() and sst.sum() > 0:
        w = sst / sst.sum()                    # exact: identity with the field score
    else:                                      # scalars, or an older cache
        evr = pod["evr"][:len(best)]
        w = evr / max(evr.sum(), 1e-30)
    r2_field_lin = float(np.nansum(w * pred["r2_lin"]))
    r2_field_quad = float(np.nansum(w * pred["r2_quad"])) \
        if not np.all(np.isnan(pred["r2_quad"])) else float("nan")

    pred.update({"learnable_modes": learnable, "k_suggested": k_suggested,
                 "r2_thresh": r2_thresh, "min_evr": min_evr, "evr": evr,
                 "r2_field_lin": r2_field_lin, "r2_field_quad": r2_field_quad})
    return pred


# ── 2F. Electromagnetic forcing ───────────────────────────────────────
#
# QUESTION ANSWERED
#   "Which parts of the electromagnetic state actually change when the current
#    is redistributed — i.e. what is the SIGNAL a surrogate has to learn, and
#    what is a fixed background it can safely be trained relative to?"
#
# THE FIX THAT MATTERS HERE.  The previous version measured the dispersion of
# the total body force f = rho*g + jxB as stored in fields_full/forces.  That
# number is meaningless as a driving-force diagnostic, for two reasons:
#
#   * SCALE.  |rho*g| ~ 2.2e4 N/m^3 while |jxB| ~ 1e2 N/m^3.  The numerator of
#     the CV is the Lorentz variation, but the denominator is essentially
#     gravity, so the CV comes out ~2 orders of magnitude too small.  A ~10%
#     variation of the Lorentz force was being reported as 0.07%.
#   * PHYSICS.  rho*g is a potential force, absorbed by the hydrostatic
#     pressure gradient; it drives no flow at all.  Including it in a
#     "driving force diversity" metric measures a term that by construction
#     cannot produce diversity.
#
# So the Lorentz force is now computed explicitly as jxB from the exported
# current density (P0) and induction (P1, projected onto the elements), and
# reported on its own.  Both it and fields_full/forces then live in element
# space, so the gravity dominance ratio printed alongside is exact and the old
# number stays interpretable.
#
# |B| is kept and IS meaningful as reported: B = B_hor + dB, and B_hor is
# computed once in preprocessing and reused for every run at a fixed total
# current, so a CV of a few tenths of a percent is the expected structural
# result, not a measurement artefact.  It is exactly what justifies training
# on perturbations about a reference run.  It also scopes the surrogate: at a
# different total-current setpoint B_hor is recomputed, and this evidence no
# longer applies.

class _FieldFluctuation:
    """Streaming ||f_i - f_mean|| / ||f_mean|| over a set of runs.

    The CV of a field's spatial MEAN magnitude answers "did the force get
    bigger?".  It cannot answer "did the force pattern change?" — the Lorentz
    force can reorganise completely while its mean magnitude is unchanged,
    which is distinction 4 of the section header applied to the forcing rather
    than to the flow.  This measures the second question, exactly as field_pod
    does for the velocity.

    Single pass and O(M) memory: sum(||f_i - f_bar||^2) expands to
    sum(||f_i||^2) - N*||f_bar||^2, so only the running field sum and a scalar
    sum of squares are needed — the snapshots are never all held at once,
    which matters at ~10^6 elements x 1251 runs.
    """

    def __init__(self):
        self.sum = None
        self.sq = 0.0
        self.n = 0

    def add(self, field: np.ndarray):
        f = np.asarray(field, dtype=np.float64).ravel()
        if self.sum is None:
            self.sum = np.zeros_like(f)
        elif f.shape != self.sum.shape:
            return                      # different mesh: skip this run
        self.sum += f
        self.sq += float(f @ f)
        self.n += 1

    def result(self) -> float:
        if self.n < 2 or self.sum is None:
            return float("nan")
        mean = self.sum / self.n
        mean_sq = float(mean @ mean)
        if mean_sq <= 0:
            return float("nan")
        var = max(self.sq / self.n - mean_sq, 0.0)
        return float(np.sqrt(var / mean_sq))


def em_forcing_variance(master: h5py.File, modes_of: dict,
                        exclude_dead: bool, dead_thresh: float) -> dict:
    vals: dict[str, list] = {"lorentz": [], "B_norm": [],
                             "j_horiz_frac_alu": [], "f_body_total": []}
    mods: dict[str, list] = {k: [] for k in vals}
    ratios = []
    fluct = {"lorentz": _FieldFluctuation(), "B": _FieldFluctuation()}

    for name in sorted(master.keys()):
        grp = master[name]
        if not isinstance(grp, h5py.Group):
            continue
        if exclude_dead and "input/currents" in grp:
            if np.any(grp["input/currents"][:].ravel() < dead_thresh):
                continue
        mode = modes_of.get(name, "unknown")

        has_j = "fields_full/courant" in grp
        has_B = "fields_full/induction" in grp
        elems = grp["mesh/cuveb_elems"][:] if "mesh/cuveb_elems" in grp else None

        if has_B:
            B = grp["fields_full/induction"][()].astype(np.float32)
            vals["B_norm"].append(float(np.linalg.norm(B, axis=1).mean()))
            mods["B_norm"].append(mode)
            fluct["B"].add(B)

        if has_j and has_B:
            j = grp["fields_full/courant"][()].astype(np.float32)

            # j is P0 (Ne, 3) and B is P1 (Nn, 3), so the two never share a
            # shape and must be brought to a common space before the cross
            # product.  Skipping on a shape mismatch would silently drop the
            # Lorentz force — the one quantity this section exists to measure.
            # Element space is the right target: it is where j, forces and the
            # material tags already live, so the resulting jxB is directly
            # comparable with fields_full/forces below, and the projection of
            # the P1 induction is exact (see _to_element_space).
            jE = j if j.shape == B.shape else _to_element_space(j, elems)
            BE = B if j.shape == B.shape else _to_element_space(B, elems)

            if jE is not None and BE is not None and jE.shape == BE.shape:
                fL = np.cross(jE, BE)                     # N/m^3
                fL_mean = float(np.linalg.norm(fL, axis=1).mean())
                vals["lorentz"].append(fL_mean)
                mods["lorentz"].append(mode)
                fluct["lorentz"].add(fL)

                if "fields_full/forces" in grp:
                    F = grp["fields_full/forces"][()].astype(np.float32)
                    tot = float(np.linalg.norm(F, axis=1).mean())
                    vals["f_body_total"].append(tot)
                    mods["f_body_total"].append(mode)
                    if fL_mean > 1e-30:
                        ratios.append(tot / fL_mean)

        # Horizontal current fraction in the METAL only: the MHD-instability
        # proxy.  Cross-currents in the pad are what the Lorentz force acts on.
        if has_j and "mesh/cuveb_refs" in grp and "scalars" in grp:
            ref_alu = grp["scalars"].attrs.get("ref_alu")
            if ref_alu is not None:
                refs = grp["mesh/cuveb_refs"][:].ravel()
                j = grp["fields_full/courant"][()].astype(np.float32)
                jm = _material_values(j, elems, refs, int(ref_alu))
                if jm is not None and jm.size:
                    jh = np.linalg.norm(jm[:, :2], axis=1).sum()
                    jz = np.abs(jm[:, 2]).sum()
                    vals["j_horiz_frac_alu"].append(float(jh / max(jz, 1e-30)))
                    mods["j_horiz_frac_alu"].append(mode)

    out = {k: _dispersion(np.array(v), mods[k]) for k, v in vals.items() if v}
    if ratios:
        out["_gravity_dominance"] = float(np.mean(ratios))
    out["_fluct"] = {k: f.result() for k, f in fluct.items() if f.n}
    return out


# ── 2G. Macroscopic flow topology ─────────────────────────────────────
#
# QUESTION ANSWERED
#   "By how much does the MAGNITUDE of the large-scale flow move across the
#    campaign?"
#
# These are plane integrals computed inside the solver (exact tetrahedral
# slicing) — kinetic energy and circulation rate on the mid-ACD plane and on
# the X/Y cross-sections.
#
# WHAT THEY DO NOT ESTABLISH.  They are scalar integrals, so they cannot show
# that the flow STRUCTURE varies: a completely reorganised circulation can
# integrate to the same kinetic energy.  Do not read a low CV here as "the
# dataset lacks diversity" — the POD spectrum (2D) is the statistic that
# answers that, and the two together give the accurate picture: a nearly
# invariant base flow, modulated at the percent level, whose modulation is
# spatially rich.
#
# Read the per-mode breakdown, not just the pooled CV.  If most runs come from
# the "gaussian" generator, the pooled number largely reports that generator's
# sigma rather than the physics of the cell.

_FLOW_METRICS = ("ke_mid_acd", "ke_plane_X", "ke_plane_Y",
                 "half_abs_flux_plane_X", "half_abs_flux_plane_Y",
                 "u_max_midacd", "u_rms_midacd", "eta_ptp")


def flow_topology_variance(master: h5py.File, modes_of: dict,
                           exclude_dead: bool, dead_thresh: float) -> dict:
    vals = {m: [] for m in _FLOW_METRICS}
    mods = {m: [] for m in _FLOW_METRICS}

    for name in sorted(master.keys()):
        grp = master[name]
        if not isinstance(grp, h5py.Group):
            continue
        if exclude_dead and "input/currents" in grp:
            if np.any(grp["input/currents"][:].ravel() < dead_thresh):
                continue
        mode = modes_of.get(name, "unknown")

        # A metric may be written by the C++ export (/scalars) or by the
        # Python postprocessing (/derived_metrics).  Take the first source
        # that has it, so a run contributes at most one value per metric.
        for m in _FLOW_METRICS:
            for src in ("scalars", "derived_metrics"):
                if src in grp and m in grp[src].attrs:
                    try:
                        v = float(np.asarray(grp[src].attrs[m]).item())
                    except Exception:
                        break
                    if np.isfinite(v):
                        vals[m].append(v)
                        mods[m].append(mode)
                    break

    return {m: _dispersion(np.array(v), mods[m]) for m, v in vals.items() if v}


# ── Reporting ─────────────────────────────────────────────────────────

_HOW_TO_READ = """\
HOW TO READ THIS REPORT
  1. DESIGN vs RESPONSE.  Section 1 describes the campaign we ran; sections
     2-6 describe how the cell answered.  A high input rank with a low field
     rank is the target outcome: the sampling was not wasted AND the physics
     is compressible.  They are not supposed to agree.
  2. ENERGY vs TRUNCATION.  Effective rank says how many modes dominate the
     energy; n95/n99 say how many are needed for safe truncation.  eff_rank
     around 4 together with n99 around 28 is one spectrum, not a conflict.
     Choose the surrogate's k from n95/n99 (and from section 3).
  3. MARGINAL vs MULTIVARIATE.  With 24 inputs, a purely linear map still
     gives every output only a moderate correlation with any single anode.
     Nonlinearity is therefore measured by held-out R^2 of a multivariate
     fit (linear vs quadratic), never by per-anode correlations.
  4. MAGNITUDE vs SHAPE.  A CV on a scalar integral measures magnitude only.
     Flow-structure diversity is measured by the POD spectrum, section 2."""


def _fmt_dispersion(name: str, d: dict, indent: str = "  ") -> list[str]:
    o = d["overall"]
    L = [f"{indent}{name:24s} CV={o['cv']:7.2f}%  "
         f"range=[{o['min']:.4e}, {o['max']:.4e}]  n={o['n']}"]
    by = d["by_mode"]
    if len(by) > 1:
        parts = "  ".join(f"{m}={s['cv']:.2f}%" for m, s in sorted(by.items()))
        L.append(f"{indent}{'':24s} by mode: {parts}")
    return L


def build_report(inp, scal, all_pods, all_preds, em, flow, mesh,
                 n_dead, n_bulk, mode_counts) -> str:
    L = ["=" * 70, "COVERAGE REPORT", f"Runs: {n_bulk} bulk + {n_dead} dead-anode",
         "=" * 70, "", _HOW_TO_READ, ""]

    # 1 -------------------------------------------------------------
    L.append("-" * 70)
    L.append("1. INPUT SPACE  (design: did we explore the operating envelope?)")
    L.append("-" * 70)
    evr = inp["evr"]
    L.append(f"  Effective rank : {inp['eff_rank']:.1f} / {inp['n_dof']}"
             f"   (max is {inp['n_dof']}, not {inp['n_dof']+1}: the currents")
    L.append(f"                   sum to a fixed total, which removes one dof)")
    L.append(f"  Dims for 95%   : {inp['dims_95']}")
    L.append(f"  PC1/PC2/PC3    : {evr[0]:.3f} / "
             f"{evr[1] if len(evr) > 1 else 0:.3f} / "
             f"{evr[2] if len(evr) > 2 else 0:.3f}")
    L.append(f"  NN distance    : min={inp['nn_min']:.3f}  "
             f"median={inp['nn_median']:.3f}")
    if inp["nn_min"] < 0.05:
        L.append("  [note] near-duplicate runs present (NN ~ 0). Intentional "
                 "duplicates are fine;")
        L.append("         unintentional ones waste simulation budget.")
    L.append(f"  Campaign mix   : " +
             ", ".join(f"{m}={c}" for m, c in sorted(mode_counts.items())))
    bulk_frac = max(mode_counts.values()) / max(sum(mode_counts.values()), 1)
    if bulk_frac > 0.5:
        L.append(f"  [note] one generator supplies {100*bulk_frac:.0f}% of the runs, "
                 f"so POOLED CVs below")
        L.append(f"         mostly report that generator's spread. Read the "
                 f"per-mode breakdowns.")

    # 2 -------------------------------------------------------------
    L.append("")
    L.append("-" * 70)
    L.append("2. FIELD COMPRESSIBILITY  (response: how many spatial patterns?)")
    L.append("-" * 70)
    for fp, pod in all_pods.items():
        if "error" in pod:
            L.append(f"  {fp}: {pod['error']}")
            continue
        L.append(f"  [{fp}]  {pod['N']} snapshots x {pod['M']} DOFs")
        L.append(f"    Effective rank : {pod['eff_rank']:.1f}    "
                 f"<- how many modes dominate the energy")
        L.append(f"    Modes for 90/95/99% : {pod['n90']} / {pod['n95']} / "
                 f"{pod['n99']}   <- what truncation needs")
        L.append(f"    Top-3 share    : {pod['top3']*100:.1f}%")
        L.append(f"    Fluctuation    : {100*pod['fluct_rel']:.2f}% of the mean "
                 f"field  <- size of the perturbation")
        L.append(f"                     (small values justify training on "
                 f"deltas w.r.t. a reference run)")

    if mesh:
        L.append("")
        L.append("  Mesh motion between runs (the POD is taken at fixed node INDEX,")
        L.append("  and the nodes move because the interface deforms):")
        for label, d in mesh.items():
            L.append(f"    {label:10s} mean {d['mean_mm']:7.3f} mm   "
                     f"max {d['max_mm']:8.3f} mm   "
                     f"({100*d['rel_domain']:.4f}% of the {d['diag_m']:.1f} m domain)")
            if "disp_over_h" in d:
                L.append(f"    {'':10s} mean element size {d['h_elem_mm']:.1f} mm "
                         f"-> displacement is {d['disp_over_h']:.3f} of one element")
        worst = max((d.get("disp_over_h", 0.0) for d in mesh.values()), default=0.0)
        if 0 < worst < 0.1:
            L.append("    -> nodes move a small fraction of an element, so the field is")
            L.append("       resampled far below the scale on which it varies: the modes")
            L.append("       capture the flow changing, not the mesh moving.")
        elif worst >= 0.1:
            L.append("    -> nodes move an appreciable fraction of an element. Part of")
            L.append("       what the modes capture may be resampling rather than flow;")
            L.append("       compare this figure against the field fluctuation above.")

    # 3 -------------------------------------------------------------
    L.append("")
    L.append("-" * 70)
    L.append("3. PREDICTABILITY OF THE POD COEFFICIENTS  (is a NN justified?)")
    L.append("-" * 70)
    L.append("  Each of the three surrogate mappings is fitted separately:")
    L.append("    r2_lin  = a LINEAR map I -> c (24 x k numbers) on unseen runs")
    L.append("    r2_quad = the same plus pairwise products of the currents")
    L.append("    gap     = what a nonlinear model can actually earn")
    L.append("  The headline R^2 is weighted by explained variance, so it is the")
    L.append("  share of the FIELD reproduced.  An unweighted mean over modes")
    L.append("  would give a 0.01%-variance noise mode the same say as mode 0.")

    for fp, pred in all_preds.items():
        L.append("")
        L.append(f"  [{fp}]")
        if "error" in pred:
            L.append(f"    {pred['error']}")
            continue
        L.append(f"    Held-out fit on {pred['n_train']} train / "
                 f"{pred['n_test']} test runs.")
        L.append(f"    linear      R^2 = {pred['r2_field_lin']:.4f}")
        L.append(f"    + quadratic R^2 = {pred['r2_field_quad']:.4f}")
        L.append(f"    nonlinear gap   = "
                 f"{pred['r2_field_quad'] - pred['r2_field_lin']:+.4f}")
        L.append(f"    predictable modes (R^2 >= {pred['r2_thresh']}): "
                 f"{pred['learnable_modes']}   suggested k = "
                 f"{pred['k_suggested']}")

        L.append("")
        L.append(f"    {'mode':>5s} {'var%':>7s} {'r2_lin':>8s} {'r2_quad':>8s} "
                 f"{'gap':>7s}   dominant anode")
        L.append("    " + "-" * 58)
        for i in range(min(20, len(pred["r2_lin"]))):
            L.append(f"    {i:5d} {100*pred['evr'][i]:7.2f} "
                     f"{pred['r2_lin'][i]:8.3f} {pred['r2_quad'][i]:8.3f} "
                     f"{pred['gap'][i]:+7.3f}   "
                     f"{pred['dominant_input'][i] + 1:d}")

        # Restrict to modes that actually carry variance.  The largest gaps in
        # the raw list belong to the noise tail, where the linear fit is poor
        # simply because there is little signal; pointing the reader at those
        # would send them to plot a mode holding 0.1% of the field.
        gaps = np.nan_to_num(pred["gap"], nan=0.0).copy()
        gaps[pred["evr"] < 0.005] = -np.inf        # keep modes above 0.5%
        worst = np.argsort(gaps)[::-1][:3]
        big = [i for i in worst if np.isfinite(gaps[i]) and gaps[i] > 0.05]
        if big:
            L.append("")
            L.append("    Nonlinearity is CONCENTRATED, not spread: the largest")
            L.append("    gaps are modes " +
                     ", ".join(f"{i} ({gaps[i]:+.3f})" for i in big) + ".")
            L.append("    Those are patterns excited by products of anode currents,")
            L.append("    i.e. by interactions between anodes rather than by any")
            L.append("    single anode's deviation. Worth plotting.")

    L.append("")
    if all_preds:
        best_gap = max((p["r2_field_quad"] - p["r2_field_lin"])
                       for p in all_preds.values() if "error" not in p)
        if best_gap < 0.02:
            L.append("  READ: the quadratic terms add almost nothing on any field, so")
            L.append("        these maps are close to linear over this campaign. A")
            L.append("        linear surrogate is the control a network must beat;")
            L.append("        report both before claiming nonlinearity.")
        else:
            L.append("  READ: the quadratic terms add real accuracy, so the map is")
            L.append("        measurably nonlinear. This is the quantitative")
            L.append("        evidence for a nonlinear surrogate.")
    L.append("  NOTE: modes with variance but low R^2 are noise as far as the")
    L.append("        surrogate is concerned; truncating them can only help.")

    # 4 -------------------------------------------------------------
    L.append("")
    L.append("-" * 70)
    L.append("4. SCALAR DIAGNOSTICS")
    L.append("-" * 70)
    pred = scal["pred"]
    if "error" not in pred:
        order = np.argsort(pred["r2_lin"])
        L.append(f"  {'quantity':42s} {'r2_lin':>8s} {'r2_quad':>8s}")
        L.append("  " + "-" * 60)
        for i in order:
            L.append(f"  {pred['names'][i]:42s} {pred['r2_lin'][i]:8.3f} "
                     f"{pred['r2_quad'][i]:8.3f}")
    if scal["unpredictable"]:
        L.append("")
        L.append("  Not predictable from the currents (R^2 < 0.1 either way):")
        L.append(f"    {'quantity':42s} {'best R^2':>9s} {'CV':>9s}")
        for c, rl, rb, cv in scal["unpredictable"]:
            cv_s = f"{cv:8.3f}%" if np.isfinite(cv) else "       --"
            L.append(f"    {c:42s} {rb:9.3f} {cv_s:>9s}")
        L.append("    Read R^2 together with the CV: a near-constant quantity has")
        L.append("    NOTHING to predict, so its R^2 is noise about zero and means")
        L.append("    'this is pinned', not 'this is chaotic'. Only a quantity that")
        L.append("    varies AND resists prediction is genuinely independent of the")
        L.append("    currents (or numerical noise).")
    if scal["low_var"]:
        L.append("")
        L.append(f"  Near-constant across runs (CV < 2%):")
        for name, cv in scal["low_var"][:15]:
            L.append(f"    {name:42s} CV={cv:6.2f}%")
        L.append("    -> carries no information for a surrogate. Check the CV is")
        L.append("       not diluted by a large constant (see section 5).")

    # 5 -------------------------------------------------------------
    L.append("")
    L.append("-" * 70)
    L.append("5. ELECTROMAGNETIC FORCING  (what varies, what is background?)")
    L.append("-" * 70)
    labels = {"lorentz": "Lorentz |jxB|", "B_norm": "induction |B|",
              "j_horiz_frac_alu": "horiz. current (metal)",
              "f_body_total": "total body force"}
    L.append("  CV below is a MAGNITUDE statistic: how much the spatial mean of")
    L.append("  each quantity moved. It cannot see a force that reorganises at")
    L.append("  constant magnitude, so the field fluctuation is reported after it.")
    L.append("")
    for key in ("lorentz", "j_horiz_frac_alu", "B_norm", "f_body_total"):
        if key in em:
            L += _fmt_dispersion(labels[key], em[key])

    fl = em.get("_fluct", {})
    if fl:
        L.append("")
        L.append("  Field fluctuation  ||f_i - f_mean|| / ||f_mean||  (SHAPE, not magnitude):")
        for key, lab in (("lorentz", "Lorentz jxB"), ("B", "induction B")):
            if key in fl and np.isfinite(fl[key]):
                L.append(f"    {lab:20s} {100*fl[key]:8.2f}%")
        L.append("    A large fluctuation next to a small CV means the forcing")
        L.append("    pattern moves while its magnitude does not — which is the")
        L.append("    signal the surrogate has to learn.")
    if "_gravity_dominance" in em:
        r = em["_gravity_dominance"]
        L.append("")
        L.append(f"  |rho*g + jxB| / |jxB| = {r:.0f}x  <- gravity dominance")
        L.append(f"  The total body force is ~{r:.0f} times the Lorentz force, so a")
        L.append(f"  CV computed on it is diluted by that factor and is NOT a")
        L.append(f"  measure of driving-force diversity. rho*g is also potential,")
        L.append(f"  so it is absorbed by the pressure and drives no flow. Quote")
        L.append(f"  the Lorentz row above instead.")
    if "B_norm" in em and em["B_norm"]["overall"]["cv"] < 1.0:
        L.append("")
        L.append("  |B| is nearly invariant, as expected: the background field is")
        L.append("  computed once and reused at a fixed total current, so only the")
        L.append("  interface-induced perturbation varies. This is what justifies")
        L.append("  perturbation (delta) learning -- and it also scopes the")
        L.append("  surrogate to THIS setpoint: change the total current and the")
        L.append("  background field is recomputed.")

    # 6 -------------------------------------------------------------
    L.append("")
    L.append("-" * 70)
    L.append("6. MACROSCOPIC FLOW TOPOLOGY  (magnitude only -- see section 2)")
    L.append("-" * 70)
    for name, d in flow.items():
        L += _fmt_dispersion(name, d)
    L.append("")
    L.append("  These are scalar integrals: they measure how much the magnitude")
    L.append("  moved, not whether the flow reorganised. Combined with section 2,")
    L.append("  small CVs plus a broad POD spectrum is the accurate reading:")
    L.append("  a nearly invariant base flow carrying a spatially rich modulation.")

    # 7 -------------------------------------------------------------
    if scal["convergence"]:
        L.append("")
        L.append("-" * 70)
        L.append("7. CONVERGENCE QUALITY  (numerics, NOT physics)")
        L.append("-" * 70)
        for name, d in scal["convergence"].items():
            if not d:
                continue
            L.append(f"  {name}")
            L.append(f"    median={d['median']:.3e}  p95={d['p95']:.3e}  "
                     f"max={d['max']:.3e}  spans {d['decades']:.1f} decades")
            if "worst_margin" in d:
                L.append(f"    rejection tolerance {d['tol']:.0e} — the WORST run "
                         f"is {d['worst_margin']:.0f}x inside it")
            by = d.get("by_mode", {})
            if len(by) > 1:
                L.append("    median by mode: " + "  ".join(
                    f"{m}={v['median']:.1e}" for m, v in sorted(by.items())))
        L.append("")
        L.append("  Reported on a log scale deliberately: these residuals span")
        L.append("  several decades, and a CV on such a sample is pinned by its")
        L.append("  largest members and says nothing but 'this is a log-scale")
        L.append("  variable'. What matters is the margin against the tolerance.")
        L.append("")
        L.append("  u.n = 0 is imposed exactly at every iteration via a Lagrange")
        L.append("  multiplier, so a nonzero max|u.n| is the pointwise residual of")
        L.append("  a constraint that holds weakly -- a convergence diagnostic.")
        L.append("  Its variation across runs says how tightly each run converged,")
        L.append("  not how stable the interface physically is. Do not read it as")
        L.append("  evidence about interface physics; it IS the right quantity to")
        L.append("  reject runs on, which is what Phase 1 uses it for.")

    return "\n".join(L)


# ── Plotting ──────────────────────────────────────────────────────────
#
# Six panels, one per question.  Panels that used to show per-anode
# correlation heat maps and the Spearman-vs-Pearson scatter were removed with
# the statistics they illustrated (see 2A).

def make_all_plots(inp, pod, all_preds, flow, X_in, out_dir):
    if not HAS_PLT:
        print("matplotlib not available; skipping plots.")
        return

    Xs, _, _ = _standardize(X_in)

    def _proj2(M):
        if M.shape[0] < 2 or M.shape[1] < 2:
            return np.zeros((M.shape[0], 2))
        Mc = M - M.mean(axis=0)
        _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
        return Mc @ Vt[:2].T

    has_pod = "error" not in pod
    preds = {k: v for k, v in (all_preds or {}).items() if "error" not in v}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Did we explore the input space?
    Xp = _proj2(Xs)
    axes[0, 0].scatter(Xp[:, 0], Xp[:, 1], s=10, alpha=0.6)
    axes[0, 0].set_title(f"Input PCA — eff. rank {inp['eff_rank']:.1f}/{inp['n_dof']}")
    axes[0, 0].set_xlabel("PC1"); axes[0, 0].set_ylabel("PC2")
    axes[0, 0].grid(alpha=0.3)

    # (0,1) Are runs distinct?
    axes[0, 1].hist(inp["nn_dists"], bins=30)
    axes[0, 1].axvline(inp["nn_median"], color="k", ls="--",
                       label=f"median={inp['nn_median']:.2f}")
    axes[0, 1].set_title("Nearest-neighbour distance (input space)")
    axes[0, 1].set_xlabel("distance"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (0,2) How compressible is the field?
    if has_pod:
        n = min(50, len(pod["cumevr"]))
        axes[0, 2].plot(range(1, n + 1), pod["cumevr"][:n] * 100, "o-", ms=3)
        for thr in (90, 95, 99):
            axes[0, 2].axhline(thr, color="gray", ls="--", alpha=0.4)
        axes[0, 2].set_title(f"POD cumulative variance — eff. rank "
                             f"{pod['eff_rank']:.1f}, n99={pod['n99']}")
        axes[0, 2].set_xlabel("# modes"); axes[0, 2].set_ylabel("cum. variance %")
        axes[0, 2].grid(alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, "no POD", ha="center", va="center",
                        transform=axes[0, 2].transAxes)

    # (1,0) Where does the energy sit?
    if has_pod:
        nb = min(20, len(pod["evr"]))
        axes[1, 0].bar(range(1, nb + 1), pod["evr"][:nb] * 100)
        axes[1, 0].set_title("POD per-mode variance")
        axes[1, 0].set_xlabel("mode"); axes[1, 0].set_ylabel("variance %")
        axes[1, 0].grid(alpha=0.3)

    # (1,1) THE key panel: is each map linear, and where does the signal end?
    # One colour per field; solid = linear, dashed = with pairwise products.
    # The vertical distance between a pair is the nonlinearity of that field.
    if preds:
        colours = ["C0", "C1", "C2", "C3"]
        for (fp, pr), col in zip(preds.items(), colours):
            short = fp.split("/")[-1] + " (" + fp.split("/")[0].replace("fields_", "") + ")"
            # Show only modes above the variance floor used for truncation:
            # the tail carries no energy, so its R^2 is not informative and
            # would otherwise take up most of the axis.
            n_show = int(np.sum(pr["evr"] >= pr.get("min_evr", 1e-4)))
            n_show = max(n_show, 5)
            idx = np.arange(1, n_show + 1)
            axes[1, 1].plot(idx, pr["r2_lin"][:n_show], "-", color=col, lw=1.5,
                            label=f"{short} lin")
            if not np.all(np.isnan(pr["r2_quad"])):
                axes[1, 1].plot(idx, pr["r2_quad"][:n_show], "--", color=col,
                                lw=1.2, alpha=0.8, label=f"{short} +quad")
            axes[1, 1].axvline(pr["k_suggested"], color=col, ls=":", alpha=0.5)
        thresh = next(iter(preds.values()))["r2_thresh"]
        axes[1, 1].axhline(thresh, color="gray", ls="--", alpha=0.6)
        axes[1, 1].set_ylim(-0.1, 1.05)
        axes[1, 1].set_title("Held-out predictability per POD coefficient\n"
                             "(dotted verticals = suggested k)")
        axes[1, 1].set_xlabel("mode"); axes[1, 1].set_ylabel("R² (unseen runs)")
        axes[1, 1].legend(fontsize=6, ncol=1); axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "not enough runs\nfor a held-out fit",
                        ha="center", va="center", transform=axes[1, 1].transAxes)

    # (1,2) Is the pooled CV just the gaussian generator?
    if flow:
        names = list(flow.keys())[:5]
        modes = sorted({m for n in names for m in flow[n]["by_mode"]})
        width = 0.8 / max(len(modes), 1)
        for i, mode in enumerate(modes):
            vals = [flow[n]["by_mode"].get(mode, {}).get("cv", 0.0) for n in names]
            axes[1, 2].bar(np.arange(len(names)) + i * width, vals,
                           width=width, label=mode)
        axes[1, 2].set_xticks(np.arange(len(names)) + 0.4)
        axes[1, 2].set_xticklabels([n.replace("_", "\n") for n in names],
                                   fontsize=7)
        axes[1, 2].set_ylabel("CV %")
        axes[1, 2].set_title("Flow-magnitude spread by campaign mode")
        axes[1, 2].legend(fontsize=7); axes[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle("Coverage analysis (bulk runs only)", fontsize=14)
    fig.tight_layout()
    path = out_dir / "coverage_full.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Plots saved to {path}")
    plt.close(fig)


# ── Phase 2 driver ────────────────────────────────────────────────────

def phase2_coverage(master_path: Path, out_dir: Path, dead_thresh: float,
                    field_paths: list[str] = None):
    """Read master_dataset.h5 (read-only) and run every coverage analysis."""
    if field_paths is None:
        field_paths = list(DEFAULT_FIELDS)

    master = h5py.File(master_path, "r")

    # Dead-anode runs are counted but excluded from the bulk statistics: a
    # zeroed anode is a different regime and would dominate every dispersion.
    all_X, _, all_names = _collect_scalars(master, exclude_dead=False,
                                           dead_thresh=0)
    n_dead = int(np.any(all_X < dead_thresh, axis=1).sum())

    X, df, run_names = _collect_scalars(master, exclude_dead=True,
                                        dead_thresh=dead_thresh)
    n_bulk = X.shape[0]
    modes_of = _load_modes(out_dir, all_names)
    modes = [modes_of.get(n, "unknown") for n in run_names]
    print(f"\nPhase 2: {n_bulk} bulk + {n_dead} dead-anode runs")

    print("\n--- 1. Input-space coverage ---")
    inp = input_space_coverage(X, modes)
    print(f"  Effective rank : {inp['eff_rank']:.1f} / {inp['n_dof']} "
          f"(max {inp['n_dof']}: the currents sum to a fixed total)")
    print(f"  Dims for 95%   : {inp['dims_95']}")
    print(f"  NN min/median  : {inp['nn_min']:.3f} / {inp['nn_median']:.3f}")

    print("\n--- 2. Field POD ---")
    all_pods = {}
    for fp in field_paths:
        print(f"\n  [{fp}]")
        pod = field_pod(master, fp, True, dead_thresh)
        all_pods[fp] = pod
        if "error" in pod:
            print(f"    {pod['error']}")
        else:
            print(f"    eff.rank={pod['eff_rank']:.1f}  "
                  f"90%->{pod['n90']}  95%->{pod['n95']}  99%->{pod['n99']} modes")
            print(f"    fluctuation about the mean field: "
                  f"{100*pod['fluct_rel']:.2f}%")

    print("\n--- 2b. Mesh motion between runs ---")
    mesh = mesh_motion(master, True, dead_thresh)
    for label, d in mesh.items():
        extra = (f"  = {d['disp_over_h']:.3f} element" if "disp_over_h" in d else "")
        print(f"  {label}: mean {d['mean_mm']:.3f} mm, max {d['max_mm']:.3f} mm{extra}")

    primary_pod, primary_field = {"error": "no POD computed"}, field_paths[0]
    for fp in field_paths:
        if "error" not in all_pods[fp]:
            primary_pod, primary_field = all_pods[fp], fp
            break

    # The three mappings the surrogate has to learn are separate problems and
    # need not be equally linear, so each gets its own fit.
    print("\n--- 3. Predictability of the POD coefficients ---")
    all_preds = {}
    for fp in field_paths:
        if "error" in all_pods[fp]:
            continue
        pr = pod_linear_model(all_pods[fp], master)
        all_preds[fp] = pr
        if "error" in pr:
            print(f"  [{fp}] {pr.get('error')}")
        else:
            print(f"  [{fp}]  field R^2: linear={pr['r2_field_lin']:.4f}  "
                  f"+quadratic={pr['r2_field_quad']:.4f}  "
                  f"gap={pr['r2_field_quad'] - pr['r2_field_lin']:+.4f}  "
                  f"k={pr['k_suggested']}")
    pod_pred = all_preds.get(primary_field, {"error": "no POD computed"})

    print("\n--- 4. Scalar diagnostics ---")
    scal = scalar_diagnostics(X, df, modes)
    print(f"  near-constant: {len(scal['low_var'])}   "
          f"unpredictable: {len(scal['unpredictable'])}")

    print("\n--- 5. Electromagnetic forcing ---")
    em = em_forcing_variance(master, modes_of, True, dead_thresh)
    for k, v in em.items():
        if k.startswith("_"):
            continue
        print(f"  {k}: CV={v['overall']['cv']:.2f}%")
    if "_gravity_dominance" in em:
        print(f"  gravity dominance |rho g + jxB|/|jxB| = "
              f"{em['_gravity_dominance']:.0f}x")

    print("\n--- 6. Flow topology (magnitude) ---")
    flow = flow_topology_variance(master, modes_of, True, dead_thresh)
    for k, v in flow.items():
        print(f"  {k}: CV={v['overall']['cv']:.2f}%")

    report = build_report(inp, scal, all_pods, all_preds, em, flow, mesh,
                          n_dead, n_bulk, inp["by_mode"])
    (out_dir / "coverage_report.txt").write_text(report)
    print(f"\nReport saved to {out_dir / 'coverage_report.txt'}")
    print("\n" + report)

    try:
        df.to_parquet(out_dir / "scalars.parquet")
        print(f"Scalars saved to {out_dir / 'scalars.parquet'}")
    except Exception:
        df.to_csv(out_dir / "scalars.csv")
        print(f"Scalars saved to {out_dir / 'scalars.csv'} (pyarrow not available)")

    make_all_plots(inp, primary_pod, all_preds, flow, X, out_dir)
    master.close()


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="Build the master dataset and analyse campaign coverage.")
    p.add_argument("--dir", required=True,
                   help="Campaign directory containing manifest.csv")
    p.add_argument("--dead-threshold", type=float, default=500.0,
                   help="Anode current below which a run counts as dead-anode")
    p.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS),
                   help="HDF5 paths of the spatial fields to run POD on "
                        "(default: the three surrogate targets)")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Analyse an existing master_dataset.h5 without rebuilding")
    args = p.parse_args()

    out_root = Path(args.dir)

    if not args.skip_phase1:
        master_path, _ = phase1_aggregate(out_root)
    else:
        master_path = out_root / "master_dataset.h5"
        print(f"Skipping Phase 1. Reading existing {master_path} ...")

    phase2_coverage(master_path, out_root, args.dead_threshold, args.fields)

    print("\n" + "=" * 70)
    print("All done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
