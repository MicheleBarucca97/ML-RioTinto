"""Phase 0.1, Tier B: the five diagnostic plane integrals.

Replicates `compute_plane_integrals` of `ml_export.cpp` in vectorised numpy, so
that the same functional can be applied to a predicted field as the solver
applies to its own.  For each tetrahedron cut by the plane the routine forms the
intersection polygon, fans it into triangles and integrates

    ke              = A * (|v0|^2+|v1|^2+|v2|^2 + v0.v1 + v0.v2 + v1.v2) / 12
    net_flux        = A * (vn01 + vn12 + vn02) / 3          vn_ij = (v_i^a+v_j^a)/2
    half_abs_flux   = A * (|vn01|+|vn12|+|vn02|) / 6

exactly as the C++ does, where `a` is the slice axis.

Plane positions, from `ml_export.cpp` and CNG2's `_geometrical_params.mac`
(metal_height 0.15, ACD 0.032, n_blocs 24, bloc_width 0.715, bloc_length 1.75):

    mid_acd   z = zp1 - ACD          = 0.150      (confirmed: scalars fiber_oz)
    plane_X   x = +0.98*geom_x_zm1/1.5 = +5.6056
    plane_-X  x = -5.6056
    plane_Y   y = +0.98*geom_y_zm1/1.5 = +1.14333
    plane_-Y  y = -1.14333

The mesh geometry is held at the reference run for every evaluation, so one
fixed functional is applied to the true and the predicted fields alike --- the
same convention as the mass weights of `field_functionals.py`.  The solver
instead uses each run's own deformed mesh, so T1 and T2 differ by that amount;
the comparison is reported as validation.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import argparse

import h5py
import numpy as np
import yaml

MASTER = "../report_ML/master_ml.h5"
_RCOND = 1e-4

METAL_H, ACD = 0.15, 0.032
GX0 = 0.98 * (24 * 0.715 / 2) / 1.5
GY0 = 0.98 * (3.5 / 2) / 1.5
PLANES = {                      # name -> (axis, value)
    "mid_acd":       (2, METAL_H + ACD - ACD),
    "plane_X":       (0, +GX0),
    "plane_minus_X": (0, -GX0),
    "plane_Y":       (1, +GY0),
    "plane_minus_Y": (1, -GY0),
}
EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def plane_integrals(coords, elems, Vfield, axis, cval, chunk=200_000):
    """Return (ke, net_flux, half_abs_flux) for one field on one plane.

    coords : [Nn,3] node positions (fixed)      elems : [Ne,4] 0-indexed
    Vfield : [Nn,3] nodal velocity
    """
    ca = coords[:, axis][elems]                                  # [Ne,4]
    hit = (ca.min(1) <= cval) & (ca.max(1) >= cval)
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return 0.0, 0.0, 0.0
    ke = nf = ha = 0.0
    for s in range(0, idx.size, chunk):
        e = elems[idx[s:s + chunk]]                              # [M,4]
        M = len(e)
        P = coords[e]                                            # [M,4,3]
        Vv = Vfield[e]                                           # [M,4,3]
        d = P[:, :, axis] - cval                                 # [M,4]
        i0, i1 = EDGES[:, 0], EDGES[:, 1]
        da, db = d[:, i0], d[:, i1]                              # [M,6]
        den = db - da
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(den) > 1e-30, -da / den, 0.0)
        good = (np.abs(den) > 1e-30) & (t >= 0.0) & (t <= 1.0)
        tt = t[:, :, None]
        pts = P[:, i0] + tt * (P[:, i1] - P[:, i0])              # [M,6,3]
        vls = Vv[:, i0] + tt * (Vv[:, i1] - Vv[:, i0])

        # drop duplicates (a vertex lying on the plane is found by two edges)
        dist = np.linalg.norm(pts[:, :, None, :] - pts[:, None, :, :], axis=3)
        dup = (dist < 1e-8) & np.triu(np.ones((6, 6), bool), 1)[None]
        good &= ~(dup & good[:, :, None]).any(1)

        nv = good.sum(1)
        order = np.argsort(~good, axis=1, kind="stable")         # valid first
        pts = np.take_along_axis(pts, order[:, :, None], 1)[:, :4]
        vls = np.take_along_axis(vls, order[:, :, None], 1)[:, :4]

        for npts in (3, 4):
            sel = nv == npts
            if not sel.any():
                continue
            p, v = pts[sel], vls[sel]
            if npts == 4:                                        # radial sort
                u_ax = 1 if axis == 0 else (2 if axis == 1 else 0)
                v_ax = 2 if axis == 0 else (0 if axis == 1 else 1)
                cu = p[:, :, u_ax].mean(1, keepdims=True)
                cv = p[:, :, v_ax].mean(1, keepdims=True)
                ang = np.arctan2(p[:, :, v_ax] - cv, p[:, :, u_ax] - cu)
                o = np.argsort(ang, axis=1)
                p = np.take_along_axis(p, o[:, :, None], 1)
                v = np.take_along_axis(v, o[:, :, None], 1)
                tris = [(0, 1, 2), (0, 2, 3)]
            else:
                tris = [(0, 1, 2)]
            for a_, b_, c_ in tris:
                p0, p1, p2 = p[:, a_], p[:, b_], p[:, c_]
                v0, v1, v2 = v[:, a_], v[:, b_], v[:, c_]
                A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
                dot = lambda x, y: (x * y).sum(1)
                ke += float((A * (dot(v0, v0) + dot(v1, v1) + dot(v2, v2)
                                  + dot(v0, v1) + dot(v0, v2) + dot(v1, v2)) / 12.0).sum())
                m01 = 0.5 * (v0[:, axis] + v1[:, axis])
                m12 = 0.5 * (v1[:, axis] + v2[:, axis])
                m02 = 0.5 * (v0[:, axis] + v2[:, axis])
                nf += float((A * (m01 + m12 + m02) / 3.0).sum())
                ha += float((A * 0.5 * (np.abs(m01) + np.abs(m12)
                                        + np.abs(m02)) / 3.0).sum())
    return ke, nf, ha


def _pairs(Z):
    iu = np.triu_indices(Z.shape[1])
    return (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]


def r2(y, yh):
    return 1.0 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_lin(A, y, B):
    b, *_ = np.linalg.lstsq(np.c_[np.ones(len(A)), A], y, rcond=_RCOND)
    return np.c_[np.ones(len(B)), B] @ b


def fit_quad(A, y, B, seed=0):
    rng = np.random.default_rng(seed)
    Qa, Qb = _pairs(A), _pairs(B)
    sd = Qa.std(0); sd[sd == 0] = 1.0
    Qa, Qb = Qa / sd, Qb / sd
    keep = np.argsort(-np.abs(Qa.T @ (y - fit_lin(A, y, A))))[:40]
    ii = rng.permutation(len(A)); cut = int(.8 * len(ii))
    p, q = ii[:cut], ii[cut:]
    best = (-np.inf, None, None)
    for cols in (keep, np.arange(Qa.shape[1])):
        Da = np.c_[np.ones(len(A)), A, Qa[:, cols]]
        for al in [1e-1, 1, 10, 1e2, 1e3, 1e4, 1e6, 1e8]:
            pen = np.r_[1e-6, np.full(A.shape[1], 1e-6), np.full(len(cols), al)]
            W = np.linalg.solve(Da[p].T @ Da[p] + np.diag(pen), Da[p].T @ y[p])
            sc = r2(y[q], Da[q] @ W)
            if sc > best[0]:
                best = (sc, cols, al)
    _, cols, al = best
    Da = np.c_[np.ones(len(A)), A, Qa[:, cols]]
    Db = np.c_[np.ones(len(B)), B, Qb[:, cols]]
    pen = np.r_[1e-6, np.full(A.shape[1], 1e-6), np.full(len(cols), al)]
    W = np.linalg.solve(Da.T @ Da + np.diag(pen), Da.T @ y)
    return Db @ W


def main(cache_only=False):
    import os
    cfg = yaml.safe_load(open("configs/config_alucell_full3d.yaml"))
    f = h5py.File(cfg["data"]["h5_path"], "r")
    rec = f["reconstruction"]
    coords = f["x_grid"][:].astype(np.float64)
    elems = rec["fluid_elems"][:]
    V = rec["V"][:].astype(np.float64)
    mu = rec["u_pod_mean"][:].astype(np.float64)
    uref = rec["u_ref"][:].astype(np.float64)
    fid = rec["fluid_node_ids"][:]
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    ids = {s: dec(f[s]["run_id"][:]) for s in ("train", "test")}
    P = {s: f[s]["P"][:].astype(np.float64) for s in ("train", "test")}
    U = {s: f[s]["U"][:].astype(np.float64) for s in ("train", "test")}
    keys = [(p, q) for p in PLANES for q in ("ke", "net_flux", "half_abs_flux")]

    CACHE = "data/plane_T2.npz"
    allids = ids["train"] + ids["test"]
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        assert list(z["_ids"]) == allids, "cache stale"
        T2 = {k: z["%s|%s" % k] for k in keys}
        print("loaded cached T2")
    else:
        T2 = {k: np.empty(len(allids)) for k in keys}
        with h5py.File(MASTER, "r") as M:
            for i, r in enumerate(allids):
                vf = M[r]["fields_full/vitesse"][:][fid].astype(np.float64)
                for pn, (ax, cv) in PLANES.items():
                    ke, nf, ha = plane_integrals(coords, elems, vf, ax, cv)
                    T2[(pn, "ke")][i] = ke
                    T2[(pn, "net_flux")][i] = nf
                    T2[(pn, "half_abs_flux")][i] = ha
                if i % 100 == 0:
                    print("  %4d/%d" % (i, len(allids)), flush=True)
        np.savez(CACHE, _ids=np.array(allids),
                 **{"%s|%s" % k: v for k, v in T2.items()})
        print("cached T2")
    if cache_only:
        return
    ntr = len(ids["train"])
    reshape = lambda X: X.reshape(len(X), -1, 3)
    recon = lambda C: reshape(C @ V.T + mu + uref)

    beta, *_ = np.linalg.lstsq(P["train"], U["train"], rcond=_RCOND)
    F_lin = recon(P["test"] @ beta)
    import torch
    from utils import build_model, resolve_checkpoint
    m = build_model(cfg)
    m.load_state_dict(torch.load(resolve_checkpoint(cfg), map_location="cpu",
                                 weights_only=True))
    m.eval()
    with torch.no_grad():
        C = m(torch.tensor(P["test"], dtype=torch.float32), None).numpy()
    F_net = recon(C.astype(np.float64))
    F_t3 = reshape(U["test"] @ V.T + mu + uref)

    def evaluate(F):
        out = {k: np.empty(len(F)) for k in keys}
        for i in range(len(F)):
            for pn, (ax, cv) in PLANES.items():
                ke, nf, ha = plane_integrals(coords, elems, F[i], ax, cv)
                out[(pn, "ke")][i] = ke
                out[(pn, "net_flux")][i] = nf
                out[(pn, "half_abs_flux")][i] = ha
        return out
    print("evaluating predicted fields ...")
    E_lin, E_net, E_t3 = evaluate(F_lin), evaluate(F_net), evaluate(F_t3)

    print("\n=== Tier B: the five diagnostic planes ===")
    print("%-14s %-14s %-5s | %8s %8s | %8s %8s | %8s"
          % ("plane", "quantity", "cls", "(i)lin", "(ii)quad",
             "(iii)fld", "(iv)net", "T3 floor"))
    print("-" * 86)
    CLS = {"ke": "L2", "net_flux": "L1s", "half_abs_flux": "L1"}
    for k in keys:
        y = T2[k]
        ytr, yte = y[:ntr], y[ntr:]
        print("%-14s %-14s %-5s | %8.3f %8.3f | %8.3f %8.3f | %8.3f"
              % (k[0], k[1], CLS[k[1]],
                 r2(yte, fit_lin(P["train"], ytr, P["test"])),
                 r2(yte, fit_quad(P["train"], ytr, P["test"])),
                 r2(yte, E_lin[k]), r2(yte, E_net[k]), r2(yte, E_t3[k])))
    print("  cls: L1s = signed linear functional (net flux), L1 = absolute, L2 = quadratic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args()
    main(a.cache_only)
