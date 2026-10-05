"""Phase 0.1 — predict the field, then evaluate the functional.

Four predictors of each scalar diagnostic, scored on one test split:

  (i)   linear regression of the scalar on the 24 currents
  (ii)  screened hierarchical-ridge quadratic on the currents
  (iii) the functional evaluated on the field of a LINEAR POD model
  (iv)  the functional evaluated on the field of the trained ResFFNN

The point of the comparison is that (i) and (ii) regress the observable, while
(iii) and (iv) regress the *field* and then compute the observable exactly.

Three truths are reported, because they disagree and the disagreement matters:

  T1  the solver's own value, from scalars.csv (its own plane quadrature)
  T2  this file's functional on the true full-resolution field   <- scored against
  T3  this file's functional on the POD-truncated true field     <- truncation floor

Every functional is defined once here and applied identically to the true and the
predicted fields, so the comparison is between predictors and not between
quadrature rules.  Weights are P1 mass-matrix (nodal volume / area), taken from
the reference run so that one fixed functional is applied to every run.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import argparse
import os

import h5py
import numpy as np

MASTER = "../report_ML/master_ml.h5"
SCALARS = "../report_ML/scalars_cluster.csv"
MIDACD_MESH = "../report_ML/midacd_mesh.npz"
CACHE = "data/functionals_T2.npz"   # truth is split-independent: same runs, same Phi
VARIANT = ""          # config suffix; "_normenv" selects the normal-envelope split

_RCOND = 1e-4          # matches compare_baselines; keeps the sum-constraint null
_ALPHAS = [1e-1, 1, 10, 1e2, 1e3, 1e4, 1e6, 1e8]
_N_KEEP = 40


# ----------------------------------------------------------------------
# Quadrature weights
# ----------------------------------------------------------------------
def tet_node_volumes(nodes, elems):
    """P1 lumped mass: each tet gives |det|/6/4 to each of its four nodes."""
    p = nodes[elems]                                    # [Ne, 4, 3]
    d = np.linalg.det(np.stack([p[:, 1] - p[:, 0],
                                p[:, 2] - p[:, 0],
                                p[:, 3] - p[:, 0]], axis=1))
    vol = np.abs(d) / 6.0
    w = np.zeros(len(nodes))
    np.add.at(w, elems.ravel(), np.repeat(vol / 4.0, 4))
    return w


def tri_node_areas(nodes, elems):
    """P1 lumped mass on a surface: each triangle gives area/3 to each vertex."""
    p = nodes[elems]                                    # [Nt, 3, 3]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0],
                                         p[:, 2] - p[:, 0]), axis=1)
    w = np.zeros(len(nodes))
    np.add.at(w, elems.ravel(), np.repeat(area / 3.0, 3))
    return w


def build_weights(target, h5_path):
    """Fixed quadrature weights for one target, from the reference run."""
    with h5py.File(h5_path, "r") as f:
        ref = f["meta"].attrs["reference_run"]
        rec = f["reconstruction"]
        fluid_ids = rec["fluid_node_ids"][:] if "fluid_node_ids" in rec else None
        fluid_el = rec["fluid_elems"][:] if "fluid_elems" in rec else None
    with h5py.File(MASTER, "r") as m:
        g = m[ref]
        if target == "full3d":
            w = tet_node_volumes(g["mesh/cuveb_nodes"][:][fluid_ids], fluid_el)
        elif target == "interface":
            w = tri_node_areas(g["mesh/interface_nodes"][:],
                               g["mesh/interface_elems"][:])
        elif target == "midacd":
            d = np.load(MIDACD_MESH)
            w = tri_node_areas(d["nodes"], d["elems"])
        else:
            raise ValueError(target)
    return w / w.sum(), fluid_ids


# ----------------------------------------------------------------------
# Functionals.  Each takes [N, M, ncomp] and the weights, returns [N].
# ----------------------------------------------------------------------
def _mag(F):
    return np.linalg.norm(F, axis=2) if F.shape[2] > 1 else np.abs(F[:, :, 0])


VELOCITY_FUNCTIONALS = {
    "u_mean":  ("L1",   lambda F, w: (_mag(F) * w).sum(1)),
    "u_rms":   ("L2",   lambda F, w: np.sqrt(((F ** 2).sum(2) * w).sum(1))),
    "ke":      ("L2",   lambda F, w: 0.5 * ((F ** 2).sum(2) * w).sum(1)),
    "u_max":   ("Linf", lambda F, w: _mag(F).max(1)),
}

INTERFACE_FUNCTIONALS = {
    "eta_mean": ("L1",   lambda F, w: (F[:, :, 0] * w).sum(1)),
    "eta_std":  ("L2",   lambda F, w: np.sqrt(
        ((F[:, :, 0] - (F[:, :, 0] * w).sum(1, keepdims=True)) ** 2 * w).sum(1))),
    "eta_max":  ("Linf", lambda F, w: F[:, :, 0].max(1)),
    "eta_min":  ("Linf", lambda F, w: F[:, :, 0].min(1)),
    "eta_ptp":  ("Linf", lambda F, w: np.ptp(F[:, :, 0], axis=1)),
}


def functionals_for(target):
    return INTERFACE_FUNCTIONALS if target == "interface" else VELOCITY_FUNCTIONALS


# ----------------------------------------------------------------------
# T2: the functional on the TRUE full-resolution field, for every run
# ----------------------------------------------------------------------
def compute_T2(target, run_ids, w, fluid_ids, ncomp):
    fns = functionals_for(target)
    out = {k: np.empty(len(run_ids)) for k in fns}
    with h5py.File(MASTER, "r") as m:
        for i, r in enumerate(run_ids):
            g = m[r]
            if target == "full3d":
                F = g["fields_full/vitesse"][:][fluid_ids][None]
            elif target == "midacd":
                F = g["fields_midacd/vitesse"][:][None]
            else:                                   # absolute interface elevation
                F = g["mesh/interface_nodes"][:, 2][None, :, None]
            for k, (_, fn) in fns.items():
                out[k][i] = fn(F.astype(np.float64), w)[0]
    return out


# ----------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------
def _pairs(Z):
    iu = np.triu_indices(Z.shape[1])
    return (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]


def r2(y, yh):
    return 1.0 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_linear_scalar(Ptr, ytr, Pte):
    D = lambda P: np.c_[np.ones(len(P)), P]
    b, *_ = np.linalg.lstsq(D(Ptr), ytr, rcond=_RCOND)
    return D(Pte) @ b


def fit_quadratic_scalar(Ptr, ytr, Pte, seed=0):
    """Screened, hierarchical ridge: penalty on the quadratic block only, so
    alpha -> inf recovers the linear fit and the comparison stays monotone."""
    rng = np.random.default_rng(seed)
    Qtr, Qte = _pairs(Ptr), _pairs(Pte)
    sd = Qtr.std(0); sd[sd == 0] = 1.0
    Qtr, Qte = Qtr / sd, Qte / sd

    # screen products against the residual of the linear fit, not the target
    res = ytr - fit_linear_scalar(Ptr, ytr, Ptr)
    keep = np.argsort(-np.abs(Qtr.T @ (res - res.mean())))[:_N_KEEP]

    best = (-np.inf, None, None)
    idx = rng.permutation(len(Ptr)); cut = int(0.8 * len(idx))
    a, b = idx[:cut], idx[cut:]
    for cols in (keep, np.arange(Qtr.shape[1])):
        Dtr = np.c_[np.ones(len(Ptr)), Ptr, Qtr[:, cols]]
        for al in _ALPHAS:
            pen = np.r_[1e-6, np.full(Ptr.shape[1], 1e-6),
                        np.full(len(cols), al)]
            wgt = np.linalg.solve(Dtr[a].T @ Dtr[a] + np.diag(pen),
                                  Dtr[a].T @ ytr[a])
            sc = r2(ytr[b], Dtr[b] @ wgt)
            if sc > best[0]:
                best = (sc, cols, al)
    _, cols, al = best
    Dtr = np.c_[np.ones(len(Ptr)), Ptr, Qtr[:, cols]]
    Dte = np.c_[np.ones(len(Pte)), Pte, Qte[:, cols]]
    pen = np.r_[1e-6, np.full(Ptr.shape[1], 1e-6), np.full(len(cols), al)]
    wgt = np.linalg.solve(Dtr.T @ Dtr + np.diag(pen), Dtr.T @ ytr)
    return Dte @ wgt


# ----------------------------------------------------------------------
def main(target, use_cache=True):
    import yaml
    cfg = yaml.safe_load(open("config_alucell_%s%s.yaml" % (target, VARIANT)))
    h5p = cfg["data"]["h5_path"]
    f = h5py.File(h5p, "r")
    ncomp = int(f["meta"].attrs["field_shape"][1])
    w, fluid_ids = build_weights(target, h5p)

    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    ids = {s: dec(f[s]["run_id"][:]) for s in ("train", "test")}
    P = {s: f[s]["P"][:].astype(np.float64) for s in ("train", "test")}
    U = {s: f[s]["U"][:].astype(np.float64) for s in ("train", "test")}

    rec = f["reconstruction"]
    V = rec["V"][:].astype(np.float64)
    mu = rec["u_pod_mean"][:].astype(np.float64)
    uref = rec["u_ref"][:].astype(np.float64)
    reshape = lambda X: X.reshape(len(X), -1, ncomp)
    recon = lambda C: reshape(C @ V.T + mu + uref)

    # ---- T2, cached (reading the master is the slow part) ----
    # The truth does not depend on the split -- it is a functional of the solver's
    # field -- so the cache is keyed by run id and is shared between splits.  Only
    # runs not already present are computed.
    ck = "%s_%s" % (CACHE.replace(".npz", ""), target)
    want = ids["train"] + ids["test"]
    have, cached = {}, None
    if use_cache and os.path.exists(ck + ".npz"):
        z = np.load(ck + ".npz", allow_pickle=True)
        cached = {k: z[k] for k in z.files if k != "_ids"}
        have = {r: i for i, r in enumerate(list(z["_ids"]))}
    missing = [r for r in want if r not in have]
    if missing:
        print("computing the truth on %d runs not in the cache ..." % len(missing))
        fresh = compute_T2(target, missing, w, fluid_ids, ncomp)
        if cached is None:
            cached, have = fresh, {r: i for i, r in enumerate(missing)}
        else:
            base = len(have)
            for k in cached:
                cached[k] = np.concatenate([cached[k], fresh[k]])
            for i, r in enumerate(missing):
                have[r] = base + i
        np.savez(ck + ".npz",
                 _ids=np.array([r for r, _ in sorted(have.items(), key=lambda kv: kv[1])]),
                 **cached)
    take = np.array([have[r] for r in want])
    T2 = {k: v[take] for k, v in cached.items()}
    ntr = len(ids["train"])
    T2tr = {k: v[:ntr] for k, v in T2.items()}
    T2te = {k: v[ntr:] for k, v in T2.items()}

    # ---- T3 and the two field predictors ----
    T3te = reshape(U["test"] @ V.T + mu + uref)
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

    fns = functionals_for(target)
    print("\n=== %s ===  (truth = T2, functional on the true full-res field)" % target)
    print("%-9s %-5s | %8s %8s | %8s %8s | %8s"
          % ("scalar", "cls", "(i)lin", "(ii)quad", "(iii)fld", "(iv)net", "T3 floor"))
    print("-" * 72)
    rows = []
    for k, (cls, fn) in fns.items():
        ytr, yte = T2tr[k], T2te[k]
        p1 = fit_linear_scalar(P["train"], ytr, P["test"])
        p2 = fit_quadratic_scalar(P["train"], ytr, P["test"])
        print("%-9s %-5s | %8.3f %8.3f | %8.3f %8.3f | %8.3f"
              % (k, cls, r2(yte, p1), r2(yte, p2),
                 r2(yte, fn(F_lin, w)), r2(yte, fn(F_net, w)),
                 r2(yte, fn(T3te, w))))
        rows.append((k, cls, yte, p1, p2, fn(F_lin, w), fn(F_net, w)))

    # The pooled score hides which operating conditions the result rests on.  The
    # rare regimes are not outliers to be tolerated: a weak or dead anode is a cell
    # around an anode replacement, and a single anode at a box limit is a localised
    # disturbance.  Those are the conditions an operator most wants a model for, so
    # whether field-then-functional survives there is the question that matters.
    te_modes = np.array(dec(f["test"]["modes"][:]))
    order = ["gaussian", "gradient", "cluster", "single", "weak", "dead"]
    present = [m for m in order if (te_modes == m).sum() >= 4]
    print("\n--- the same comparison, per operating condition ---")
    print("%-9s %-9s %4s | %8s %8s | %8s %8s"
          % ("scalar", "regime", "n", "(i)lin", "(ii)quad", "(iii)fld", "(iv)net"))
    print("-" * 66)
    for k, cls, yte, p1, p2, fl, fnet in rows:
        for m in present:
            j = te_modes == m
            if np.var(yte[j]) <= 0:
                continue
            print("%-9s %-9s %4d | %8.3f %8.3f | %8.3f %8.3f"
                  % (k, m, j.sum(), r2(yte[j], p1[j]), r2(yte[j], p2[j]),
                     r2(yte[j], fl[j]), r2(yte[j], fnet[j])))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["midacd", "full3d", "interface"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--variant", default="",
                    help="config suffix, e.g. _normenv for the normal-envelope split")
    a = ap.parse_args()
    VARIANT = a.variant                       # module-level; main() reads it
    globals()["VARIANT"] = a.variant
    main(a.target, use_cache=not a.no_cache)
