"""Is the multi-modality of Section 5 a property of the cell, or of the network?

ke on an AFFINE field model is a convex quadratic in p, so on a polytope it has a
unique minimiser and multi-start must agree.  The surrogate used in Section 5 is the
network, which is not affine.  Running the identical probe on the linear field model
separates the two.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import sys, numpy as np, h5py
sys.path.insert(0, "/home/barucca/ML-RioTinto")
from scipy.optimize import minimize
from field_functionals import build_weights

I_MEAN, I_MIN, I_MAX = 490000/24, 16400.0, 24400.0
P_LO, P_HI = I_MIN/I_MEAN-1, I_MAX/I_MEAN-1
_RCOND = 1e-4

def probe(target, n_starts=20, seed=0):
    h5 = "/home/barucca/ML-RioTinto/data/%s_pod_delta.h5" % target
    f = h5py.File(h5, "r"); rec = f["reconstruction"]
    V = rec["V"][:].astype(np.float64); mu = rec["u_pod_mean"][:].astype(np.float64)
    ur = rec["u_ref"][:].astype(np.float64)
    A = np.linalg.lstsq(f["train"]["P"][:].astype(np.float64),
                        f["train"]["U"][:].astype(np.float64), rcond=_RCOND)[0]
    w, _ = build_weights(target, h5)
    nc = 1 if target == "interface" else 3
    W = np.repeat(w, nc) if nc == 3 else w
    base = mu + ur

    def obj_and_grad(p):                      # ke on the affine field model
        u = base + (p @ A) @ V.T
        val = float((W * u * u).sum())
        g = 2.0 * ((W * u) @ V) @ A.T
        return val, g

    cons = [{"type": "eq", "fun": lambda p: p.sum(), "jac": lambda p: np.ones_like(p)}]
    bnds = [(P_LO, P_HI)] * 24
    rng = np.random.default_rng(seed)
    S, vals = [], []
    for _ in range(n_starts):
        p0 = rng.uniform(P_LO, P_HI, 24); p0 -= p0.mean()
        r = minimize(obj_and_grad, p0, jac=True, method="SLSQP",
                     bounds=bnds, constraints=cons, options={"maxiter": 400, "ftol": 1e-14})
        S.append(r.x); vals.append(r.fun)
    S = np.array(S); vals = np.array(vals)
    I = I_MEAN * (1.0 + S)
    D = np.linalg.norm(I[:, None, :] - I[None, :, :], axis=2) / np.sqrt(24)
    iu = np.triu_indices(n_starts, 1)
    so = (vals.max() - vals.min()) / abs(vals.min())
    ss = np.median(D[iu]) / (I_MAX - I_MIN)
    print("%-10s  objective spread %8.4f%%   solution spread %6.2f%%"
          % (target, 100 * so, 100 * ss))

print("ke / eta_std^2 on the LINEAR field model (convex; unique minimiser expected)")
for t in ("full3d", "midacd", "interface"):
    probe(t)
print("\nSection 5, same objectives on the NETWORK:")
print("full3d      objective spread   0.0010%%   solution spread   0.30%%")
print("midacd      objective spread   7.1700%%   solution spread  10.70%%")
print("interface   objective spread   5.3000%%   solution spread  36.70%%")
