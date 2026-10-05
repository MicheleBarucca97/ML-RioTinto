"""Phase 0.6 — is the inverse problem well posed?

Minimise a scalar functional of the surrogate field over the anode currents,
subject to the physical constraints

    sum_k I_k = I_tot = 490 kA ,      I_min <= I_k <= I_max ,

from many random starts.  If the objective values agree but the solutions do
not, the problem has a null space and every later optimum must be regularised
and reported with its null-space dimension.

In the normalised coordinates the dataset uses, p_k = I_k / I_mean - 1, the
constraint becomes sum_k p_k = 0 with box [I_min/I_mean - 1, I_max/I_mean - 1].

The gradient comes from torch autograd through the network and the POD
reconstruction, so the optimiser sees the exact surrogate derivative.
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
import torch
import yaml
from scipy.optimize import minimize

from utils import build_model, resolve_checkpoint

I_TOTAL, N_ANODES = 490_000.0, 24
I_MEAN = I_TOTAL / N_ANODES
I_MIN, I_MAX = 16_400.0, 24_400.0
P_LO, P_HI = I_MIN / I_MEAN - 1.0, I_MAX / I_MEAN - 1.0


OBJECTIVES = {
    # smooth L2 functionals; both are minimised
    "eta_std": lambda F, w: torch.sqrt(
        ((F[:, :, 0] - (F[:, :, 0] * w).sum(1, keepdim=True)) ** 2 * w).sum(1)),
    "ke":      lambda F, w: 0.5 * ((F ** 2).sum(2) * w).sum(1),
}


def load_surrogate(target):
    cfg = yaml.safe_load(open("config_alucell_%s.yaml" % target))
    f = h5py.File(cfg["data"]["h5_path"], "r")
    ncomp = int(f["meta"].attrs["field_shape"][1])
    rec = f["reconstruction"]
    V = torch.tensor(rec["V"][:], dtype=torch.float64)
    mu = torch.tensor(rec["u_pod_mean"][:], dtype=torch.float64)
    uref = torch.tensor(rec["u_ref"][:], dtype=torch.float64)
    m = build_model(cfg)
    m.load_state_dict(torch.load(resolve_checkpoint(cfg), map_location="cpu",
                                 weights_only=True))
    m.eval().double()
    Ptr = f["train/P"][:].astype(np.float64)
    return m, V, mu, uref, ncomp, Ptr


def main(target, objective, n_starts, seed):
    from field_functionals import build_weights
    cfg = yaml.safe_load(open("config_alucell_%s.yaml" % target))
    w_np, _ = build_weights(target, cfg["data"]["h5_path"])
    w = torch.tensor(w_np, dtype=torch.float64)
    m, V, mu, uref, ncomp, Ptr = load_surrogate(target)
    fn = OBJECTIVES[objective]

    def obj_and_grad(p):
        t = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        c = m(t[None], None)          # model cast to double in load_surrogate
        F = (c @ V.T + mu + uref).reshape(1, -1, ncomp)
        v = fn(F, w)[0]
        v.backward()
        return float(v.item()), t.grad.numpy().copy()

    cons = [{"type": "eq",
             "fun": lambda p: p.sum(),
             "jac": lambda p: np.ones_like(p)}]
    bnds = [(P_LO, P_HI)] * N_ANODES

    rng = np.random.default_rng(seed)
    sols, vals = [], []
    for _ in range(n_starts):
        p0 = rng.uniform(P_LO, P_HI, N_ANODES)
        p0 -= p0.mean()                              # onto the hyperplane
        r = minimize(obj_and_grad, p0, jac=True, method="SLSQP",
                     bounds=bnds, constraints=cons,
                     options={"maxiter": 400, "ftol": 1e-12})
        sols.append(r.x); vals.append(r.fun)
    S = np.array(sols); vals = np.array(vals)

    I = I_MEAN * (1.0 + S)                            # back to amperes
    D = np.linalg.norm(I[:, None, :] - I[None, :, :], axis=2) / np.sqrt(N_ANODES)
    iu = np.triu_indices(n_starts, 1)

    print("\n=== %s / minimise %s — %d random starts ===" % (target, objective, n_starts))
    print("  objective   best %.6e   worst %.6e   spread %.3f%% of best"
          % (vals.min(), vals.max(), 100 * (vals.max() - vals.min()) / abs(vals.min())))
    print("  constraint  max |sum I - I_tot| = %.2e A" % np.abs(I.sum(1) - I_TOTAL).max())
    print("  bounds      min %.1f A   max %.1f A   (limits %.0f / %.0f)"
          % (I.min(), I.max(), I_MIN, I_MAX))
    print("  solutions   pairwise RMS distance: median %.0f A/anode, max %.0f A/anode"
          % (np.median(D[iu]), D[iu].max()))
    print("  per-anode spread across starts: median %.0f A, max %.0f A"
          % (np.median(I.std(0)), I.std(0).max()))
    box = I_MAX - I_MIN
    spread_obj = (vals.max() - vals.min()) / abs(vals.min())
    spread_sol = np.median(D[iu]) / box
    print("\n  VERDICT: objective spread %.3f%%, solution spread %.1f%% of the"
          " box (%.0f A)." % (100 * spread_obj, 100 * spread_sol, box))
    if spread_sol < 0.02:
        print("  => WELL POSED: the optimum is essentially unique.")
    elif spread_obj < 0.01:
        print("  => NULL SPACE: the objective agrees to <1% while the currents")
        print("     differ materially. Regularise (Tikhonov / restrict to the")
        print("     leading right singular vectors of A) and report the")
        print("     null-space dimension beside the optimum.")
    else:
        print("  => MULTI-MODAL: distinct local minima with materially different")
        print("     currents AND different objective values. Multi-start is")
        print("     mandatory; report the spread, not a single optimum.")
    return S, vals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="interface",
                    choices=["midacd", "full3d", "interface"])
    ap.add_argument("--objective", default="eta_std", choices=list(OBJECTIVES))
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.target, a.objective, a.starts, a.seed)
