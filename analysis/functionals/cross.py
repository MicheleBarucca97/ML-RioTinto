"""Section 2.2: size the three terms of Phi = ||u_ref||^2 + 2<u_ref,du> + ||du||^2.

Protocol, stated because the numbers depend on all three choices:
  runs      all 1594 accepted runs (train + val + test), no split -- nothing is fitted
  field     the TRUE field from the master, not its POD reconstruction
  form      each target's OWN quadratic functional: ||u||^2_w for the velocities
            (that is u_rms^2, and ke up to rho/2), and the weighted variance
            ||h - <h>_w||^2_w for the interface (that is eta_std^2)
  weighting the mass-matrix weights of the functional itself, normalised to sum 1,
            so that this diagnostic is computed with the same inner product as the
            Phi it is explaining.  The unweighted Euclidean version is printed
            beside it because it is not the same number: on the interface the two
            differ by a factor 2.4.

Writing du = eps ||u_ref|| nhat and cos = <nhat, u_ref>/||u_ref||, the ratio in the
last column is exactly Var(eps^2) / (4 Var(eps cos)), which is how it is checked.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import h5py
import numpy as np

from field_functionals import build_weights

MASTER = "../report_ML/master_ml.h5"
FIELD = {"full3d": ("fields_full/vitesse", 3),
         "midacd": ("fields_midacd/vitesse", 3),
         "interface": ("fields_interface/h", 1)}


def terms(du, uref, w):
    """(cross, quad) under the inner product with weights w."""
    return 2.0 * (du * (w * uref)).sum(1), ((du ** 2) * w).sum(1)


def centre(F, w, nc):
    """Subtract the weighted mean, for a target whose functional is a variance.

    eta_std is the area-weighted standard deviation about each run's OWN weighted
    mean, so the quadratic form behind it is ||h - <h>_w||^2_w and not ||h||^2_w.
    The expansion of Proposition 1 still applies, because centring is linear, but
    it must be applied to the centred field -- which also makes the choice of
    height datum irrelevant, the constant dropping out.
    """
    if nc != 1:
        return F
    return F - (F * w).sum(-1, keepdims=True)


def main():
    print("%-10s %11s %11s %11s %9s %9s %9s" % (
        "target", "|u_ref|^2", "sd(linear)", "sd(quad)", "q/l mass", "q/l unw", "lin share"))
    print("-" * 76)
    m = h5py.File(MASTER, "r")
    for name in ("full3d", "midacd", "interface"):
        h5 = "data/%s_pod_delta.h5" % name
        f = h5py.File(h5, "r")
        rec = f["reconstruction"]
        uref = rec["u_ref"][:].astype(np.float64)
        fid = rec["fluid_node_ids"][:] if "fluid_node_ids" in rec else None
        dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
        ids = sum((dec(f[s]["run_id"][:]) for s in ("train", "val", "test")), [])
        key, nc = FIELD[name]
        w, _ = build_weights(name, h5)
        W = np.repeat(w, nc) if nc == 3 else w

        uref_q = centre(uref, W, nc)
        du = np.empty((len(ids), len(uref)))
        for i, r in enumerate(ids):
            u = m[r][key][:]
            if fid is not None:
                u = u[fid]
            du[i] = centre(u.astype(np.float64).ravel(), W, nc) - uref_q

        cr, qd = terms(du, uref_q, W)
        cru, qdu = terms(du, uref_q, np.ones_like(W))
        share = 100 * cr.var() / (cr.var() + qd.var())
        print("%-10s %11.4e %11.4e %11.4e %9.3f %9.3f %8.1f%%" % (
            name, float((W * uref_q ** 2).sum()), cr.std(), qd.std(),
            qd.var() / cr.var(), qdu.var() / cru.var(), share))
    m.close()


if __name__ == "__main__":
    main()
