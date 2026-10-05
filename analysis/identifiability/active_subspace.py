"""Phase 0.4, completion: does the network find input directions the linear map does not?

For a general f the active subspace is spanned by the dominant eigenvectors of
C = E[grad f grad f^T].  For an affine f, grad f is the constant A^T and
C = A^T A, so the active subspace is exactly the span of the leading right
singular vectors of A.  The construction therefore extends the sensitivity
operator to the trained network by automatic differentiation, and comparing the
two subspaces measures what the network actually adds.

Reported:
  - singular spectra of A (linear) and of C_net^{1/2} (network), side by side;
  - principal angles between the leading r-dimensional subspaces;
  - the subspace overlap  ||W_lin^T W_net||_F^2 / r,  which is 1 iff the two
    r-dimensional subspaces coincide and 0 iff they are orthogonal.

The gradient is taken through the network only, i.e. of the map
p -> c(theta; p), which is the same object A approximates.  Since V_k has
orthonormal columns the field-space spectrum equals the coefficient-space one
(see the report, Lemma "the spectrum is computable on the small matrix").
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

from utils import build_model, resolve_checkpoint

_RCOND = 1e-4


def main(target, ranks):
    cfg = yaml.safe_load(open("config_alucell_%s.yaml" % target))
    f = h5py.File(cfg["data"]["h5_path"], "r")
    Ptr = f["train/P"][:].astype(np.float64)
    Utr = f["train/U"][:].astype(np.float64)
    Pte = f["test/P"][:].astype(np.float64)

    # ---- linear sensitivity operator -------------------------------------
    beta, *_ = np.linalg.lstsq(Ptr, Utr, rcond=_RCOND)   # [24, k]
    A = beta.T                                            # [k, 24]
    _, s_lin, Wt_lin = np.linalg.svd(A, full_matrices=False)
    W_lin = Wt_lin.T                                      # [24, .] right sing. vecs

    # ---- network active subspace ----------------------------------------
    m = build_model(cfg)
    m.load_state_dict(torch.load(resolve_checkpoint(cfg), map_location="cpu",
                                 weights_only=True))
    m.eval().double()
    k = A.shape[0]
    C = np.zeros((24, 24))
    X = torch.tensor(Pte, dtype=torch.float64)
    for i in range(len(X)):
        J = torch.autograd.functional.jacobian(
            lambda p: m(p[None], None)[0], X[i], vectorize=True)   # [k, 24]
        Ji = J.detach().numpy()
        C += Ji.T @ Ji
    C /= len(X)
    w, W_net = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, W_net = np.clip(w[o], 0, None), W_net[:, o]
    s_net = np.sqrt(w)

    print("\n=== %s === active subspace of the network vs the linear operator" % target)
    print("  (network gradients averaged over the %d test runs)" % len(X))
    print("\n  %-4s %12s %12s %8s" % ("i", "sigma_i(A)", "sigma_i(net)", "ratio"))
    for i in list(range(6)) + [21, 22, 23]:
        rr = ("%8.3f" % (s_net[i] / s_lin[i])) if s_lin[i] > 1e-12 else "     inf"
        print("  %-4d %12.5f %12.5f %s" % (i + 1, s_lin[i], s_net[i], rr))

    print("\n  %-6s %14s %14s" % ("rank r", "overlap", "max principal angle"))
    for r in ranks:
        Wa, Wb = W_lin[:, :r], W_net[:, :r]
        ov = float(np.sum((Wa.T @ Wb) ** 2) / r)
        sv = np.linalg.svd(Wa.T @ Wb, compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(sv.min(), -1, 1)))
        print("  %-6d %14.4f %13.1f deg" % (r, ov, ang))
    print("\n  overlap = ||W_lin^T W_net||_F^2 / r : 1 = identical subspaces,")
    print("  0 = orthogonal.  A high overlap means the network has learned")
    print("  curvature along the directions the linear map already found,")
    print("  not new input directions.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="full3d",
                    choices=["midacd", "full3d", "interface"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 3, 5, 10, 15, 23])
    a = ap.parse_args()
    main(a.target, a.ranks)
