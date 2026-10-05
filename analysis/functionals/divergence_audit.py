"""How far from divergence-free are the snapshots, and what does the POD cost?

`rem:ml-constraints` argues that incompressibility is a linear constraint satisfied by
every snapshot, hence by their mean, hence by every mode and every reconstruction.  The
argument needs one discrete divergence operator shared by all snapshots.  It is not:
B depends on node positions, and those move -- a few mm with the interface inside one
geometry, up to 70 mm between geometries.  This measures what that actually costs.

Three numbers, all dimensionless, all on the fluid submesh:

  eta   = ||div u||_{L2} / ||grad u||_{L2}     how far from divergence-free a field is
  the solver's own snapshots                    the benchmark: stabilised P1-P1 is not
                                                pointwise divergence-free either
  the POD reconstruction at k                   what truncation adds
  a cross-geometry combination                  what stacking different meshes adds

For a P1 field on tetrahedra the gradient is exact and constant per element, so both
norms are element sums with no quadrature error of their own.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import argparse
import numpy as np
import h5py

from transfer_scoping import tet_gradients


def eta(nodes, elems, u):
    """||div u|| / ||grad u|| in L2, both from the exact per-element P1 gradient."""
    g, vol = tet_gradients(nodes, elems, u)        # g[e,i,c] = d u_c / d x_i
    div = g[:, 0, 0] + g[:, 1, 1] + g[:, 2, 2]
    n_div = np.sqrt(np.sum(vol * div ** 2))
    n_grad = np.sqrt(np.sum(vol * np.sum(g ** 2, axis=(1, 2))))
    return n_div / max(n_grad, 1e-300)


def main(h5, master, n_runs):
    f = h5py.File(h5, "r")
    rec = f["reconstruction"]
    fid = rec["fluid_node_ids"][:]
    fel = rec["fluid_elems"][:]
    V = rec["modes"][:] if "modes" in rec else None
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    ids = dec(f["test"]["run_id"][:])[:n_runs]
    ref = f["meta"].attrs["reference_run"]

    M = h5py.File(master, "r")
    Xref = M[ref]["mesh/cuveb_nodes"][:][fid].astype(np.float64)
    print("fluid nodes %d, tets %d, %d runs\n" % (len(fid), len(fel), len(ids)))

    print("%-14s %12s %12s   %s" % ("run", "eta solver", "eta on Xref", "ratio"))
    print("-" * 56)
    rows = []
    for r in ids:
        X = M[r]["mesh/cuveb_nodes"][:][fid].astype(np.float64)
        u = M[r]["fields_full/vitesse"][:][fid].astype(np.float64)
        a = eta(X, fel, u)                 # on its own mesh: the solver's own residual
        b = eta(Xref, fel, u)              # same values, a different mesh: the error
        rows.append((a, b))                #   made by pretending geometries share one
        print("%-14s %12.4e %12.4e   %7.2f" % (r, a, b, b / max(a, 1e-300)))
    M.close()
    a = np.array(rows)
    print("\nmedian eta on its own mesh   %.4e   <- the stabilisation level, the benchmark"
          % np.median(a[:, 0]))
    print("median eta on a foreign mesh %.4e   <- what a shared operator would report"
          % np.median(a[:, 1]))
    print("median ratio                 %.2f" % np.median(a[:, 1] / a[:, 0]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/full3d_pod_delta.h5")
    p.add_argument("--master", default="../report_ML/master_ml.h5")
    p.add_argument("--runs", type=int, default=8)
    a = p.parse_args()
    main(a.h5, a.master, a.runs)
