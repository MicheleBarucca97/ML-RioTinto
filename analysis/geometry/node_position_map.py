"""Are the node positions themselves a closed-form function of the design?

In the relabelled frame the surrogate predicts the field at logical node k.  Turning
that into a field on physical space needs to know where node k is at the requested
design point, i.e. that geometry's mesh -- unless the positions are predictable too.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import numpy as np, sys
sys.path.insert(0, "/work/gr-pi/alu-data/CNG/CNG2/geom_pilot")
import canonical_relabel as cr


# Geometry-pilot run directories. The relabelling dumps
# (ASCII_cuveb_nodes_initial) are written by the export_initial_cuveb gate in
# stat_dataset/stationary.mac; they are not kept on disk between campaigns, so
# regenerate them before running this. Override the location with GEOM_ROOT.
import os as _os
GEOM_ROOT = _os.environ.get("GEOM_ROOT",
                            _os.path.expanduser("~/alucell_runs/geometry_pilot"))

dirs = [_os.path.join(GEOM_ROOT, "g%02d" % i) for i in range(1, 13)]
X = []
for d in dirs:
    A = cr.read_ascii_table("%s/%s" % (d, cr.MESHES["cuveb"]["dump"]), 3)
    p, _, _ = cr.canonical_permutation(A)
    X.append(A[p])
X = np.stack(X)
g = np.array([cr.design_point(d) for d in dirs]) * 1e3
M = np.c_[np.ones(12), g[:, 0] - g[:, 0].mean(), g[:, 1] - g[:, 1].mean()]
print("nodes %d ; X(d,s) = X0 + Xd.dd + Xs.ds, affine, fitted per node" % X.shape[1])
err = []
for i in range(12):
    k = np.ones(12, bool); k[i] = False
    B = np.linalg.lstsq(M[k], X[k].reshape(11, -1), rcond=None)[0]
    err.append(np.abs((M[i] @ B).reshape(-1, 3) - X[i]).max())
err = np.array(err) * 1e6
print("leave-one-out max node-position error: median %.4f um   worst %.4f um"
      % (np.median(err), err.max()))
print("(elements are tens of mm, so 1 um is 1e-5 of one)")
