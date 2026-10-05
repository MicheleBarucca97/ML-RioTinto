"""Does stacking snapshots from different geometries break incompressibility?

Within one geometry the answer is no (divergence_audit.py: ratio 1.00 over 8 runs,
node motion of a few mm).  Across geometries the nodes move up to 70 mm, and a POD
mode is a linear combination of fields that each satisfy a *different* discrete
divergence operator.  Two questions:

  (a) how wrong is it to evaluate one geometry's field on another's mesh?
  (b) what is the divergence of an actual POD reconstruction, on the mesh it claims
      to live on?

eta = ||div u|| / ||grad u|| in L2, exact per tetrahedron for a P1 field.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import numpy as np, h5py, sys
sys.path.insert(0, "/work/gr-pi/alu-data/CNG/CNG2/geom_pilot")
sys.path.insert(0, "/home/barucca/ML-RioTinto")
import canonical_relabel as cr

def eta(nodes, elems, u):
    p = nodes[elems]; f = u[elems]
    e1, e2, e3 = p[:,1]-p[:,0], p[:,2]-p[:,0], p[:,3]-p[:,0]
    J = np.stack([e1,e2,e3], axis=1)
    det = np.linalg.det(J); ok = np.abs(det) > 1e-18
    Ji = np.zeros_like(J); Ji[ok] = np.linalg.inv(J[ok])
    g = np.einsum("eij,ejc->eic", Ji, f[:,1:]-f[:,0:1])
    vol = np.abs(det)/6.0
    div = g[:,0,0]+g[:,1,1]+g[:,2,2]
    return np.sqrt(np.sum(vol*div**2)) / max(np.sqrt(np.sum(vol*np.sum(g**2,axis=(1,2)))), 1e-300)


# Geometry-pilot run directories. The relabelling dumps
# (ASCII_cuveb_nodes_initial) are written by the export_initial_cuveb gate in
# stat_dataset/stationary.mac; they are not kept on disk between campaigns, so
# regenerate them before running this. Override the location with GEOM_ROOT.
import os as _os
GEOM_ROOT = _os.environ.get("GEOM_ROOT",
                            _os.path.expanduser("~/alucell_runs/geometry_pilot"))

dirs = [_os.path.join(GEOM_ROOT, "g%02d" % i) for i in range(1, 13)]
X, U = [], []
for d in dirs:
    A = cr.read_ascii_table("%s/%s" % (d, cr.MESHES["cuveb"]["dump"]), 3)
    p, _, _ = cr.canonical_permutation(A)
    X.append(A[p])
    with h5py.File(cr.run_h5(d), "r") as f:
        U.append(np.asarray(f["fields_full/vitesse"], dtype=np.float64)[p])
        E = np.asarray(f["mesh/cuveb_elems"])
        Xs = np.asarray(f["mesh/cuveb_nodes"], dtype=np.float64)[p]
    inv = np.empty(len(p), np.int64); inv[p] = np.arange(len(p))
    if 'Eref' not in dir(): Eref = inv[E - (1 if E.min()==1 else 0)]
    X[-1] = Xs                                   # the SOLVED mesh: where u actually lives
X = np.stack(X); U = np.stack(U)

print("(a) one geometry's field evaluated on another's mesh")
print("%-6s %12s %12s %8s" % ("pair", "own mesh", "foreign", "ratio"))
for j in [1, 3, 11]:
    a = eta(X[0], Eref, U[0]); b = eta(X[j], Eref, U[0])
    print("%-6s %12.4e %12.4e %8.2f" % ("g01/g%02d" % (j+1), a, b, b/a))

print("\n(b) POD reconstruction across the 12 geometries, on each own mesh")
Um = U.mean(0); D = (U - Um).reshape(12, -1)
w, V = np.linalg.eigh(D @ D.T); V = V[:, ::-1]; w = np.clip(w[::-1], 0, None)
B = (V.T @ D)                                     # unnormalised modes
B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-300)
for k in [1, 3, 6]:
    C = D @ B[:k].T
    R = (C @ B[:k]).reshape(12, -1, 3) + Um
    es = [eta(X[i], Eref, R[i]) for i in range(12)]
    e0 = [eta(X[i], Eref, U[i]) for i in range(12)]
    print("  k=%d  median eta reconstruction %.4e   solver %.4e   ratio %.2f"
          % (k, np.median(es), np.median(e0), np.median(es)/np.median(e0)))
