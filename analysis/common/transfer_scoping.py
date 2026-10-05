"""Phase 1.0, scoping: how much of the snapshot variation is mesh motion?

The reference-domain transfer exists to remove a confound: the POD of Chapter 2
is taken in the node-index frame, where entry k of a snapshot is the field at
node k -- but node k sits at a slightly different place in every run, because
the mesh follows the deformed interface.  The decomposition therefore mixes a
change of the field with a change of the mesh.

Transferring every run to one common geometry removes the second part.  Whether
that is worth doing depends on how large it is, and that can be estimated
without performing the transfer at all.  Writing X_i for run i's node positions
and u_i for its field, the field of run i evaluated at the *reference* positions
is, to first order in the displacement,

    u_i^ref(X_ref) = u_i(X_i) + (grad u_i) . (X_ref - X_i) + O(|dX|^2),

so the magnitude of the correction is |grad u_i| |dX_i|.  This script measures
that against the run-to-run perturbation |u_i - u_ref| that the surrogate is
trained on.  If the correction is small compared with the perturbation, the
node-index frame is already almost the reference frame, the transfer can remove
little variance, and the POD rank will not drop much.

Gradients are recovered per tetrahedron (exact for P1) and averaged to nodes
with volume weights.
"""
import argparse

import h5py
import numpy as np

MASTER = "../report_ML/master_ml.h5"


def tet_gradients(nodes, elems, field):
    """Per-element gradient of a P1 field on tetrahedra.

    Returns (grad [Ne, 3, ncomp], vol [Ne]).  For a linear field on a simplex
    the gradient is exact and constant per element.
    """
    p = nodes[elems]                                   # [Ne,4,3]
    f = field[elems]                                   # [Ne,4,nc]
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    J = np.stack([e1, e2, e3], axis=1)                 # [Ne,3,3]
    det = np.linalg.det(J)
    ok = np.abs(det) > 1e-18
    Jinv = np.zeros_like(J)
    Jinv[ok] = np.linalg.inv(J[ok])
    df = f[:, 1:] - f[:, 0:1]                          # [Ne,3,nc]
    grad = np.einsum("eij,ejc->eic", Jinv, df)         # [Ne,3,nc]
    return grad, np.abs(det) / 6.0


def nodal_average(elems, per_elem, weights, n_nodes):
    """Volume-weighted scatter of a per-element quantity to nodes."""
    shp = per_elem.shape[1:]
    acc = np.zeros((n_nodes,) + shp)
    wsum = np.zeros(n_nodes)
    for k in range(4):
        np.add.at(acc, elems[:, k], per_elem * weights[:, None, None])
        np.add.at(wsum, elems[:, k], weights)
    wsum[wsum == 0] = 1.0
    return acc / wsum.reshape((-1,) + (1,) * len(shp))


def main(n_runs, seed):
    cfg_h5 = "data/full3d_pod_delta.h5"
    f = h5py.File(cfg_h5, "r")
    rec = f["reconstruction"]
    fid = rec["fluid_node_ids"][:]
    fel = rec["fluid_elems"][:]
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    ids = dec(f["train"]["run_id"][:])
    ref_name = f["meta"].attrs["reference_run"]

    rng = np.random.default_rng(seed)
    sample = [ids[i] for i in rng.choice(len(ids), min(n_runs, len(ids)), replace=False)]

    with h5py.File(MASTER, "r") as M:
        Xref = M[ref_name]["mesh/cuveb_nodes"][:][fid].astype(np.float64)
        uref = M[ref_name]["fields_full/vitesse"][:][fid].astype(np.float64)
        print("fluid nodes %d, fluid tets %d, reference run %s"
              % (len(fid), len(fel), ref_name))
        print("\n%-12s %11s %11s %11s %9s" %
              ("run", "|du| (pert)", "|G.dX| corr", "ratio", "max|dX|"))
        print("-" * 60)
        rows = []
        for r in sample:
            X = M[r]["mesh/cuveb_nodes"][:][fid].astype(np.float64)
            u = M[r]["fields_full/vitesse"][:][fid].astype(np.float64)
            dX = Xref - X                                   # to the reference frame
            g, vol = tet_gradients(X, fel, u)               # [Ne,3,3]
            gn = nodal_average(fel, g, vol, len(fid))       # [Nn,3,3]
            corr = np.einsum("nic,ni->nc", gn, dX)          # (grad u).dX  [Nn,3]
            du = u - uref
            n_du = np.linalg.norm(du)
            n_co = np.linalg.norm(corr)
            rows.append((n_du, n_co))
            print("%-12s %11.4e %11.4e %11.4f %9.2e"
                  % (r, n_du, n_co, n_co / max(n_du, 1e-30), np.abs(dX).max()))
    a = np.array(rows)
    ratio = a[:, 1] / np.maximum(a[:, 0], 1e-30)
    print("\nmesh-motion correction as a fraction of the learned perturbation:")
    print("  median %.3f   mean %.3f   min %.3f   max %.3f"
          % (np.median(ratio), ratio.mean(), ratio.min(), ratio.max()))
    print("\nReading: this is the share of each snapshot that the transfer would")
    print("move.  Well below 1 means the node-index frame is already close to a")
    print("common frame and the POD rank cannot drop much; of order 1 or more")
    print("means the present decomposition is substantially contaminated by mesh")
    print("motion and the transfer is essential.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.runs, a.seed)
