"""Phase 1.0: the free pilot — does the reference-domain transfer reduce the POD rank?

The POD of Chapter 2 is taken in the node-index frame: entry k of a snapshot is
the field at node k, but node k sits at a slightly different place in every run
because the mesh follows the deformed interface.  The decomposition therefore
mixes a change of the field with a change of the mesh, and the reference-domain
construction exists to separate them.

The existing campaign is a free test of this, because its fluid domain already
varies in shape from run to run.  Transferring every run to one common geometry
and recomputing the decomposition answers the question with no solver runs at
all.  If the transfer does what it claims, the rank must drop.

Transfer.  All runs share connectivity and node identity, and the displacement
is a few millimetres against a mesh size of tens of millimetres, so the transfer
to the reference positions is the first-order pull-back

    u_i^ref(X_ref) = u_i(X_i) + (grad u_i) . (X_ref - X_i) + O(|dX|^2),

with the gradient exact per tetrahedron for a P1 field and averaged to nodes
with volume weights.  The neglected term is second order in a displacement of
~5e-3 m, i.e. ~2e-5 m^2 times the field curvature.  Section `roundtrip` measures
what it actually costs.

Note this is *not* the Piola transform: at this stage the map is a small
displacement of the same mesh, not a change of geometry, so there is no Jacobian
to account for beyond the identity.  The Piola question arises in Phase 1.1,
where the geometry genuinely differs.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import argparse
import time

import h5py
import numpy as np

from transfer_scoping import tet_gradients, nodal_average

MASTER = "../report_ML/master_ml.h5"
H5 = "data/full3d_pod_delta.h5"
OUT = "data/transfer_pilot.npz"


def spectrum(S, label):
    """POD by the method of snapshots on rows of S (already centred)."""
    K = S @ S.T
    w = np.linalg.eigvalsh(K)[::-1]
    w = np.clip(w, 0, None)
    evr = w / w.sum()
    cum = np.cumsum(evr)
    n = lambda t: int(np.searchsorted(cum, t) + 1)
    eff = 1.0 / np.sum(evr ** 2)
    print("  %-22s eff.rank %6.2f   n90 %4d   n95 %4d   n99 %4d   top3 %5.1f%%"
          % (label, eff, n(0.90), n(0.95), n(0.99), 100 * evr[:3].sum()))
    return evr


def main(limit, do_roundtrip):
    t0 = time.time()
    f = h5py.File(H5, "r")
    rec = f["reconstruction"]
    fid = rec["fluid_node_ids"][:]
    fel = rec["fluid_elems"][:]
    ref_name = f["meta"].attrs["reference_run"]
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    ids = dec(f["train"]["run_id"][:])
    if limit:
        ids = ids[:limit]
    N, Nn = len(ids), len(fid)
    print("transferring %d training runs, %d fluid nodes" % (N, Nn))

    M = h5py.File(MASTER, "r")
    Xref = M[ref_name]["mesh/cuveb_nodes"][:][fid].astype(np.float64)
    uref = M[ref_name]["fields_full/vitesse"][:][fid].astype(np.float64)

    A = np.empty((N, Nn * 3), dtype=np.float32)     # node-index frame (as today)
    B = np.empty((N, Nn * 3), dtype=np.float32)     # transferred to the reference
    rt = []
    for i, r in enumerate(ids):
        X = M[r]["mesh/cuveb_nodes"][:][fid].astype(np.float64)
        u = M[r]["fields_full/vitesse"][:][fid].astype(np.float64)
        dX = Xref - X
        g, vol = tet_gradients(X, fel, u)
        gn = nodal_average(fel, g, vol, Nn)
        u_ref_frame = u + np.einsum("nic,ni->nc", gn, dX)
        A[i] = (u - uref).ravel()
        B[i] = (u_ref_frame - uref).ravel()
        if do_roundtrip and i < 5:
            # push back to the run's own frame and compare with the original
            g2, vol2 = tet_gradients(Xref, fel, u_ref_frame)
            gn2 = nodal_average(fel, g2, vol2, Nn)
            back = u_ref_frame + np.einsum("nic,ni->nc", gn2, -dX)
            rt.append(np.linalg.norm(back - u) / np.linalg.norm(u))
        if i % 100 == 0:
            print("  %4d/%d  (%.1f min)" % (i, N, (time.time() - t0) / 60), flush=True)
    M.close()

    if rt:
        print("\nround-trip (transfer then back), relative L2 on 5 runs:")
        print("  " + "  ".join("%.3e" % v for v in rt))
        print("  -> the neglected second-order term; compare against the")
        print("     discretisation error of the mesh family, not against zero.")

    print("\nPOD spectra, %d training runs, delta about the reference run:" % N)
    A -= A.mean(0, keepdims=True)
    B -= B.mean(0, keepdims=True)
    evr_a = spectrum(A.astype(np.float64), "node-index frame")
    evr_b = spectrum(B.astype(np.float64), "reference frame")
    np.savez(OUT, evr_node=evr_a, evr_ref=evr_b, ids=np.array(ids))
    print("\nsaved spectra to %s   (total %.1f min)" % (OUT, (time.time() - t0) / 60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="use only the first N runs")
    ap.add_argument("--no-roundtrip", action="store_true")
    a = ap.parse_args()
    main(a.limit, not a.no_roundtrip)
