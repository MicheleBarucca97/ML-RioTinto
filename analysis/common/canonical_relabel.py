"""Phase 1: build one snapshot matrix across geometries by *relabelling*, not resampling.

The problem.  POD needs a matrix in which column j means the same physical thing in
every row.  Two runs at different (ACD, immersion) share neither node numbering nor
node positions: the mesh is regenerated, and the stationary module additionally
re-extracts `cuveb` from `allmesh` with its own numbering.  The obvious remedy is to
resample every run onto a fixed grid, but a fixed grid cannot be flat here -- the
metal/bath interface is a dome spanning 86 mm, 2.7x the ACD, so a plane through the
datum puts 38% of the plan area on the wrong side of it, and a fixed grid node would
be metal in one geometry and bath in another.  An interface-conforming grid would fix
that at the cost of a warp and a second interpolation error.

The remedy used here needs neither.  The mesh generator produces a *layered* mesh: the
same number of z-levels, with the same node count on each level, at every (ACD,
immersion) -- verified at the two opposite corners of the design, 2126 levels each, the
busiest carrying exactly 11025 nodes.  Ordering the nodes of each mesh by

    (rank of its z-level,  y,  x)

therefore defines a bijection between geometries, node k being the same *logical* node
-- same layer, same fibre -- at a different place.  Snapshots are then stacked directly,
with no interpolation at all.  This is the node-index frame the campaign already uses
for interface motion within a geometry, and 1.0 measured that removing it does not help.

The one catch, and why `ASCII_cuveb_nodes_initial` exists.  The rule works on the mesh
as generated.  It does *not* work on the solved mesh: one iteration of the free-surface
loop deforms the nodes in 3D and the level structure is gone (~200000 distinct z-levels
instead of 2126, max |dy| of 6.11 m between geometries under a sort of the deformed
coordinates).  So stat_dataset/stationary.mac dumps the undeformed nodes before the
loop starts, and the permutation is computed from those and applied to the solved
fields, which are stored in the same node numbering.

Usage
    canonical_relabel.py check   DIR [DIR ...]      # do the geometries correspond?
    canonical_relabel.py spectra DIR [DIR ...]      # Gate 1: rank of the (d,s) direction
each DIR holding ASCII_cuveb_nodes_initial and run.h5 for one geometry.
"""
import os
import re
import sys

import numpy as np


def design_point(d):
    """(ACD, immersion) for a geometry, read from its own mesh parameters.

    immersion = bath_height - ACD, the submerged depth of an anode, which is the
    second design variable; metal_height is held at its nominal value throughout.
    Returns (None, None) if the mesh directory cannot be found.
    """
    d = d.rstrip("/")
    parent, name = os.path.split(d)
    for cand in (d, os.path.join(parent, name[2:] if name[1:2] == "_" else name)):
        f = os.path.join(cand, "data", "_geometrical_params.mac")
        if not os.path.exists(f):
            continue
        txt = open(f).read()
        get = lambda k: next((float(m.group(1)) for m in
                              re.finditer(r"\(\s*%s\s*=\s*([0-9.eE+-]+)\s*\)" % k, txt)), None)
        acd, bath = get("ACD"), get("bath_height")
        if acd is not None and bath is not None:
            return acd, bath - acd
    return None, None


def run_h5(d):
    """The solved run that goes with a dump directory.

    The dump of the undeformed mesh is produced by its own short job, in a
    directory beside the solve rather than inside it, so `d_g07/` is paired with
    `s_g07/run.h5`.  A run.h5 sitting in the dump directory itself wins.
    """
    d = d.rstrip("/")
    local = os.path.join(d, "run.h5")
    if os.path.exists(local):
        return local
    parent, name = os.path.split(d)
    if name.startswith("d_"):
        return os.path.join(parent, "s_" + name[2:], "run.h5")
    return local


# --- what is being relabelled -------------------------------------------------------
# Both surrogate targets need the same treatment.  They differ only in which arrays
# carry them and, because the undeformed interface is flat, in whether there are any
# z-levels to rank at all.
MESHES = {
    "cuveb":     dict(dump="ASCII_cuveb_nodes_initial",
                      elems="mesh/cuveb_elems",
                      nodes="mesh/cuveb_nodes",
                      field="fields_full/vitesse",
                      layered=True),
    "interface": dict(dump="ASCII_interface_nodes_initial",
                      elems="mesh/interface_elems",
                      nodes="mesh/interface_nodes",
                      field="fields_interface/h",
                      layered=False),
}
MESH = MESHES["cuveb"]


# --- reading alucell's ASCII tables -------------------------------------------------
def read_ascii_table(path, ncomp):
    """Read a table written with ds_wrbr = ('x_y', i8, <ncomp>e25.15).

    Each record is a literal tag, a 1-based row index and `ncomp` reals.  The tag and
    the index are fixed-width and may run together, so the reals are taken from the
    end of the line rather than by field position.
    """
    vals = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < ncomp:
                continue
            try:
                row = [float(x) for x in parts[-ncomp:]]
            except ValueError:
                continue
            vals.append(row)
    return np.asarray(vals, dtype=np.float64)


# --- the canonical order ------------------------------------------------------------
def z_levels(z, tol=1e-7):
    """Group z into levels and return (level index per node, the level values).

    The generated mesh has exact levels, so any tolerance well below the layer spacing
    works; `tol` only guards against round-trip through 25-digit ASCII.
    """
    order = np.argsort(z, kind="stable")
    zs = z[order]
    new = np.empty(len(zs), dtype=bool)
    new[0] = True
    np.greater(np.diff(zs), tol, out=new[1:])
    lvl_of_sorted = np.cumsum(new) - 1
    lvl = np.empty(len(z), dtype=np.int64)
    lvl[order] = lvl_of_sorted
    return lvl, zs[new]


def canonical_permutation(X, tol=1e-7, xy_quantum=1e-9, layered=True):
    """Node order by (z-level rank, y, x).  Returns (perm, level index, level values).

    `perm` is such that X[perm] is in canonical order, so a field F in the mesh's own
    numbering becomes F[perm] in the frame shared with every other geometry.

    x and y are quantised first.  Within a level many nodes share a y exactly, and x
    is what separates them; if two such y differ only by round-off the sort would
    order them by that round-off and the two geometries could disagree.  Quantising to
    a nanometre -- six orders of magnitude below the smallest mesh feature -- makes
    them exact ties again, so x decides, in both geometries alike.

    The sloped walls scale the plan positions with height, and by a different factor
    in each geometry, but the scaling is positive and about the origin, so it leaves
    the order within a level unchanged.  That is what makes the rule geometry-free.
    """
    if layered:
        lvl, zvals = z_levels(X[:, 2], tol)
    else:
        # the undeformed interface is a single flat sheet: one level, order is (y, x)
        lvl = np.zeros(len(X), dtype=np.int64)
        zvals = X[:1, 2]
    xq = np.round(X[:, 0] / xy_quantum)
    yq = np.round(X[:, 1] / xy_quantum)
    perm = np.lexsort((xq, yq, lvl))
    return perm, lvl, zvals


def level_profile(lvl, n_levels):
    return np.bincount(lvl, minlength=n_levels)


def relabelled_connectivity(elems, perm):
    """Element table rewritten in the canonical node numbering, in a canonical order.

    This is the decisive test.  Node counts and level profiles can agree by accident;
    the connectivity cannot.  If two geometries give the same table here, the canonical
    order has matched the same logical node in both, and a snapshot matrix built by
    relabelling is well defined.  Each element's nodes are sorted (orientation is not
    part of the correspondence) and the elements are then sorted as tuples.
    """
    inv = np.empty(len(perm), dtype=np.int64)
    inv[perm] = np.arange(len(perm))
    e = np.sort(inv[elems], axis=1)
    return e[np.lexsort(e.T[::-1])]


# --- the correspondence check -------------------------------------------------------
def check(dirs):
    ref = None
    print("%-22s %9s %8s %10s   %s" % ("geometry", "nodes", "levels", "busiest", "verdict"))
    print("-" * 78)
    keep = []
    for d in dirs:
        X = read_ascii_table("%s/%s" % (d.rstrip("/"), MESH["dump"]), 3)
        perm, lvl, zvals = canonical_permutation(X, layered=MESH["layered"])
        prof = level_profile(lvl, len(zvals))
        name = d.rstrip("/").split("/")[-1]
        if ref is None:
            ref = (len(X), len(zvals), prof, X[perm])
            verdict = "(reference)"
        else:
            ok = (len(X) == ref[0] and len(zvals) == ref[1]
                  and np.array_equal(prof, ref[2]))
            verdict = "corresponds" if ok else "DIFFERS -- relabelling is not valid"
        print("%-22s %9d %8d %10d   %s" % (name, len(X), len(zvals), prof.max(), verdict))
        keep.append((name, X[perm], zvals, perm))

    # how far each matched node moves, which is the geometry change itself
    print("\ndisplacement of matched nodes against the first geometry (mm):")
    print("%-22s %10s %10s %10s" % ("geometry", "max |dx|", "max |dy|", "max |dz|"))
    for name, Xp, _, _ in keep[1:]:
        d = np.abs(Xp - keep[0][1]).max(axis=0) * 1e3
        print("%-22s %10.2f %10.2f %10.2f" % (name, d[0], d[1], d[2]))
    print("\nReading: dz should be the geometry change (bath_height and ACD), and dx, dy")
    print("only the sloped walls.  A dy of metres means the sort has matched the wrong")
    print("nodes -- which is what happens if this is run on the deformed mesh.")

    # first, a per-geometry sanity check that the dump and the solve share a numbering
    import h5py
    print("\ndump against its own solved mesh (same numbering, node k moved by the")
    print("free-surface loop).  dz should be the interface excursion, tens of mm;")
    print("metres in any column would mean the two are not the same node list.")
    print("%-22s %10s %10s %10s" % ("geometry", "max |dx|", "max |dy|", "max |dz|"))
    for (name, _, _, _), d in zip(keep, dirs):
        try:
            with h5py.File(run_h5(d), "r") as f:
                Xs = np.asarray(f[MESH["nodes"]], dtype=np.float64)
        except (OSError, KeyError):
            continue
        X0 = read_ascii_table("%s/%s" % (d.rstrip("/"), MESH["dump"]), 3)
        if len(X0) != len(Xs):
            print("%-22s  length mismatch %d vs %d" % (name, len(X0), len(Xs)))
            continue
        dd = np.abs(Xs - X0).max(axis=0) * 1e3
        print("%-22s %10.2f %10.2f %10.2f" % (name, dd[0], dd[1], dd[2]))

    # the decisive test: does the connectivity agree once relabelled?
    print("\nconnectivity in the canonical numbering:")
    base = None
    for (name, _, _, perm), d in zip(keep, dirs):
        h5 = run_h5(d)
        try:
            with h5py.File(h5, "r") as f:
                elems = np.asarray(f[MESH["elems"]])
        except (OSError, KeyError):
            print("  %-22s (no run.h5 -- skipped)" % name)
            continue
        if elems.min() == 1:                      # alucell tables are 1-based
            elems = elems - 1
        c = relabelled_connectivity(elems.astype(np.int64), perm)
        if base is None:
            base = c
            print("  %-22s (reference)  %d elements" % (name, len(c)))
        else:
            same = base.shape == c.shape and np.array_equal(base, c)
            print("  %-22s %s" % (name, "IDENTICAL" if same else
                                  "DIFFERS -- relabelling does NOT give a valid correspondence"))
    return keep


def spectra(dirs):
    """Gate 1: the singular values of the relabelled snapshot matrix."""
    import h5py
    keep = check(dirs)
    rows, names = [], []
    for (name, _, _, _), d in zip(keep, dirs):
        X = read_ascii_table("%s/%s" % (d.rstrip("/"), MESH["dump"]), 3)
        perm, _, _ = canonical_permutation(X, layered=MESH["layered"])
        with h5py.File(run_h5(d), "r") as f:
            u = np.asarray(f[MESH["field"]])
        rows.append(u[perm].ravel().astype(np.float64))
        names.append(name)
    G = np.vstack(rows)
    G -= G.mean(0, keepdims=True)
    w = np.clip(np.linalg.eigvalsh(G @ G.T)[::-1], 0, None)
    evr = w / w.sum()
    cum = np.cumsum(evr)
    print("\nPOD of the %d-geometry snapshot matrix (uniform current, mean removed):" % len(G))
    for i, (e, c) in enumerate(zip(evr, cum), 1):
        print("  mode %2d   evr %8.5f   cumulative %9.6f" % (i, e, c))
    print("\neffective rank %.2f" % (1.0 / np.sum(evr ** 2)))
    n = lambda t: int(np.searchsorted(cum, t) + 1)
    print("modes for 99%%: %d    99.9%%: %d    99.99%%: %d" % (n(.99), n(.999), n(.9999)))
    print("\nGate 1: pass if 3 modes reach 99.9%% and 5 reach 99.99%%;")
    print("        fail if 8 or more of %d are needed for 99%%." % len(G))

    # Is the geometry direction not merely low-rank but *smooth*?  Decision 2 models
    # A and Q as affine in (d, s); that is only legitimate if the modal coefficients
    # themselves are close to affine in (d, s).  Rank alone does not say so.
    gs = [design_point(d) for d in dirs]
    if all(a is not None for a, _ in gs):
        P = np.array([[1.0, a, b] for a, b in gs])
        w2, V = np.linalg.eigh(G @ G.T)
        V = V[:, ::-1]
        C = V * np.sqrt(np.clip(w2[::-1], 0, None))        # modal coefficients per run
        print("\nare the coefficients affine in (ACD, immersion)?")
        print("  %-6s %10s %10s %10s" % ("mode", "evr", "R2 affine", "R2 + quad"))
        Pq = np.c_[P, P[:, 1] ** 2, P[:, 2] ** 2, P[:, 1] * P[:, 2]]
        for m in range(min(5, C.shape[1])):
            y = C[:, m]
            ss = np.sum((y - y.mean()) ** 2)
            r2 = lambda M: 1 - np.sum((y - M @ np.linalg.lstsq(M, y, rcond=None)[0]) ** 2) / max(ss, 1e-300)
            print("  %-6d %10.5f %10.4f %10.4f" % (m + 1, evr[m], r2(P), r2(Pq)))
        print("  (an affine R2 near 1 on the leading modes is what licenses the")
        print("   pooled A(d,s) / Q(d,s) expansion; a large jump to the quadratic")
        print("   column says the expansion needs a second order in the geometry.)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] in MESHES:          # optional leading mesh selector
        MESH = MESHES[args.pop(0)]
    if len(args) < 2:
        sys.exit(__doc__ + "\n  a leading 'cuveb' or 'interface' selects the target"
                           " (default cuveb)\n")
    print("target mesh: %s  (%s)\n" % (MESH["dump"], MESH["field"]))
    (check if args[0] == "check" else spectra)(args[1:])
