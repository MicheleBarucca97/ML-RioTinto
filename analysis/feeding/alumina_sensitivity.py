"""How much does an error in the flow move the alumina homogeneity?

The composed pipeline of Section 10 -- predict the flow, then run the alumina solver
on the predicted flow -- is worth building only if the alumina field is not violently
sensitive to the flow error the surrogate leaves behind.  That sensitivity has never
been measured, and it is not obvious in either direction: the velocity enters the
transport equation advectively, so it sets *where* the alumina goes, but diffusion and
the averaging over a feeding cycle both smooth the result.

This builds perturbed copies of a converged velocity field at a range of amplitudes,
to be fed to the alumina model through the ASCII override in alumin/load_data.mac.
Running the model on each and comparing the standard deviation of the cycle-averaged
dissolved alumina -- the homogeneity objective used in the literature -- gives the
sensitivity as a curve rather than a single point.

The perturbation is smooth rather than white, because a POD-based surrogate's residual
is: it lives in, or just outside, the span of a few dozen smooth modes.  It is built
from low-frequency trigonometric modes over the cell's bounding box and scaled to a
prescribed relative L2 norm.  It is not a replica of the network's residual, which
cannot be had on this mesh; the point of sweeping the amplitude is that if the answer
is benign across the whole range then the exact character stops mattering.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import sys
import numpy as np

sys.path.insert(0, ".")
from canonical_relabel import read_ascii_table

SRC = "/home/barucca/alumina_probe/ASCII_cuveb_vitesse"
NODES = "/home/barucca/alumina_probe/ASCII_cuveb_nodes"


def smooth_field(X, n_modes=6, seed=0):
    """A divergence-agnostic smooth random vector field on the node set."""
    rng = np.random.default_rng(seed)
    L = X.max(0) - X.min(0)
    Xn = (X - X.min(0)) / L
    out = np.zeros_like(X)
    for _ in range(n_modes):
        k = rng.integers(1, 4, size=3)
        ph = rng.uniform(0, 2 * np.pi, size=3)
        a = rng.normal(size=3)
        basis = np.cos(2 * np.pi * (Xn * k) .sum(1) + ph[0])
        out += np.outer(basis, a)
    return out


def write_ascii(path, name, A):
    """alucell's ('x_y',i8,3e25.15) table, byte-compatible with what `write` emits.

    The header widths are copied from a file alucell wrote itself rather than
    guessed, because gen/read_tr.mac locates the counts by token position.
    """
    with open(path, "w") as f:
        f.write("\n Name_of br dataset     %s\n" % name)
        f.write(" Number_of_nodes       %11d\n" % len(A))
        f.write(" Number_of_coordinates %11d\n\n" % A.shape[1])
        for i, row in enumerate(A, 1):
            f.write("x_y%8d" % i + "".join("%25.15E" % v for v in row) + "\n")


def main(amplitudes):
    u = read_ascii_table(SRC, 3)
    X = read_ascii_table(NODES, 3) if __import__("os").path.exists(NODES) else None
    nu = np.linalg.norm(u)
    print("velocity field: %d nodes, |u| = %.6e" % (len(u), nu))
    if X is None:
        print("node coordinates absent; using an index-based smooth basis")
        X = np.c_[np.arange(len(u)), np.zeros(len(u)), np.zeros(len(u))]
    for eps in amplitudes:
        e = smooth_field(X, seed=int(eps * 1000))
        e *= eps * nu / np.linalg.norm(e)
        out = "/home/barucca/alumina_probe/ASCII_cuveb_vitesse_in_%03d" % round(eps * 1000)
        write_ascii(out, "cuveb_vitesse_in", u + e)
        print("  eps = %5.1f%%   ->  %s   (check |e|/|u| = %.4f)"
              % (100 * eps, out.split("/")[-1], np.linalg.norm(e) / nu))


if __name__ == "__main__":
    main([0.0, 0.005, 0.02, 0.05, 0.10])
