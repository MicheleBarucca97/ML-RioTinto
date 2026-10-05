#!/usr/bin/env python
"""Write a campaign run's SOLVER velocity in the form the alumina module reads.

Companion to surrogate_velocity_for_alumin.py: same file format, but the field is
the truth from the campaign rather than a network prediction, so the two can be fed
to the alumina model under otherwise identical conditions.
"""
import sys
import h5py
import numpy as np

MASTER = "/home/barucca/report_ML/master_ml.h5"


def write_ascii(path, name, A):
    with open(path, "w") as f:
        f.write("\n Name_of br dataset     %s\n" % name)
        f.write(" Number_of_nodes       %11d\n" % len(A))
        f.write(" Number_of_coordinates %11d\n\n" % A.shape[1])
        for i, row in enumerate(A, 1):
            f.write("x_y%8d" % i + "".join("%25.15E" % v for v in row) + "\n")


def main(run, out):
    with h5py.File(MASTER, "r") as f:
        if run not in f:
            sys.exit("no such run: %s" % run)
        U = f[f"{run}/fields_full/vitesse"][:].astype(np.float64)
        I = f[f"{run}/input/currents"][:]
    print(f"{run}: |u|max = {np.linalg.norm(U,axis=1).max():.5f} m/s, "
          f"currents {I.min():.1f}..{I.max():.1f} kA, "
          f"spread {100*(I.max()-I.min())/I.mean():.1f}%")
    write_ascii(out, "cuveb_vitesse_in", U)
    print("  ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
