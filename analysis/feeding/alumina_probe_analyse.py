"""Read the per-cycle mid-ACD fields and report the homogeneity metric.

Two things are wanted from the same files.  First, whether the alumina field has
settled into a quasi-periodic state -- if the cycle-to-cycle change in the metric is
still large at the last cycle, the run was too short and the comparison between
amplitudes means nothing.  Second, the metric itself: the standard deviation of the
cycle-averaged dissolved alumina on the mid-ACD plane, which is the homogeneity
objective used in the literature.
"""
import glob
import re
import sys

import numpy as np


def read_scalar(path):
    """Values only: data rows are the ones tagged 'x_y'.

    Taking the last token of any line with two or more of them also swallows the
    header -- `Number_of_nodes 1241` becomes a concentration of 1241 wt%, which
    inflates the standard deviation by an order of magnitude and is invisible in
    the mean.
    """
    vals = []
    for line in open(path):
        if not line.startswith("x_y"):
            continue
        try:
            vals.append(float(line.split()[-1]))
        except (ValueError, IndexError):
            pass
    return np.asarray(vals)


def series(tag):
    fs = sorted(glob.glob("/home/barucca/alumina_probe/cyc_%s_*.txt" % tag))
    return [(int(re.search(r"_(\d+)\.txt$", f).group(1)), read_scalar(f)) for f in fs]


def main(tags):
    print("%-6s %6s %10s %12s %12s   %s"
          % ("eps", "cycles", "n", "mean (wt%)", "sd (wt%)", "settled?"))
    print("-" * 72)
    ref = None
    for t in tags:
        s = series(t)
        if not s:
            print("%-6s %6s   (no output)" % (t, "-"))
            continue
        sds = [v.std() for _, v in s]
        last = s[-1][1]
        drift = abs(sds[-1] - sds[-2]) / max(abs(sds[-1]), 1e-30) if len(sds) > 1 else float("nan")
        eps = int(t) / 10.0
        print("%-6.1f %6d %10d %12.5f %12.6f   last-cycle change %.2f%%"
              % (eps, len(s), len(last), last.mean(), last.std(), 100 * drift))
        if t == "000":
            ref = last
    if ref is not None:
        print("\nchange in the homogeneity metric against the unperturbed flow:")
        print("%-8s %14s %14s %12s" % ("eps", "sd (wt%)", "delta sd", "relative"))
        for t in tags:
            s = series(t)
            if not s or t == "000":
                continue
            v = s[-1][1]
            d = v.std() - ref.std()
            print("%-8.1f %14.6f %14.6f %11.2f%%"
                  % (int(t) / 10.0, v.std(), d, 100 * d / ref.std()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["000", "005", "020", "050", "100"])
