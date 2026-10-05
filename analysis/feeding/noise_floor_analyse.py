#!/usr/bin/env python
"""Level-1 noise floor: spread in the objective across cyclic rotations of the feeder
firing order.

At the periodic state the seven rotations are exact time-shifts of one another (the
firing times are uniform, 84/7 = 12 s), so they must give the same period-averaged
objective.  Whatever spread remains is non-convergence error, and it is the floor
below which no optimisation result can be believed.

Objective, following Hofer (EPFL TH 5023, eq. 8.3.4): the spatial variance of the
alumina concentration, averaged over one feeding period.  A period average is
phase-invariant, so the rotations can be compared cycle by cycle with no alignment.
"""
import sys, glob, os
import numpy as np

PERIOD = 84.0

def read_run(d):
    f = os.path.join(d, "control", "CONTROL_MEANVAR_SUMMARY.csv")
    t, var = [], []
    with open(f) as fh:
        for line in fh:
            p = [x.strip() for x in line.split(",") if x.strip()]
            if len(p) < 3:
                continue
            try:
                ti, _, vi = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                continue          # header
            t.append(ti); var.append(vi)
    return np.array(t), np.array(var)

def period_means(t, var):
    """Mean variance within each complete feeding cycle."""
    out = {}
    cyc = np.ceil(t / PERIOD).astype(int)          # t in (84(n-1), 84n] -> cycle n
    for n in np.unique(cyc):
        if n < 1:
            continue
        m = cyc == n
        if m.sum() >= 6:                           # require a near-complete cycle
            out[int(n)] = var[m].mean()
    return out

def main(root):
    runs = sorted(glob.glob(os.path.join(root, "rot*")))
    if not runs:
        sys.exit(f"no rot* directories under {root}")
    pm, labels = [], []
    for d in runs:
        try:
            t, v = read_run(d)
        except FileNotFoundError:
            print(f"  [skip] {os.path.basename(d)}: no control file yet"); continue
        if len(t) == 0:
            print(f"  [skip] {os.path.basename(d)}: control file empty"); continue
        pm.append(period_means(t, v)); labels.append(os.path.basename(d))
    if len(pm) < 2:
        sys.exit("need at least two completed rotations")

    common = sorted(set.intersection(*[set(d) for d in pm]))
    print(f"rotations: {', '.join(labels)}")
    print(f"complete cycles common to all: {len(common)}\n")
    print(f"{'cycle':>6} {'t [s]':>7} {'mean J':>12} {'sigma [wt%]':>12} "
          f"{'spread %':>9} {'std %':>8}")
    print("-" * 60)
    rows = []
    for n in common:
        vals = np.array([d[n] for d in pm])
        mu = vals.mean()
        spread = 100 * (vals.max() - vals.min()) / mu
        sd = 100 * vals.std(ddof=1) / mu
        rows.append((n, mu, spread, sd))
        print(f"{n:>6} {n*PERIOD:>7.0f} {mu:>12.6f} {np.sqrt(mu):>12.4f} "
              f"{spread:>8.2f}% {sd:>7.2f}%")

    if rows:
        n, mu, spread, sd = rows[-1]
        print(f"\nNOISE FLOOR at cycle {n} (t = {n*PERIOD:.0f} s):")
        print(f"  peak-to-peak spread across rotations : {spread:.2f}%  of J")
        print(f"  standard deviation across rotations  : {sd:.2f}%  of J")
        print(f"  -> an optimisation gain is only credible above ~{spread:.1f}% in J")
        print(f"     (Hofer measured 1.8% for u_a and 0.5% for u_b, Tables 10.5/10.9)")
        print("\nper-rotation J at the final common cycle:")
        for lab, d in zip(labels, pm):
            print(f"  {lab:>6}: {d[n]:.6f}   (sigma = {np.sqrt(d[n]):.4f} wt%)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/barucca/alumina_noise")
