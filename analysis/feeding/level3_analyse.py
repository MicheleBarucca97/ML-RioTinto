#!/usr/bin/env python
"""Level 3: does the optimal feeder schedule move when the anode currents change?

Objective, following Hofer (EPFL TH 5023, eq. 8.3.4): the spatial variance of the
dissolved alumina, averaged over one feeding period.

The period is 168 s, not 84.  With f_Qm in {0.5, 1, 2} the per-feeder periods are
168, 84 and 42 s, so the pattern only repeats on their least common multiple.  For
the nominal configuration, where every feeder is on 84 s, averaging over 168 s gives
the same answer as averaging over 84, so one window serves every configuration.

Three numbers decide the question:
  J(w*_uniform, u_uniform)  the best schedule for nominal currents
  J(w*_weak,    u_weak)     the best schedule for anode-change currents
  J(w*_uniform, u_weak)     the nominal schedule, run at anode-change currents
The gap between the last two is what re-optimising is worth.  It is only meaningful
above the rotation noise floor, which is measured here from the same runs.
"""
import os, sys, glob
import numpy as np

PERIOD = 168.0
RES = "/work/gr-pi/alu-data/CNG/CNG2/level3/results"
DESIGN = "/home/barucca/alumina_level3/design.csv"


def read_csv(path):
    t, v = [], []
    for line in open(path):
        p = [x.strip() for x in line.split(",") if x.strip()]
        if len(p) < 3:
            continue
        try:
            ti, _, vi = float(p[0]), float(p[1]), float(p[2])
        except ValueError:
            continue
        t.append(ti); v.append(vi)
    return np.array(t), np.array(v)


def objective(path):
    """Period-averaged variance over the last complete 168 s window."""
    t, v = read_csv(path)
    if len(t) < 14:
        return None
    last = np.floor(t.max() / PERIOD) * PERIOD
    m = (t > last - PERIOD) & (t <= last)
    return v[m].mean() if m.sum() >= 12 else None


def load(kind, vel):
    out = {}
    for f in glob.glob(os.path.join(RES, f"{kind}_{vel}_*.csv")):
        k = int(os.path.basename(f).split("_")[-1].split(".")[0])
        J = objective(f)
        if J is not None:
            out[k] = J
    return out


def design():
    rows = {}
    for line in open(DESIGN).read().splitlines()[1:]:
        p = line.split(",")
        rows[int(p[0])] = (p[1], tuple(float(x) for x in p[2:]))
    return rows


def fmt(cfg):
    return " ".join(f"{x:g}" for x in cfg)


def main():
    D = design()
    floor, cfgJ = {}, {}
    for vel in ("uniform", "weak"):
        rot = load("rot", vel)
        if len(rot) >= 2:
            a = np.array(list(rot.values()))
            floor[vel] = 100 * (a.max() - a.min()) / a.mean()
        cfgJ[vel] = load("cfg", vel)
        print(f"{vel:>8}: {len(cfgJ[vel]):>3}/211 configurations, "
              f"{len(rot)}/7 rotations"
              + (f", noise floor {floor[vel]:.2f}%" if vel in floor else ""))

    if not all(len(cfgJ[v]) for v in ("uniform", "weak")):
        sys.exit("\nnot enough results yet")

    common = sorted(set(cfgJ["uniform"]) & set(cfgJ["weak"]))
    print(f"\n{len(common)} configurations evaluated on both velocity fields")

    bu = min(cfgJ["uniform"], key=cfgJ["uniform"].get)
    bw = min(cfgJ["weak"],    key=cfgJ["weak"].get)
    print(f"\nbest at uniform : cfg {bu:>3} [{fmt(D[bu][1])}]  J = {cfgJ['uniform'][bu]:.6f}")
    print(f"best at weak    : cfg {bw:>3} [{fmt(D[bw][1])}]  J = {cfgJ['weak'][bw]:.6f}")
    print(f"the optimum {'MOVES' if bu != bw else 'does NOT move'}")

    if bu in cfgJ["weak"]:
        Ja, Jb = cfgJ["weak"][bu], cfgJ["weak"][bw]
        gain = 100 * (Ja - Jb) / Ja
        f = floor.get("weak", float("nan"))
        print(f"\nVALUE OF RE-OPTIMISING, at weak currents:")
        print(f"  nominal-optimal schedule : J = {Ja:.6f}")
        print(f"  weak-optimal schedule    : J = {Jb:.6f}")
        print(f"  gain                     : {gain:.2f}%   (noise floor {f:.2f}%)")
        print(f"  -> {'REAL' if gain > 3*f else 'WITHIN NOISE'}"
              f"    [Hofer Table 10.4 got 1.1-7.4% from re-weighting]")

    if 0 in cfgJ["uniform"] and 0 in cfgJ["weak"]:
        for vel, b in (("uniform", bu), ("weak", bw)):
            n = cfgJ[vel][0]
            print(f"\ngain over current practice at {vel}: "
                  f"{100*(n-cfgJ[vel][b])/n:.2f}%  (nominal J = {n:.6f})")

    u = np.array([cfgJ["uniform"][k] for k in common])
    w = np.array([cfgJ["weak"][k] for k in common])
    from scipy.stats import spearmanr, pearsonr
    print(f"\nlandscape agreement across the two flows:")
    print(f"  Spearman rank rho = {spearmanr(u, w).statistic:.4f}")
    print(f"  Pearson r         = {pearsonr(u, w)[0]:.4f}")
    ru = {k: i for i, k in enumerate(sorted(common, key=lambda k: cfgJ['uniform'][k]))}
    rw = {k: i for i, k in enumerate(sorted(common, key=lambda k: cfgJ['weak'][k]))}
    print(f"  largest rank move = {max(abs(ru[k]-rw[k]) for k in common)} places "
          f"out of {len(common)}")

    print(f"\ntop 5 at each flow:")
    for vel in ("uniform", "weak"):
        print(f"  {vel}:")
        for k in sorted(cfgJ[vel], key=cfgJ[vel].get)[:5]:
            other = cfgJ["weak" if vel == "uniform" else "uniform"].get(k)
            o = f", other flow J = {other:.6f}" if other else ""
            print(f"    cfg {k:>3} [{fmt(D[k][1])}]  J = {cfgJ[vel][k]:.6f}{o}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RES = sys.argv[1]
    main()
