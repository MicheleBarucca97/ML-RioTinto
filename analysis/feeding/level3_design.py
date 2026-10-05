#!/usr/bin/env python
"""The feeder design space: per-feeder injection frequency, Patouillet's variable.

f_Qm_i scales feeder i's firing rate -- period_i = 84/f_Qm_i at a fixed 1.1205 kg dose
(alucell/sou/fes_alumin/feeding_table.cpp:106 makes the per-injection weight
independent of f_Qm, so f_Qm is exactly the mass-flow fraction).

Sum(f_Qm) = 7 keeps the cell's total feed rate equal to consumption.  Without that
constraint the bath drifts and J rises for a reason that has nothing to do with
distribution (Hofer, thesis section 8.3).  With values in {0.5, 1, 2} the constraint
forces #(at 2) = #(at 0.5)/2, giving two families plus the nominal.
"""
from itertools import combinations
import csv, sys

VALUES = (0.5, 1.0, 2.0)
N = 7


def design():
    out = [((1.0,) * N, "nominal")]
    for a, c in ((2, 1), (4, 2)):                     # #halved, #doubled
        for half in combinations(range(N), a):
            rest = [i for i in range(N) if i not in half]
            for dbl in combinations(rest, c):
                cfg = [1.0] * N
                for i in half:
                    cfg[i] = 0.5
                for i in dbl:
                    cfg[i] = 2.0
                out.append((tuple(cfg), f"h{a}d{c}"))
    return out


if __name__ == "__main__":
    d = design()
    assert all(abs(sum(c) - 7) < 1e-9 for c, _ in d)
    assert len({c for c, _ in d}) == len(d), "duplicate configuration"
    w = csv.writer(sys.stdout if len(sys.argv) < 2 else open(sys.argv[1], "w"))
    w.writerow(["idx", "family"] + [f"f_Qm_{i+1}" for i in range(N)])
    for k, (c, fam) in enumerate(d):
        w.writerow([k, fam] + list(c))
    print(f"{len(d)} configurations", file=sys.stderr)
