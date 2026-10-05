#!/usr/bin/env python
"""Design of experiments on the feeder load weights (Hofer, thesis section 8.3).

Seven weights on the simplex sum(w)=1, so six free coordinates.  J is strictly convex
in w (his section 10.3), which is what makes a fitted quadratic the right instrument:
28 coefficients (1 + 6 linear + 6 square + 15 cross), so ~45 well-placed points
locate the optimum and leave 17 degrees of freedom to check the fit.

Hofer's own optima sat at 0.72-1.25 times uniform (Table 10.4), so sampling +-50%
about uniform brackets them comfortably.
"""
import numpy as np, itertools, sys

N, NOM, SPAN = 7, 1.0 / 7, 0.5
rng = np.random.default_rng(7)


def terms(d):
    """Quadratic model row for the 6 free deviations."""
    r = [1.0] + list(d) + [d[i] * d[i] for i in range(6)]
    r += [d[i] * d[j] for i, j in itertools.combinations(range(6), 2)]
    return np.array(r)


def to_w(d6):
    d = np.append(d6, -d6.sum())
    return NOM * (1 + d)


def candidates(n=4000):
    out = []
    while len(out) < n:
        d = rng.uniform(-SPAN, SPAN, N)
        d -= d.mean()                      # onto sum(d)=0
        if np.abs(d).max() <= SPAN:
            out.append(d[:6])
        # else reject: the projection pushed a feeder outside the box
    return np.array(out)


def main(path):
    # Structured seed: uniform, then one feeder up with its mirror down
    seed = [np.zeros(6)]
    for i, j in [(0, 6), (1, 5), (2, 4), (0, 3), (3, 6), (1, 3)]:
        d = np.zeros(N); d[i] = SPAN * 0.8; d[j] = -SPAN * 0.8
        seed.append(d[:6])
    D = list(seed)

    # Greedy D-optimal augmentation
    C = candidates()
    X = np.array([terms(d) for d in D])
    while len(D) < 45:
        best, bv = None, -np.inf
        for c in C:
            Xa = np.vstack([X, terms(c)])
            s = np.linalg.svd(Xa, compute_uv=False)
            v = np.sum(np.log(s[s > 1e-12]))          # log-det surrogate, rank-safe
            if v > bv:
                bv, best = v, c
        D.append(best); X = np.vstack([X, terms(best)])

    W = np.array([to_w(d) for d in D])
    assert np.allclose(W.sum(1), 1), "weights must sum to one"
    assert W.min() > 0, "a weight went non-positive"
    s = np.linalg.svd(X, compute_uv=False)
    print(f"{len(D)} points; design matrix {X.shape}; condition number {s[0]/s[-1]:.1f}",
          file=sys.stderr)
    print(f"weight range {W.min()/NOM:.2f}..{W.max()/NOM:.2f} x uniform", file=sys.stderr)

    with open(path, "w") as f:
        f.write("idx," + ",".join(f"w{i+1}" for i in range(N)) + "\n")
        for k, w in enumerate(W):
            f.write(f"{k}," + ",".join(f"{x:.12f}" for x in w) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/barucca/alumina_level3/design_hofer.csv")
