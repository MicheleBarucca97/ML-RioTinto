"""Can the planned campaign actually determine A(g) and Q(g)?

The plan says "2 anchor geometries x 92 runs -> Q".  That claim was never checked, and
the quadratic-in-geometry finding (Measurement: affine predicts only 92.1% of the
interface variance out of sample) raises the number of unknowns by a factor of two.
This builds the actual design matrix and takes its rank.

Model, per POD mode.  Currents enter as a deviation p on the hyperplane 1'p = 0, so
work in y = B'p with B an orthonormal basis of that 23-dimensional space:

    c(y, g) = phi(g) (x) [ y ; screened products y_i y_j ]

with phi(g) the geometry basis -- [1, dd, ds] affine, or with the three second-order
terms as well.  What each run block contributes is *not* the same:

  foldover pair at +-delta h_j
      odd  part  (c(+) - c(-))/2 = delta a.h_j            -> one linear number
      even part  (c(+) + c(-))/2 - c(0) = delta^2 h_j'Q h_j / 2
                                                          -> ONE quadratic number,
                                                             the diagonal of Q in the
                                                             Hadamard basis
  interior random run                                     -> one mixed-direction number

So 23 foldover pairs give 23 of the 276 free entries of Q, not 276.  That is the thing
the plan's sentence hides, and it is why the anchors alone cannot pin Q(g).
"""
import numpy as np


def hadamard24():
    """Paley construction, order 24 (q = 23 prime).  Rows 2..24 sum to zero."""
    q = 23
    res = {(i * i) % q for i in range(1, q)}
    chi = np.array([0 if i == 0 else (1 if i in res else -1) for i in range(q)])
    Q = np.array([[chi[(j - i) % q] for j in range(q)] for i in range(q)])
    S = np.zeros((q + 1, q + 1))
    S[0, 1:] = 1; S[1:, 0] = -1; S[1:, 1:] = Q
    H = S + np.eye(q + 1)
    assert np.allclose(H @ H.T, 24 * np.eye(24)), "not Hadamard"
    return H


def basis_null_ones(n=24):
    B, _ = np.linalg.qr(np.eye(n) - np.ones((n, n)) / n)
    return B[:, :n - 1]                                    # 24 x 23, columns span 1^perp


def geom_phi(g, order):
    d, s = g[:, 0], g[:, 1]
    d = (d - d.mean()) / d.std(); s = (s - s.mean()) / s.std()
    cols = [np.ones_like(d), d, s]
    if order == 2:
        cols += [d * d, s * s, d * s]
    return np.stack(cols, 1)


def features(Y, keep):
    """[ y ; the retained products ] for each row of Y (n x 23)."""
    idx = np.triu_indices(Y.shape[1])
    prod = (Y[:, idx[0]] * Y[:, idx[1]])[:, keep]
    return np.concatenate([Y, prod], 1)


def build(n_sweep, n_anchor, n_interior, keep_q, seed=0):
    rng = np.random.default_rng(seed)
    H = hadamard24()[1:]                                    # 23 sum-zero rows
    B = basis_null_ones()
    # design points: sweep on a grid over the box, anchors at two interior points
    k = int(np.ceil(np.sqrt(n_sweep)))
    dd, ss = np.meshgrid(np.linspace(25, 45, k), np.linspace(125, 175, k))
    G = np.c_[dd.ravel(), ss.ravel()][:n_sweep]
    G = np.vstack([G, np.array([[30., 140.], [40., 160.]])[:n_anchor]])
    rows_g, rows_p = [], []
    for gi, g in enumerate(G):
        anchor = gi >= n_sweep
        amps = [4000., 2000.] if anchor else [4000.]
        for a in amps:                                      # foldover, both signs
            for h in H:
                for sgn in (+1, -1):
                    rows_g.append(g); rows_p.append(sgn * a * h / 24.)
        rows_g.append(g); rows_p.append(np.zeros(24))       # the base run
        if not anchor:
            for _ in range(n_interior):
                v = rng.normal(size=24); v -= v.mean()
                rows_g.append(g); rows_p.append(4000. * v / np.abs(v).max() / 24.)
    G_all = np.array(rows_g); P = np.array(rows_p)
    Y = P @ B
    Y /= max(np.abs(Y).max(), 1e-30)
    return G_all, Y, len(G)


def report(n_sweep, n_anchor=2, n_interior=7, n_keep=40):
    keep = np.arange(n_keep)                                # a screened subset, as fitted
    G, Y, n_geom = build(n_sweep, n_anchor, n_interior, keep)
    F = features(Y, keep)
    print("\n%d sweep + %d anchor geometries -> %d runs, %d distinct geometries"
          % (n_sweep, n_anchor, len(Y), n_geom))
    for order, name in [(1, "affine in g"), (2, "quadratic in g")]:
        Phi = geom_phi(G, order)
        X = np.einsum("ng,nf->ngf", Phi, F).reshape(len(F), -1)
        r = np.linalg.matrix_rank(X, tol=1e-8)
        sv = np.linalg.svd(X, compute_uv=False)
        cond = sv[0] / max(sv[r - 1], 1e-300)
        print("  %-16s cols %4d  rank %4d  %-12s oversampling %5.1fx  cond %.2e"
              % (name, X.shape[1], r,
                 "FULL RANK" if r == X.shape[1] else "RANK-DEFICIENT",
                 len(X) / X.shape[1], cond))


if __name__ == "__main__":
    for n in (24, 16, 12):
        report(n)
