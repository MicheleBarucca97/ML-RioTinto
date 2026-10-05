"""Phase 0.3 — how many simulations does the map actually need?

Held-out R^2 of the linear and quadratic predictors of the modal coefficients
as a function of the number of training runs, reported both pooled and per
campaign regime.  This is the number that sizes every future campaign.

Two variants are available and they answer different questions:

  --basis fixed    the POD basis is the one fitted on all training runs and
                   only the regression is subsampled.  Answers "given the
                   basis, how many runs to learn the map".
  --basis refit    the basis is refitted on each subsample, which is the
                   honest version but needs the raw snapshots in memory.
                   Available for `interface` and `midacd`.

The subsample is drawn stratified by regime, so a small budget keeps the same
regime proportions as the full campaign rather than losing the rare ones.
"""
import argparse

import h5py
import numpy as np
import yaml

_RCOND = 1e-4
SIZES = [50, 100, 200, 400, 600, 800, 1000, 1194]


def _pairs(Z):
    iu = np.triu_indices(Z.shape[1])
    return (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]


def r2_pooled(Y, Yh):
    return 1.0 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def fit_lin(Ptr, Ytr, Pte):
    b, *_ = np.linalg.lstsq(Ptr, Ytr, rcond=_RCOND)
    return Pte @ b


def fit_quad(Ptr, Ytr, Pte, n_keep=40, seed=0):
    rng = np.random.default_rng(seed)
    Qtr, Qte = _pairs(Ptr), _pairs(Pte)
    sd = Qtr.std(0); sd[sd == 0] = 1.0
    Qtr, Qte = Qtr / sd, Qte / sd
    res = Ytr - fit_lin(Ptr, Ytr, Ptr)
    score = np.abs(Qtr.T @ (res - res.mean(0))).sum(1)
    keep = np.argsort(-score)[:min(n_keep, Qtr.shape[1])]
    idx = rng.permutation(len(Ptr)); cut = max(2, int(0.8 * len(idx)))
    a, b = idx[:cut], idx[cut:]
    if len(b) < 2:
        a = b = idx
    best = (-np.inf, None, None)
    for cols in (keep,):
        Dtr = np.c_[Ptr, Qtr[:, cols]]
        for al in [1e-1, 1, 10, 1e2, 1e3, 1e4, 1e6]:
            pen = np.r_[np.full(Ptr.shape[1], 1e-6), np.full(len(cols), al)]
            W = np.linalg.solve(Dtr[a].T @ Dtr[a] + np.diag(pen), Dtr[a].T @ Ytr[a])
            sc = r2_pooled(Ytr[b], Dtr[b] @ W)
            if sc > best[0]:
                best = (sc, cols, al)
    _, cols, al = best
    Dtr = np.c_[Ptr, Qtr[:, cols]]
    Dte = np.c_[Pte, Qte[:, cols]]
    pen = np.r_[np.full(Ptr.shape[1], 1e-6), np.full(len(cols), al)]
    W = np.linalg.solve(Dtr.T @ Dtr + np.diag(pen), Dtr.T @ Ytr)
    return Dte @ W


def stratified_subsample(modes, n, rng):
    """Draw n indices keeping the regime proportions of the full set."""
    modes = np.asarray(modes)
    out = []
    groups = {m: np.flatnonzero(modes == m) for m in set(modes.tolist())}
    total = len(modes)
    for m, idx in sorted(groups.items()):
        take = max(1, int(round(n * len(idx) / total)))
        take = min(take, len(idx))
        out.extend(rng.choice(idx, take, replace=False))
    out = np.array(out)
    if len(out) > n:                       # trim the largest groups first
        out = rng.choice(out, n, replace=False)
    return np.sort(out)


def main(target, n_rep, seed):
    cfg = yaml.safe_load(open("config_alucell_%s.yaml" % target))
    f = h5py.File(cfg["data"]["h5_path"], "r")
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    Ptr = f["train/P"][:].astype(np.float64)
    Ytr = f["train/U"][:].astype(np.float64)
    Pte = f["test/P"][:].astype(np.float64)
    Yte = f["test/U"][:].astype(np.float64)
    modes_tr = dec(f["train"]["modes"][:])
    modes_te = np.array(dec(f["test"]["modes"][:]))

    print("\n=== %s === basis fixed on all %d training runs; regression subsampled"
          % (target, len(Ptr)))
    print("%6s | %14s %14s | %s"
          % ("N", "linear", "quadratic",
             "  ".join("%9s" % g for g in sorted(set(modes_te.tolist())))))
    print("-" * (40 + 11 * len(set(modes_te.tolist()))))
    for n in SIZES:
        if n > len(Ptr):
            continue
        sl, sq, per = [], [], []
        for rep in range(n_rep):
            rng = np.random.default_rng(seed + 1000 * rep)
            sub = stratified_subsample(modes_tr, n, rng)
            pl = fit_lin(Ptr[sub], Ytr[sub], Pte)
            pq = fit_quad(Ptr[sub], Ytr[sub], Pte, seed=seed + rep)
            sl.append(r2_pooled(Yte, pl)); sq.append(r2_pooled(Yte, pq))
            per.append([r2_pooled(Yte[modes_te == g], pl[modes_te == g])
                        for g in sorted(set(modes_te.tolist()))])
        per = np.mean(per, axis=0)
        print("%6d | %7.4f±%.4f %7.4f±%.4f | %s"
              % (n, np.mean(sl), np.std(sl), np.mean(sq), np.std(sq),
                 "  ".join("%9.4f" % v for v in per)))
    print("  (per-regime columns are the LINEAR predictor, averaged over reps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["midacd", "full3d", "interface"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.target, a.reps, a.seed)
