"""
compare_baselines.py — score a linear map, a quadratic map and the trained
network on ONE test split with ONE metric.

Why this exists
---------------
The trained surrogate and the campaign coverage analysis report numbers that
sound comparable and are not: the surrogate's R^2 is a per-sample SPATIAL R^2
on the RECONSTRUCTED field, whose variance is dominated by the reference flow
that delta-learning supplies for free, while the coverage analysis reports the
held-out share of BETWEEN-RUN variance explained.  A network can therefore look
excellent on the first and add nothing over a 24 x k matrix on the second.

This script removes the ambiguity: same test runs, same three metrics, three
predictors.  It needs no retraining — the two regressions take seconds and the
network only runs inference.

    python compare_baselines.py --config config_alucell.yaml
    python compare_baselines.py --config config_alucell.yaml --model models/best_model_ResFFNN.pth

What to conclude
----------------
  linear ~ network      the network is not earning its parameters; ship the
                        matrix, which is interpretable and trivially invertible
  quadratic ~ network   the nonlinearity is pairwise; a polynomial is enough
  network > quadratic   the network earns its keep, and by a stated margin
"""

import argparse
from collections import defaultdict

import h5py
import numpy as np
import torch
import yaml

from dataset import (GaussianDataset, load_reconstruction_context,
                     reconstruct_field, load_test_modes)
from utils import build_model, resolve_checkpoint


# ---------------------------------------------------------------------------
# Regression baselines
# ---------------------------------------------------------------------------

#: Singular values below this fraction of the largest are discarded.  The anode
#: currents sum to a fixed total, so the design matrix is exactly rank
#: deficient; keeping that direction inverts numerical noise and produces huge,
#: meaningless coefficients (predictions barely change, but the fit is junk).
_RCOND = 1e-6


def _pairwise(Z):
    """[N, d] -> [N, d(d+1)/2]: the products z_i z_j, i <= j."""
    d = Z.shape[1]
    iu = np.triu_indices(d)
    return (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]


def _hier_ridge(D_lin, D_quad, y, alpha):
    """Fit [linear | quadratic] jointly, penalising the quadratic block only.

    A single ridge over both blocks shrinks the linear coefficients too, so a
    strongly regularised fit tends to zero rather than to the linear solution
    and can score worse than plain linear.  Penalising only the quadratic block
    means alpha -> infinity recovers the linear fit exactly, so the comparison
    is monotone by construction.
    """
    D = np.hstack([D_lin, D_quad])
    lam0 = 1e-6 * max(len(D), 1)
    P = np.concatenate([np.full(D_lin.shape[1], lam0),
                        np.full(D_quad.shape[1], alpha)])
    return np.linalg.solve(D.T @ D + np.diag(P), D.T @ y)


def fit_linear(P_tr, Y_tr, P_te):
    beta, *_ = np.linalg.lstsq(P_tr, Y_tr, rcond=_RCOND)
    return P_te @ beta, beta.size


def fit_quadratic(P_tr, Y_tr, P_te, seed=0, n_keep=40,
                  alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e6, 1e8)):
    """Screened, per-target ridge on the pairwise products.

    Two per-target choices, both made on a validation slice carved out of the
    training runs so the test split stays untouched:

      * WHICH products.  Plain ridge over all d(d+1)/2 of them cannot recover a
        sparse nonlinearity — it shrinks every coefficient equally and never
        concentrates weight, so one true product among ~300 collinear features
        is buried.  Screening by correlation with the linear residual finds it.
        The full basis is offered as well, because a dense quadratic form (any
        energy-like quantity is one) would be truncated by screening.
      * HOW MUCH regularisation, since targets differ wildly in how much
        quadratic signal they carry.
    """
    n_tr, m = Y_tr.shape[0], Y_tr.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tr)
    n_val = max(5, int(0.2 * n_tr))
    va, tr2 = perm[:n_val], perm[n_val:]

    Q_tr, Q_te = _pairwise(P_tr), _pairwise(P_te)
    mu, sd = Q_tr.mean(0), Q_tr.std(0)
    sd = np.where(sd < 1e-15, 1.0, sd)
    Q_tr, Q_te = (Q_tr - mu) / sd, (Q_te - mu) / sd

    b_in, *_ = np.linalg.lstsq(P_tr[tr2], Y_tr[tr2], rcond=_RCOND)
    resid_in = Y_tr[tr2] - P_tr[tr2] @ b_in
    beta, *_ = np.linalg.lstsq(P_tr, Y_tr, rcond=_RCOND)
    resid_tr = Y_tr - P_tr @ beta

    score_in = Q_tr[tr2].T @ resid_in / max(len(tr2), 1)
    score_tr = Q_tr.T @ resid_tr / max(n_tr, 1)
    n_feat = Q_tr.shape[1]
    cands = [min(n_keep, n_feat)] + ([n_feat] if n_feat > n_keep else [])

    pred = np.empty((P_te.shape[0], m))
    n_coef = 0
    for k in range(m):
        o_in = np.argsort(np.abs(score_in[:, k]))[::-1]
        o_tr = np.argsort(np.abs(score_tr[:, k]))[::-1]
        best, best_cfg = -np.inf, (cands[0], alphas[-1])
        for n_sel in cands:
            sel = o_in[:n_sel]
            D_va = np.hstack([P_tr[va], Q_tr[np.ix_(va, sel)]])
            for a in alphas:
                W = _hier_ridge(P_tr[tr2], Q_tr[np.ix_(tr2, sel)],
                                Y_tr[np.ix_(tr2, [k])], a)
                r = _r2_flat(Y_tr[np.ix_(va, [k])], D_va @ W)
                if np.isfinite(r) and r > best:
                    best, best_cfg = r, (n_sel, a)
        n_sel, a = best_cfg
        sel = o_tr[:n_sel]
        W = _hier_ridge(P_tr, Q_tr[np.ix_(np.arange(n_tr), sel)],
                        Y_tr[:, [k]], a)
        pred[:, k] = (np.hstack([P_te, Q_te[np.ix_(np.arange(len(P_te)), sel)]])
                      @ W)[:, 0]
        n_coef += P_tr.shape[1] + n_sel
    return pred, n_coef


# ---------------------------------------------------------------------------
# Reduction for raw-field targets
# ---------------------------------------------------------------------------

#: Above this many target columns the regressions are fitted in a POD basis
#: instead of column by column.
MAX_DIRECT_TARGETS = 200


def maybe_reduce(Y_tr, Y_te, k):
    """Work in a POD basis when the stored target is a raw field.

    Datasets prepared without --pod store the field itself, so a per-target
    quadratic fit would mean tens of thousands of regressions.  It would also
    be the wrong comparison: POD_MLP reduces the field internally, so the fair
    baseline reduces it too, with the same number of modes.

    The basis is computed on the TRAINING split only.  Scoring is done against
    the raw field afterwards, so the truncation error is charged to the
    baselines exactly as it is charged to POD_MLP.

    Returns (C_tr, C_te, basis) with basis = (V, mean) or None if no reduction
    was needed.
    """
    if Y_tr.shape[1] <= MAX_DIRECT_TARGETS:
        return Y_tr, Y_te, None

    k = min(k, Y_tr.shape[0] - 1, Y_tr.shape[1])
    mean = Y_tr.mean(axis=0)
    Uc = Y_tr - mean
    G = (Uc @ Uc.T) / max(len(Uc) - 1, 1)
    w, Q = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:k]
    V = Uc.T @ Q[:, idx]
    V /= np.linalg.norm(V, axis=0, keepdims=True).clip(min=1e-12)

    evr = np.maximum(w[idx], 0.0).sum() / max(np.maximum(w, 0.0).sum(), 1e-30)
    print(f"  raw field with {Y_tr.shape[1]} columns -> POD basis with {k} "
          f"modes ({100*evr:.2f}% of variance)")
    return Uc @ V, (Y_te - mean) @ V, (V, mean)


def expand(C, basis):
    """Coefficients back to field space (identity if no reduction was used)."""
    if basis is None:
        return C
    V, mean = basis
    return C @ V.T + mean


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _r2_flat(Y_true, Y_pred):
    """Dataset-level R^2 over every entry, not per sample."""
    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-30))


def score(name, raw_pred, raw_true, ctx, n_coef, modes):
    """Metrics for one predictor, in both spaces.

    r2_delta is computed in MODEL-OUTPUT space, i.e. on the perturbation (or
    its POD coefficients).  Because the POD basis is orthonormal, the R^2 over
    the coefficients equals the R^2 over the reconstructed perturbation field,
    so this is directly comparable to the coverage analysis's variance-weighted
    number.  It is the metric that answers "how much of the run-to-run
    variation did this predictor capture", and the one a model that returns the
    reference for every input would score 0 on.

    rel_l2 is on the reconstructed physical field, which is what an engineer
    asks for — but its denominator contains the invariant reference flow, so it
    flatters every predictor and must not be read alone.
    """
    r2_delta = _r2_flat(raw_true, raw_pred)

    rec_p = reconstruct_field(raw_pred, ctx)
    rec_t = reconstruct_field(raw_true, ctx)
    rmse = np.sqrt(np.mean((rec_p - rec_t) ** 2, axis=1))
    rel = (np.linalg.norm(rec_p - rec_t, axis=1)
           / (np.linalg.norm(rec_t, axis=1) + 1e-12))

    by_mode = {}
    if modes:
        g = defaultdict(list)
        for i, mo in enumerate(modes):
            g[mo].append(i)
        by_mode = {mo: float(rel[idx].mean()) for mo, idx in sorted(g.items())}

    return {"name": name, "r2_delta": r2_delta, "rmse": rmse.mean(),
            "rel_mean": rel.mean(), "rel_std": rel.std(), "rel_max": rel.max(),
            "n_coef": n_coef, "by_mode": by_mode}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path, model_path, seed, h5_override=None,
         type_override=None, modes_override=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Overrides so one config serves all three mappings: they differ only in
    # the dataset, the model type and the number of modes.
    if h5_override:
        cfg["data"]["h5_path"] = h5_override
    if type_override:
        cfg["model"]["type"] = type_override
    if modes_override:
        cfg["model"]["n_modes"] = modes_override
    h5_path = cfg["data"]["h5_path"]
    normalize = cfg["data"].get("normalize", False)

    with h5py.File(h5_path, "r") as f:
        P_tr = f["train"]["P"][:].astype(np.float64)
        Y_tr = f["train"]["U"][:].astype(np.float64)
        P_te = f["test"]["P"][:].astype(np.float64)
        Y_te = f["test"]["U"][:].astype(np.float64)
    ctx = load_reconstruction_context(h5_path)
    modes = load_test_modes(h5_path)

    n_modes = cfg["model"].get("n_modes", 50)

    print(f"Config    : {config_path}")
    print(f"Data      : {h5_path}")
    print(f"Train/test: {len(P_tr)} / {len(P_te)} runs")
    print(f"Target    : {Y_tr.shape[1]} outputs "
          f"({'POD coefficients' if ctx['V'] is not None else 'raw field'})")
    print(f"Delta     : {ctx['delta_learning']}\n")

    # Raw-field targets are reduced first; coefficient targets pass through.
    C_tr, C_te, basis = maybe_reduce(Y_tr, Y_te, n_modes)

    results = []
    print("Fitting linear map …")
    pred_lin, n_lin = fit_linear(P_tr, C_tr, P_te)
    results.append(score("linear", expand(pred_lin, basis), Y_te, ctx, n_lin, modes))

    print("Fitting quadratic map …")
    pred_quad, n_quad = fit_quadratic(P_tr, C_tr, P_te, seed=seed)
    results.append(score("quadratic", expand(pred_quad, basis), Y_te, ctx,
                         n_quad, modes))

    # --- trained network ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = cfg["model"]["type"]
    model_path = model_path or resolve_checkpoint(cfg)
    try:
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
        model.eval()
        n_nn = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ds = GaussianDataset(h5_path, split="test", normalize=normalize)
        with h5py.File(h5_path, "r") as f:
            x_grid = torch.from_numpy(f["x_grid"][:]).to(device)
        P_in = torch.from_numpy(P_te.astype(np.float32)).to(device)
        if normalize:
            P_in = (P_in - ds.p_mean.to(device)) / ds.p_std.to(device)
        with torch.no_grad():
            out = []
            for i in range(0, len(P_in), 16):
                out.append(model(P_in[i:i + 16], x_grid).cpu().numpy())
        pred_nn = np.vstack(out).astype(np.float64)
        if normalize:
            pred_nn = pred_nn * ds.u_std.numpy() + ds.u_mean.numpy()
        print(f"Loaded network from {model_path}")
        results.append(score(f"network ({tag})", pred_nn, Y_te, ctx, n_nn, modes))
    except FileNotFoundError:
        print(f"[skip] no checkpoint at {model_path} — regressions only")

    # --- report ---
    print("\n" + "=" * 78)
    print(f"{'predictor':22s} {'R2 (perturbation)':>18s} {'RMSE':>12s} "
          f"{'rel-L2 (field)':>16s} {'coefs':>8s}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:22s} {r['r2_delta']:18.4f} {r['rmse']:12.4e} "
              f"{r['rel_mean']:10.4f} ± {r['rel_std']:.4f} {r['n_coef']:8d}")
    print("=" * 78)
    print("R2 (perturbation) is the column to compare: it is the share of the")
    print("run-to-run variation captured, and equals the coverage analysis's")
    print("variance-weighted R2.  rel-L2 has the invariant reference flow in its")
    print("denominator and flatters every row, so it is reported but not ranked.")

    if any(r["by_mode"] for r in results):
        print("\nrel-L2 by campaign regime:")
        all_modes = sorted({m for r in results for m in r["by_mode"]})
        print(f"  {'predictor':22s} " + "".join(f"{m:>11s}" for m in all_modes))
        for r in results:
            row = "".join(f"{r['by_mode'].get(m, float('nan')):11.5f}"
                          for m in all_modes)
            print(f"  {r['name']:22s} {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--h5", default=None, help="override data.h5_path")
    ap.add_argument("--model-type", default=None,
                    help="override model.type (e.g. POD_MLP)")
    ap.add_argument("--n-modes", type=int, default=None,
                    help="override model.n_modes; also sets the number of POD "
                         "modes the baselines use for a raw-field target")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.config, a.model, a.seed, a.h5, a.model_type, a.n_modes)
