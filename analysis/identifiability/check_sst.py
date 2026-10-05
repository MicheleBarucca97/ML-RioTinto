import h5py, numpy as np
hdr = ("field", "w=evr (old)", "w=sst (new)", "pooled", "err_old", "err_new")
print("%-10s %14s %14s %12s %10s %10s" % hdr)
for name in ["midacd", "full3d", "interface"]:
    f = h5py.File("data/%s_pod_delta.h5" % name, "r")
    P = np.vstack([f[s]["P"][:] for s in ("train", "val", "test")]).astype(np.float64)
    C = np.vstack([f[s]["U"][:] for s in ("train", "val", "test")]).astype(np.float64)
    evr = f["reconstruction"]["evr"][:].astype(np.float64)
    N, m = C.shape
    rng = np.random.default_rng(0); perm = rng.permutation(N); nt = int(0.25 * N)
    te, tr = perm[:nt], perm[nt:]
    sx = np.where(P[tr].std(0) < 1e-15, 1, P[tr].std(0))
    sy = np.where(C[tr].std(0) < 1e-15, 1, C[tr].std(0))
    Zx = (P - P[tr].mean(0)) / sx
    Zy = (C - C[tr].mean(0)) / sy
    beta, *_ = np.linalg.lstsq(Zx[tr], Zy[tr], rcond=1e-6)
    Yp = Zx[te] @ beta
    sse = ((Zy[te] - Yp) ** 2).sum(0)
    sst_std = ((Zy[te] - Zy[te].mean(0)) ** 2).sum(0)
    r2m = 1 - sse / np.maximum(sst_std, 1e-30)
    sst_test = sst_std * sy ** 2                 # exactly what the patch computes
    w_new = sst_test / sst_test.sum()
    w_old = evr[:m] / evr[:m].sum()
    Cp = Yp * sy + C[tr].mean(0)
    pooled = 1 - ((C[te] - Cp) ** 2).sum() / ((C[te] - C[te].mean(0)) ** 2).sum()
    a = float((w_old * r2m).sum()); b = float((w_new * r2m).sum())
    print("%-10s %14.8f %14.8f %12.8f %+10.2e %+10.2e"
          % (name, a, b, pooled, a - pooled, b - pooled))
