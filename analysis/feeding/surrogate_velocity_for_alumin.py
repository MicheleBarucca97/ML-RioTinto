"""Write the surrogate's predicted velocity in the form the alumina module reads.

This is the substitution that makes Gate 4 a real test rather than a proxy: the
alumina model is run twice on the same current vector, once on the solver's flow and
once on the network's prediction of it, with everything else identical.  The earlier
probe perturbed the flow by a smooth random field, which turned out to act like an
added circulation and told us about mixing rather than about surrogate error.

Two details matter.  The surrogate lives on the fluid submesh -- metal and bath -- while
the alumina module reads a velocity on the whole cell mesh; the velocity is exactly
zero at every non-fluid node (verified over the campaign), so the prediction is
embedded into a zero array.  And the file must be byte-compatible with what alucell's
`write` emits, because gen/read_tr.mac locates the record counts by token position.
"""

import sys as _sys, pathlib as _pathlib
# Studies live in analysis/<topic>/ but use the pipeline modules at the repo root
# and the shared modules in analysis/common/, so both go on the path. Keeps
# `python analysis/<topic>/x.py` working, with data paths relative to the cwd.
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "analysis" / "common")]

import argparse
import sys

import h5py
import numpy as np
import yaml

H5 = "data/full3d_pod_delta.h5"
CFG = "configs/config_alucell_full3d.yaml"
N_CUVEB = 401838


def write_ascii(path, name, A):
    with open(path, "w") as f:
        f.write("\n Name_of br dataset     %s\n" % name)
        f.write(" Number_of_nodes       %11d\n" % len(A))
        f.write(" Number_of_coordinates %11d\n\n" % A.shape[1])
        for i, row in enumerate(A, 1):
            f.write("x_y%8d" % i + "".join("%25.15E" % v for v in row) + "\n")


def main(run, out):
    import torch
    from utils import build_model, resolve_checkpoint

    f = h5py.File(H5, "r")
    rec = f["reconstruction"]
    V = rec["V"][:].astype(np.float64)
    base = rec["u_pod_mean"][:].astype(np.float64) + rec["u_ref"][:].astype(np.float64)
    fid = rec["fluid_node_ids"][:]
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]

    split = row = None
    for s in ("train", "val", "test"):
        ids = dec(f[s]["run_id"][:])
        if run in ids:
            split, row = s, ids.index(run)
            break
    if split is None:
        sys.exit("run %s is not in the reduced dataset" % run)
    p = f[split]["P"][row].astype(np.float32)
    print("run %s found in the %s split" % (run, split))

    cfg = yaml.safe_load(open(CFG))
    m = build_model(cfg)
    m.load_state_dict(torch.load(resolve_checkpoint(cfg), map_location="cpu",
                                 weights_only=True))
    m.eval()
    with torch.no_grad():
        c = m(torch.tensor(p[None, :]), None).numpy()[0].astype(np.float64)

    u_fluid = (V @ c + base).reshape(-1, 3)
    u_true_coef = f[split]["U"][row].astype(np.float64)
    u_truth = (V @ u_true_coef + base).reshape(-1, 3)
    err = np.linalg.norm(u_fluid - u_truth) / np.linalg.norm(u_truth)
    print("prediction vs the POD-truncated truth: relative L2 = %.5f" % err)

    full = np.zeros((N_CUVEB, 3))
    full[fid] = u_fluid
    write_ascii(out, "cuveb_vitesse_in", full)
    print("wrote %s  (%d rows, %d non-zero)" % (out, len(full), int((full != 0).any(1).sum())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="stat_0015")
    ap.add_argument("--out", default="/tmp/ASCII_cuveb_vitesse_in_surrogate")
    a = ap.parse_args()
    main(a.run, a.out)
