"""
campaign.py — plan and run a batch of pot simulations.

Workflow
--------
    # 1. Build the plan (writes manifest.csv + creates folders + CSVs)
    python run_campaign_cluster.py plan --out runs/ --template stat_dataset/ \
        --mix uniform=1,gaussian=700,gradient=119,cluster=90,single=40,dead=50

    # 2. Launch — fires every pending job in the background and exits immediately
    python run_campaign_cluster.py run --out runs/

    # 3. Check progress any time (refreshes states from exit markers)
    python run_campaign_cluster.py status --out runs/

Each run lives in its own folder (./stat_0001, ...) with:
    anode_data.csv   <- generated current fractions
    status.json      <- {state, pid, started, finished, returncode, error}
    run.log          <- captured stdout+stderr from the simulation
    .exit_code       <- written by the wrapper when the job finishes
    (sim outputs written by your solver)

The manifest.csv at the campaign root is the plan: one row per run with
its id, mode, seed, folder. It is written ONCE by `plan` and never
modified by parallel workers — workers only touch their own status.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

# ─────────────────────────────────────────────────────────────────────────────
# Current-distribution generators
# ─────────────────────────────────────────────────────────────────────────────
N_ANODES = 24
I_TOTAL = 490_000.0           # A
I_MIN, I_MAX = 16_400.0, 24_400.0
I_MEAN = I_TOTAL / N_ANODES   # ≈ 20 416 A


def _truncnorm(loc, scale, size, rng, lo=I_MIN, hi=I_MAX):
    """Truncated normal about `loc`, bounded by [lo, hi].

    The bounds must be passed relative to the caller's `loc`: with the old
    signature a call like _truncnorm(0, 150, ...) computed a = I_MIN/150 = 109
    and sampled the far tail, returning ~I_MIN for every element instead of
    jitter about zero.
    """
    a, b = (lo - loc) / scale, (hi - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=rng)


def _rebalance(vals: np.ndarray, total: float) -> np.ndarray:
    """Affinely rescale `vals` so they sum to `total` while staying in [I_MIN,I_MAX].
    Iterates because clipping breaks the sum. Raises if it can't converge."""
    for _ in range(50):
        s = vals.sum()
        if abs(s - total) < 1e-6:
            return vals
        vals = vals + (total - s) / len(vals)
        vals = np.clip(vals, I_MIN, I_MAX)
    raise RuntimeError("rebalance failed — target outside feasible range")


def _rebalance_free(vals: np.ndarray, total: float, pinned) -> np.ndarray:
    """Rescale only the entries NOT in `pinned` so the sum reaches `total`.

    `_rebalance` clips every entry to [I_MIN, I_MAX], which silently drags a
    deliberately out-of-box anode (a weak one) back to I_MIN and pulls an
    anode pinned at a bound away from it. Here the pinned anodes keep their
    value and the free ones absorb the difference.
    """
    vals = np.asarray(vals, dtype=float).copy()
    free = np.setdiff1d(np.arange(len(vals)), np.atleast_1d(pinned))
    if len(free) == 0:
        raise RuntimeError("rebalance_free: every anode is pinned")
    for _ in range(200):
        s = vals.sum()
        if abs(s - total) < 1e-6:
            return vals
        vals[free] = np.clip(vals[free] + (total - s) / len(free),
                             I_MIN, I_MAX)
    raise RuntimeError("rebalance_free failed — target outside feasible range")


def gen_uniform(rng):
    return np.full(N_ANODES, I_MEAN)


def gen_gaussian(rng, sigma=600.0):
    v = _truncnorm(I_MEAN, sigma, N_ANODES, rng)
    return _rebalance(v, I_TOTAL)


def gen_gradient(rng):
    slope = rng.uniform(-2500, 2500)
    ramp = np.linspace(-slope / 2, slope / 2, N_ANODES)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    v = I_MEAN + ramp + rng.normal(0, 50, N_ANODES)    
    return _rebalance(np.clip(v, I_MIN, I_MAX), I_TOTAL)


def gen_cluster(rng):
    width = rng.integers(2, 5)
    start = rng.integers(0, N_ANODES - width)
    weakness = rng.uniform(2000, 4000)
    v = np.full(N_ANODES, I_MEAN) + rng.normal(0, 50, N_ANODES) 
    v[start:start + width] -= weakness
    return _rebalance(np.clip(v, I_MIN, I_MAX), I_TOTAL)


def gen_single(rng):
    """One anode pinned at a box bound (min or max); the other 23 rebalance.

    Both bounds stay feasible: pinning at I_MAX leaves the rest at a mean of
    20 243 A, at I_MIN at 20 591 A -- both inside [I_MIN, I_MAX].
    """
    idx = int(rng.integers(0, N_ANODES))
    v = _truncnorm(I_MEAN, 150, N_ANODES, rng)
    v[idx] = I_MIN if rng.random() < 0.5 else I_MAX
    return _rebalance_free(v, I_TOTAL, idx)


def gen_dead(rng):
    """1–3 disconnected anodes (current = 0), rest carry the load."""
    n_dead = 1
    dead = rng.choice(N_ANODES, size=n_dead, replace=False)
    active = np.setdiff1d(np.arange(N_ANODES), dead)
    target = I_TOTAL / len(active)
    if not (I_MIN <= target <= I_MAX):
        return gen_gaussian(rng)                     # fall back if infeasible
    v_active = _truncnorm(target, 400, len(active), rng)
    v_active = _rebalance(v_active, I_TOTAL)
    out = np.zeros(N_ANODES)
    out[active] = v_active
    return out

def gen_weak(rng):
    """1-3 weak anodes (low but nonzero current), rest carry the load.

    The weak anodes sit BELOW I_MIN by design, so they must be pinned: a
    plain rebalance clips them back up to I_MIN and the regime disappears.
    """
    n_weak = int(rng.integers(1, 4))
    weak = rng.choice(N_ANODES, size=n_weak, replace=False)
    v = _truncnorm(I_MEAN, 50, N_ANODES, rng)
    v[weak] = rng.uniform(2000, 8000, size=n_weak)
    return _rebalance_free(v, I_TOTAL, weak)


GENERATORS = {
    "uniform":  gen_uniform,
    "gaussian": gen_gaussian,
    "gradient": gen_gradient,
    "cluster":  gen_cluster,
    "single":   gen_single,
    "dead":     gen_dead,
    "weak":     gen_weak,
}

CSV_HEADER = "Anode,Current\n"


def write_currents_csv(path: Path, currents_A: np.ndarray) -> None:
    """Write absolute currents in Amps. The macro will detect sum == cell_current
    and skip the rescale branch — no rounding ambiguity."""
    assert currents_A.shape == (N_ANODES,)
    assert abs(currents_A.sum() - I_TOTAL) < 1e-3
    with path.open("w") as f:
        f.write(CSV_HEADER)
        for i, c in enumerate(currents_A, start=1):
            f.write(f"{i},{c:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plan
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunSpec:
    run_id: int
    mode: str
    seed: int
    folder: str


def parse_mix(s: str) -> dict[str, int]:
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        if k not in GENERATORS:
            sys.exit(f"unknown mode '{k}'. valid: {list(GENERATORS)}")
        out[k] = int(v)
    return out


def cmd_plan(args):
    out_root = Path(args.out)
    template = Path(args.template)
    if not template.is_dir():
        sys.exit(f"template folder not found: {template}")
    out_root.mkdir(parents=True, exist_ok=True)

    mix = parse_mix(args.mix)
    rng = np.random.default_rng(args.seed)

    runs: list[RunSpec] = []
    rid = 1
    for mode, count in mix.items():
        for _ in range(count):
            runs.append(RunSpec(rid, mode, int(rng.integers(0, 2**31 - 1)),
                                f"stat_{rid:04d}"))
            rid += 1

    # Shuffle so failures cluster less by mode (helps spot systematic bugs early)
    rng.shuffle(runs)

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "mode", "seed", "folder"])
        for r in runs:
            w.writerow([r.run_id, r.mode, r.seed, r.folder])

    # Materialize each run folder + CSV
    for r in runs:
        dest = out_root / r.folder
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(template, dest)
        sub_rng = np.random.default_rng(r.seed)
        currents = GENERATORS[r.mode](sub_rng)
        write_currents_csv(dest / "anode_data.csv", currents)
        (dest / "status.json").write_text(json.dumps(
            {"state": "pending", "mode": r.mode, "seed": r.seed}))

    print(f"planned {len(runs)} runs in {out_root}")
    print(f"  mix: {mix}")
    print(f"  manifest: {manifest}")


# ─────────────────────────────────────────────────────────────────────────────
# Run & Status (SLURM Version)
# ─────────────────────────────────────────────────────────────────────────────
def read_status(folder: Path) -> dict:
    p = folder / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"state": "missing"}


def write_status(folder: Path, **kwargs) -> None:
    p = folder / "status.json"
    cur = read_status(folder)
    cur.update(kwargs)
    p.write_text(json.dumps(cur, indent=2))


def slurm_job_alive(job_id: str) -> bool:
    """True if the SLURM job is still in the queue (pending/running)."""
    try:
        # -h removes the header. If stdout is empty, the job is no longer in squeue.
        res = subprocess.run(["squeue", "-j", str(job_id), "-h"], 
                             capture_output=True, text=True)
        return len(res.stdout.strip()) > 0
    except FileNotFoundError:
        return False # squeue command not found


def refresh_state(folder: Path) -> str:
    """Look at status.json + .exit_code + SLURM queue."""
    s = read_status(folder)
    state = s.get("state", "missing")
    if state in ("done", "failed", "pending", "missing"):
        return state

    # state == "running": check the exit-code marker
    exit_file = folder / ".exit_code"
    if exit_file.exists():
        try:
            code = int(exit_file.read_text().strip())
        except ValueError:
            code = -1
        new_state = "done" if code == 0 else "failed"
        write_status(folder, state=new_state, returncode=code, finished=time.time())
        return new_state

    # No marker yet — is the process still alive in SLURM?
    job_id = s.get("slurm_id")
    if job_id and slurm_job_alive(job_id):
        return "running"

    # The job is gone and left no .exit_code.  Alucell writes its own STATUS
    # file, and that is the real ground truth: SLURM reports COMPLETED 0:0 for
    # a job whose solver died mid-solve, and only STATUS records it.  Without
    # this branch every finished run is mislabelled "SLURM job vanished".
    status_file = folder / "STATUS"
    if status_file.exists():
        txt = status_file.read_text().strip()
        verdict = txt.split()[0].upper() if txt else "EMPTY"
        new_state = "done" if verdict == "SUCCESS" else "failed"
        write_status(folder, state=new_state, alucell_status=verdict,
                     finished=time.time())
        return new_state

    write_status(folder, state="failed",
                 error="SLURM job vanished (e.g., timeout, OOM) without writing exit code",
                 finished=time.time())
    return "failed"


def launch_one(folder: Path) -> str:
    """Submit job via sbatch. Returns the SLURM Job ID."""
    exit_marker = folder / ".exit_code"
    if exit_marker.exists():
        exit_marker.unlink()

    # Call sbatch from inside the target folder
    res = subprocess.run(
        ["sbatch", "job.run"],
        cwd=folder,
        capture_output=True,
        text=True,
        check=True
    )
    
    # sbatch output usually looks like: "Submitted batch job 123456"
    job_id = res.stdout.strip().split()[-1]

    write_status(folder, state="running", slurm_id=job_id, started=time.time())
    return job_id


def cmd_run(args):
    out_root = Path(args.out)
    manifest = out_root / "manifest.csv"
    if not manifest.exists():
        sys.exit("no manifest — run `plan` first")

    rows = list(csv.DictReader(manifest.open()))
    todo = []
    
    for row in rows:
        folder = out_root / row["folder"]
        st = refresh_state(folder)
        if st in ("done", "running"):
            continue
        if st == "failed" and not args.retry_failed:
            continue
        todo.append(folder)

    if args.limit:
        todo = todo[:args.limit]

    if not todo:
        print("nothing to launch")
        return

    print(f"submitting {len(todo)} jobs to SLURM…")
    for folder in todo:
        job_id = launch_one(folder)
        print(f"  {folder.name}  slurm_id={job_id}")
    print("done. orchestrator exits — check progress with `status`.")

# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────
def cmd_status(args):
    out_root = Path(args.out)
    manifest = out_root / "manifest.csv"
    if not manifest.exists():
        sys.exit("no manifest")
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "missing": 0}
    by_mode: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(manifest.open()):
        st = refresh_state(out_root / row["folder"])
        counts[st] = counts.get(st, 0) + 1
        by_mode.setdefault(row["mode"], {}).setdefault(st, 0)
        by_mode[row["mode"]][st] = by_mode[row["mode"]].get(st, 0) + 1
    print("overall:", counts)
    print("by mode:")
    for mode, c in by_mode.items():
        print(f"  {mode:10s} {c}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan")
    sp.add_argument("--out", required=True)
    sp.add_argument("--template", required=True)
    sp.add_argument("--mix", required=True,
                    help="e.g. gaussian=200,gradient=200,cluster=100")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run")
    sp.add_argument("--out", required=True)
    sp.add_argument("--limit", type=int, default=0,
                    help="launch at most this many pending jobs (0 = all)")
    sp.add_argument("--retry-failed", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
