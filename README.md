# A surrogate for the anode currents of a CNG2 aluminium cell

Maps the 24 anode currents of a Hall–Héroult cell to the magnetohydrodynamic
flow it produces, learning the perturbation about a uniform-current reference so
the network predicts only what changes. Built on a campaign of 1594 Alucell
solves, and used to ask what a predicted field is good for once something
downstream consumes it.

The argument and every measurement are in **`docs/phase0_report.pdf`**. This
README is for running the code.

## Layout

```
train.py evaluate.py models.py dataset.py utils.py    the pipeline
prepare_alucell.py compare_baselines.py               data prep, and the controls
configs/           one YAML per experiment
analysis/          one-off studies, grouped by the question they answer
  feeding/         alumina feeding: design, convergence, Hofer's objective
  geometry/        relabelling across geometries, mesh topology, transfer
  functionals/     functionals of a predicted field, divergence audits
  identifiability/ sensitivity operator, active subspace, learning curves
  plotting/        figures for the note
alucell/           solver-side harnesses and campaign drivers
docs/              the note, and where the run data lives
```

Generated directories — `data/`, `models/`, `plots/`, `logs/`, `results/` — are
reproducible from the above and are not versioned.

## Running

Everything runs from the repository root.

```bash
pip install -r requirements.txt

# 1. build a dataset from the campaign
python prepare_alucell.py --master ../report_ML/master_ml.h5 \
    --manifest ../report_ML/manifest.csv \
    --mapping velocity_full --delta --pod --n-modes 64 \
    --include-dead --output data/full3d_pod_delta.h5

# 2. train, evaluate, and score against the linear and quadratic controls
python train.py            --config configs/config_alucell_full3d.yaml
python evaluate.py         --config configs/config_alucell_full3d.yaml
python compare_baselines.py --config configs/config_alucell_full3d.yaml
```

`--mapping` is one of `velocity_full`, `velocity_midacd`, `interface`.

## Reading the metrics

Report them **per regime**, never aggregated. The eight `weak` runs carry about
92% of the test-set variance, so a mean or a standard deviation over the whole
split is dominated by them and says little. `weak`, `single` and `dead` are
anode replacements — scheduled operations, not faults — and they are where a
surrogate is actually consulted.

Two error measures answer different questions and are not interchangeable:
`r2_delta` lives in model-output space, on the perturbation, and says how much of
the physical fluctuation was captured; `rel_l2` lives on the reconstructed field
with the reference added back, and is what an engineer reads.

## Where the rest is

- `CLAUDE.md` — working rules for this project, including the convergence traps.
- `CONVENTIONS.md` — writing conventions for the note.
- `.claude/solver.md` — the Alucell pipeline, its paths and its traps.
- `.claude/measurements.md` — every number established, and the withdrawn ones.
- `docs/RUNS.md` — where the simulation run directories live.
