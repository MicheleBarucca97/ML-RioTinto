# Where the run data lives

Alucell run directories are solver inputs and outputs, not source, so they are
kept outside this repository and excluded by `.gitignore`.

| what | where |
|---|---|
| geometry pilot, 12 geometries | `~/alucell_runs/geometry_pilot/` (`g01`…`g12`, 3.1 GB) |
| alumina feeding, local | `~/alumina_meshtest/`, `~/alumina_noise/`, `~/alumina_level3/` |
| alumina feeding, cluster | `jed:/work/gr-pi/alu-data/CNG/CNG2/level3/` |
| campaign master dataset | `../report_ML/master_ml.h5` with `manifest.csv` |
| Alucell case tree | `/home/barucca/alucell_gitlab/alu-data/` |

The two meshes are not interchangeable: `alu-data_master` carries 261 402 nodes
and the campaign carries 401 838. A velocity written for one aborts on the other
with `ensight_vari FATAL wrong pernod/perell`. See `.claude/solver.md`.

## Regenerating the relabelling dumps

`analysis/geometry/node_position_map.py` and `divergence_cross_geometry.py` read
`ASCII_cuveb_nodes_initial` from each geometry directory. Those dumps are written
by the `export_initial_cuveb` gate in `stat_dataset/stationary.mac` and are not
kept between campaigns, so they must be regenerated before either script runs.
Both take the run root from `GEOM_ROOT`, defaulting to
`~/alucell_runs/geometry_pilot`.
