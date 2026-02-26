Strucure of the folder:
'''
benchmarks/
  __init__.py        ← get_benchmark(cfg) registry
  benchmark_1d.py    ← 1-D generation
  benchmark_3d.py    ← 3-D generation (vectorised + test split added)
generate.py          ← 4-line dispatcher, reads cfg["benchmark"]
config_1d.yaml       ← benchmark-specific configs
config_3d.yaml
dataset.py           ← unchanged (schema is the same for both)
models.py            ← spatial_dim-aware throughout
utils.py             ← build_model + load_x_grid (single source of truth)
train.py             ← benchmark-agnostic
evaluate.py          ← 1-D: curve plots; 3-D: z=0 slice colour maps
'''

How to use the folder:
'''
python generate.py --config config_1d.yaml --plot
python generate.py --config config_3d.yaml

python train.py    --config config_1d.yaml
python train.py    --config config_3d.yaml

python evaluate.py --config config_1d.yaml
python evaluate.py --config config_3d.yaml
'''
