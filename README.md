# Benchmark-Based Training Framework

This repository provides a benchmark-agnostic framework for dataset generation, training, and evaluation.  
Both 1D and 3D benchmarks share the same dataset schema and training pipeline.

---

## 📂 Project Structure

~~~text
benchmarks/
  __init__.py        ← get_benchmark(cfg) registry
  benchmark_1d.py    ← 1-D generation
  benchmark_3d.py    ← 3-D generation (vectorised + test split added)

generate.py          ← 4-line dispatcher, reads cfg["benchmark"]

config_1d.yaml       ← benchmark-specific config (1D)
config_3d.yaml       ← benchmark-specific config (3D)

dataset.py           ← unchanged (schema is the same for both)
models.py            ← spatial_dim-aware throughout
utils.py             ← build_model + load_x_grid (single source of truth)

train.py             ← benchmark-agnostic training
evaluate.py          ← 
                        - 1-D: curve plots
                        - 3-D: z=0 slice colour maps
~~~

---

## 🚀 Usage

### 1️⃣ Generate Dataset

#### 1D Benchmark
~~~bash
python generate.py --config config_1d.yaml --plot
~~~

#### 3D Benchmark
~~~bash
python generate.py --config config_3d.yaml
~~~

---

### 2️⃣ Train Model

#### 1D Benchmark
~~~bash
python train.py --config config_1d.yaml
~~~

#### 3D Benchmark
~~~bash
python train.py --config config_3d.yaml
~~~

---

### 3️⃣ Evaluate Model

#### 1D Benchmark
~~~bash
python evaluate.py --config config_1d.yaml
~~~

#### 3D Benchmark
~~~bash
python evaluate.py --config config_3d.yaml
~~~

---

## 🔎 Notes

- The dataset schema is identical for both benchmarks.
- The training pipeline is fully benchmark-agnostic.
- Spatial dimensionality is handled internally via `spatial_dim`.
- `utils.py` serves as the single source of truth for model construction and grid loading.
