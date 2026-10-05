"""
Dataset generation entry point.

Reads the 'benchmark' key from the config file and delegates to the
appropriate generator in the benchmarks/ package.

Usage
-----
    python generate.py --config configs/config_1d.yaml
    python generate.py --config configs/config_3d.yaml
    python generate.py --config configs/config_1d.yaml --plot
"""

import argparse
import yaml
from benchmarks import get_benchmark


def main():
    parser = argparse.ArgumentParser(description="Generate a Gaussian benchmark dataset")
    parser.add_argument("--config", required=True,
                        help="Path to a config_*.yaml file")
    parser.add_argument("--plot", action="store_true",
                        help="Preview a few random training samples after generation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    benchmark = cfg.get("benchmark", "1d")
    print(f"Benchmark: {benchmark}  |  config: {args.config}")

    generate_fn = get_benchmark(cfg)
    generate_fn(cfg, plot=args.plot)


if __name__ == "__main__":
    main()
