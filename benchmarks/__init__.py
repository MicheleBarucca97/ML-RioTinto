"""
Benchmark registry.

Usage
-----
    from benchmarks import get_benchmark
    gen_fn = get_benchmark(cfg)   # returns the generate(cfg, plot) callable
    gen_fn(cfg, plot=True)
"""

from benchmarks.benchmark_1d import generate as generate_1d
from benchmarks.benchmark_3d import generate as generate_3d


_REGISTRY = {
    "1d": generate_1d,
    "3d": generate_3d,
}


def get_benchmark(cfg: dict):
    """Return the generation function for the benchmark named in *cfg*.

    Raises KeyError with a helpful message if the name is unknown.
    """
    key = cfg.get("benchmark", "1d")
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{key}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]
