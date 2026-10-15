"""Statistical OL benchmark / reference layer.

Consumes analyze_video() JSON. Does not run pose, tracking, or 3D reconstruction.
"""

from oline_cv.benchmark.builder import build_benchmark
from oline_cv.benchmark.compare import attach_benchmark_comparison, compare_rep
from oline_cv.benchmark.features import extract_features
from oline_cv.benchmark.schema import FEATURE_REGISTRY, SCHEMA_VERSION
from oline_cv.benchmark.storage import load_benchmark, save_benchmark

__all__ = [
    "SCHEMA_VERSION",
    "FEATURE_REGISTRY",
    "extract_features",
    "build_benchmark",
    "compare_rep",
    "attach_benchmark_comparison",
    "load_benchmark",
    "save_benchmark",
]
