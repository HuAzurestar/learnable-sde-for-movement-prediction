"""PIRC-22 terrain representation and capacity benchmark contracts."""

from .conditioners import (
    CONDITIONER_SPECS,
    ConditionerError,
    TerrainConditioner,
    TrainingConfig,
    build_conditioner,
    fit_conditioner,
    load_conditioner,
    predict_conditioner,
)
from .consumer import (
    BenchmarkConsumerError,
    build_pirc17_ablation_configs,
    load_benchmark_selection_binding,
)
from .representations import (
    DEFAULT_MATRIX_PATH,
    RepresentationCandidate,
    RepresentationMatrix,
    RepresentationMatrixError,
    load_representation_matrix,
)
from .runner import BenchmarkRunner, BenchmarkRunnerError, PreparedBenchmark
from .selection import (
    BenchmarkSelectionError,
    build_benchmark_selection,
    select_benchmark,
    write_benchmark_selection,
)

__all__ = [
    "CONDITIONER_SPECS",
    "DEFAULT_MATRIX_PATH",
    "BenchmarkRunner",
    "BenchmarkRunnerError",
    "BenchmarkConsumerError",
    "BenchmarkSelectionError",
    "ConditionerError",
    "RepresentationCandidate",
    "RepresentationMatrix",
    "RepresentationMatrixError",
    "PreparedBenchmark",
    "TerrainConditioner",
    "TrainingConfig",
    "build_conditioner",
    "build_benchmark_selection",
    "build_pirc17_ablation_configs",
    "fit_conditioner",
    "load_conditioner",
    "load_representation_matrix",
    "load_benchmark_selection_binding",
    "predict_conditioner",
    "select_benchmark",
    "write_benchmark_selection",
]
