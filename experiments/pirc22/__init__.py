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
from .representations import (
    DEFAULT_MATRIX_PATH,
    RepresentationCandidate,
    RepresentationMatrix,
    RepresentationMatrixError,
    load_representation_matrix,
)

__all__ = [
    "CONDITIONER_SPECS",
    "DEFAULT_MATRIX_PATH",
    "ConditionerError",
    "RepresentationCandidate",
    "RepresentationMatrix",
    "RepresentationMatrixError",
    "TerrainConditioner",
    "TrainingConfig",
    "build_conditioner",
    "fit_conditioner",
    "load_conditioner",
    "load_representation_matrix",
    "predict_conditioner",
]
