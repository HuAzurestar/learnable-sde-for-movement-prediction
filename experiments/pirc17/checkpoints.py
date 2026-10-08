"""Validated restoration of complete causal dynamics, not conditioner alone."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.pirc22.conditioners import load_conditioner
from experiments.pirc22.consumer import load_benchmark_selection_binding
from .dynamics import BASE_COLUMNS, DYNAMICS_VERSION, LearnedDynamics
from .origins import frozen_array


def restore_dynamics(identity):
    payload=dict(identity)
    digest=payload.pop("sha256",None)
    actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest!=actual or payload.get("version")!=DYNAMICS_VERSION:
        raise ValueError("dynamics checkpoint version/hash mismatch")
    binding=load_benchmark_selection_binding()
    if payload.get("pirc22_consumer_identity")!=binding["consumer_identity_sha256"] or payload.get("base_columns")!=list(BASE_COLUMNS):
        raise ValueError("dynamics checkpoint input semantics changed")
    conditioner=load_conditioner(payload["conditioner_checkpoint"])
    if payload["conditioner_checkpoint"]["training_config"]!=binding["selected_configuration"]["training_config"]:
        raise ValueError("frozen training budget changed")
    covariance=frozen_array(payload["diffusion_covariance_m2_per_s"],(2,2))
    eigenvalues,vectors=np.linalg.eigh(covariance)
    if not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0) or np.min(eigenvalues)<-1e-10:
        raise ValueError("diffusion covariance is not positive semidefinite")
    return LearnedDynamics(frozen_array(payload["base_weights"],(len(BASE_COLUMNS),2)),conditioner,
        payload["conditioner_checkpoint"],frozen_array(vectors@np.diag(np.sqrt(np.maximum(eigenvalues,0)))),dict(identity))
