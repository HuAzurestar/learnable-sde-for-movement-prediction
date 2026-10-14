"""Ten terrain configurations with history retained in every baseline."""
from __future__ import annotations

from copy import copy
import hashlib
import json

import numpy as np

from experiments.nex326.pirc21_adapter import FeatureSelection
from experiments.pirc22.consumer import COMPOSITION_DEPENDENCIES, FACTOR_GROUPS, load_benchmark_selection_binding
from .features import CanonicalEncoder


def terrain_configurations():
    binding=load_benchmark_selection_binding()
    selected=binding["selected_configuration"]
    factors=("road","river","worldcover","surface")
    groups={"base":(),"all-terrain":factors}
    groups.update({"loo-"+f:tuple(g for g in factors if g!=f) for f in factors})
    groups.update({"lio-"+f:(f,) for f in factors})
    output={}
    for name,included in groups.items():
        keep={v for f in (*included,"history") for v in FACTOR_GROUPS[f]}
        variants=tuple(v for v in selected["variant_ids"] if v in keep)
        compositions=tuple(c for c in selected["composition_ids"] if set(COMPOSITION_DEPENDENCIES[c])<=set(variants))
        payload={"schema_version":"pirc17-history-preserving-terrain-config-v1","configuration_id":name,
            "terrain_factors":list(included),"baseline_history":"retained_in_every_configuration",
            "variant_ids":list(variants),"composition_ids":list(compositions),
            "pirc22_consumer_identity":binding["consumer_identity_sha256"]}
        payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        output[name]=payload
    return output


def configuration_encoder(full: CanonicalEncoder, name):
    """Reuse the already verified snapshot, changing only its feature selection."""
    selected=terrain_configurations()[name]
    if not set(selected["variant_ids"])<=set(full.adapter.selection.variant_ids) or not set(selected["composition_ids"])<=set(full.adapter.selection.composition_ids):
        raise ValueError("full encoder does not contain frozen configuration inputs")
    adapter=copy(full.adapter)
    adapter.selection=FeatureSelection(variant_ids=tuple(selected["variant_ids"]),
        composition_ids=tuple(selected["composition_ids"]),include_validity_indicators=True)
    adapter._validate_selection()
    return CanonicalEncoder(adapter)


def subset_matrix(matrix, full_columns, selected_columns):
    values=np.asarray(matrix,dtype=float)
    if values.ndim!=2 or values.shape[1]!=2*len(full_columns) or len(set(full_columns))!=len(full_columns):
        raise ValueError("numeric-plus-validity full matrix and unique columns required")
    indexes=[full_columns.index(c) for c in selected_columns]
    return values[:,indexes+[i+len(full_columns) for i in indexes]]
