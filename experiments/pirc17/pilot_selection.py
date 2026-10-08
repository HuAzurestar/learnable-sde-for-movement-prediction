"""Deterministic development pilot selection without accessing labels/scores."""

SELECTIONS={
    "lexical_origins":"first validation sample IDs in lexical order, not selected by scores",
    "lexical_independent_blocks":"first lexical validation sample per independent block, then first sample IDs; no score selection",
}


def select_windows(windows, *, count, policy="lexical_origins"):
    if type(count) is not int or count<1 or policy not in SELECTIONS:
        raise ValueError("positive integer pilot count and registered selection policy required")
    if any(w.role!="validation" for w in windows):
        raise ValueError("pilot selection admits validation only")
    ids=[w.sample_id for w in windows]
    if any(not w.sample_id or not w.block_id for w in windows) or len(ids)!=len(set(ids)):
        raise ValueError("unique samples and explicit independent blocks required")
    candidates=sorted(windows,key=lambda w:w.sample_id)
    if policy=="lexical_independent_blocks":
        selected=[]
        blocks=set()
        for window in candidates:
            if window.block_id not in blocks:
                selected.append(window)
                blocks.add(window.block_id)
        candidates=selected
    if len(candidates)<count:
        raise ValueError("requested pilot count exceeds available origins/independent blocks")
    return candidates[:count]
