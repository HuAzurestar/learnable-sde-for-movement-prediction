"""Pure original population geometry decoding, after parent authorization."""
import math
from infrastructure.research_store import ResearchError

def parse_population(raw,source,lineage):
    if raw is None:
        return {"source_id":source["source_id"],"status":"CONTRACT_REJECTED","segments":[]}
    from application.pirc26_data import _bounded_document
    document=_bounded_document(raw.decode("utf-8"))
    if (document.get("schema_version")!="pirc26-observed-block-v1" or document.get("time_unit")!="s"
            or document.get("state_units")!=["m","m","m/s","m/s"] or not isinstance(document.get("segments"),list)
            or not 1<=len(document["segments"])<=256):
        raise ResearchError("CONTRACT_MISMATCH","original physical population schema differs")
    allowed=set(source.get("segment_ids",[])); selected=[]; seen=set()
    for segment in document["segments"]:
        sid=segment["segment_id"]
        if sid in seen:
            raise ResearchError("CONTRACT_MISMATCH","duplicate original segment")
        seen.add(sid)
        if source["role"]=="train" and sid not in allowed:
            continue
        t,p=segment["time"],segment["position"]
        if (not 3<=len(t)<=4098 or len(t)!=len(p)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in t)
                or any(a>=b for a,b in zip(t,t[1:])) or
                any(not isinstance(row,list) or len(row)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in row) for row in p)):
            raise ResearchError("CONTRACT_MISMATCH","finite native segment geometry required")
        selected.append({"segment_id":sid,"times":t,"positions":p})
    if source["role"]=="train" and {s["segment_id"] for s in selected}!=allowed:
        raise ResearchError("CONTRACT_MISMATCH","pooled artifact is missing original source segments")
    return {"source_id":source["source_id"],"status":"AVAILABLE","segments":selected,"lineage":lineage}
