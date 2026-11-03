"""Build a local reviewable config from explicit metadata; performs no registration.

python scripts/build_walking_pilot_config.py --inventory ... --grants ...
    --geometry ... --runtime-root ... --store-id ... --output ...
Fill the independent qualification and new-study artifact grant before freezing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from application.walking_pilot import ARMS,cells,code_identity
from application.walking_pilot_sources import sources_from_metadata
from infrastructure.research_store import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("inventory","grants","geometry","runtime-root","store-id","output"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    def load(path):
        p=Path(path).absolute(); raw=p.read_bytes()
        return json.loads(raw),{"path":str(p),"sha256":hashlib.sha256(raw).hexdigest()}
    inventory,ir=load(args.inventory); grants,gr=load(args.grants); geometry,mr=load(args.geometry)
    config={"schema_version":"pirc26-walking-config-v1","source_mode":"registered-artifacts",
            "pilot_id":"pirc26-walking-20261027-v1","study_id":inventory["study_id"],
            "runtime_root":str(Path(args.runtime_root).absolute()),"store_id":args.store_id,
            "motion":"walking","final_test":False,"speed_limit":inventory["walking_speed_assumption"]["U_m_s"],
            "preparation_arm":"A-S",
            "speed_basis":inventory["walking_speed_assumption"]["basis"],"speed_frozen_at":grants["registered_at"],
            "source_inventory":ir,"source_grants":gr,"source_geometry":mr,
            "sources":sources_from_metadata(inventory,grants,geometry),"train_steps":100,
            "arms":{a:("pirc26-m0-o1" if a=="M0-UO2" else "pirc26-m2-o1" if a=="M2-UO2" else "pirc26-walking-"+a.lower()) for a in ARMS},
            "qualification":{"manifest_id":"REPLACE_WITH_INDEPENDENT_QUALIFICATION","sha256":"0"*64},
            "continuation_approval":{"manifest_id":"REPLACE_WITH_EXACT_OWNER_CONTINUATION_APPROVAL","sha256":"0"*64},
            "artifact_grant":{"authorization_id":"REPLACE_WITH_NEW_STUDY_INTERNAL_GRANT","version":"v1"}}
    policy={"schema_version":"pirc26-walking-budget-v1","pilot_id":config["pilot_id"],"study_id":config["study_id"],
            "global_ms":1800000,"training_ms":120000,"expires_at":inventory["expires_at"],"motion":"walking",
            "final_test":False,"speed_limit":config["speed_limit"],"arms":config["arms"],"config_hash":digest(config),
            "approval":{"manifest_id":grants["evidence_manifest_id"],"sha256":grants["evidence_hash"]}}
    out=Path(args.output)
    out.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    out.with_suffix(".registration-template.json").write_text(json.dumps({"policy":policy,"cells":cells(config),
        "code_hash":code_identity(),"data_hash":digest(config["sources"]),
        "protocol_hash":digest({"inventory":ir["sha256"],"geometry":mr["sha256"]}),
        "note":"Recompute config_hash after filling qualification/grant; preserve original M0/M2 registered family tuples; do not register until independent admission."},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"config":str(out),"source_count":len(config["sources"]),"payload_reads":0}))


if __name__=="__main__":
    main()
