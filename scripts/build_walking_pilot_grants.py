"""Emit v2 owner-registration templates from existing grant METADATA only.

No ResearchStore is opened and no grants or budgets are registered. The owner
must register the returned exact versions before using a rebuilt config.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from application.walking_pilot_sources import consumer_scope


def templates(source_document,internal_receipt,pilot_id):
    sources=deepcopy(source_document)
    for grant in sources["grants"]:
        if grant.get("version")!="v1":
            raise ValueError("exact original v1 source grant metadata required")
        grant["version"]="v2"
        if "consumer_scope" in grant:
            grant["consumer_scope_description"]=grant["consumer_scope"]
        grant["consumer_scope"]=consumer_scope(pilot_id)
    internal=deepcopy(internal_receipt["grant"])
    if internal.get("version")!="v1" or internal["pilot_study_id"]!=sources["pilot_study_id"]:
        raise ValueError("matching original v1 internal grant metadata required")
    internal["version"]="v2"
    internal["consumer_scope"]=consumer_scope(pilot_id,internal=True)
    # These old receipt fields describe v1, never a registration of this template.
    for key in ("registered_at","head"):
        if key in sources:
            sources["previous_v1_"+key]=sources.pop(key)
    sources["registration_status"]="TEMPLATE_ONLY_NOT_REGISTERED"
    return sources,internal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-grants","internal-grant","pilot-id","output-directory"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    sources=json.loads(Path(args.source_grants).read_text(encoding="utf-8-sig"))
    internal=json.loads(Path(args.internal_grant).read_text(encoding="utf-8-sig"))
    sources,internal=templates(sources,internal,args.pilot_id)
    output=Path(args.output_directory); output.mkdir(parents=True,exist_ok=True)
    for name,value in (("source-grants.v2.template.json",sources),("internal-grant.v2.template.json",internal)):
        path=output/name
        if path.exists():
            raise FileExistsError("refusing to overwrite an existing owner template: "+str(path))
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"registration_status":"TEMPLATE_ONLY_NOT_REGISTERED","payload_reads":0,"output":str(output)}))


if __name__=="__main__":
    main()
