"""Actual synthetic refused admission; no worker/provider/Paper outcome mocks."""
from application.research_budget import BudgetSpec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, digest
from tests.test_research_upstream_admission import prepared
from tests.test_shared_upstream_snapshot import snapshot_fixture


def recorded_view(root, mutation="missing-file", *, visibility="synthetic"):
    path, manifest, accepted = snapshot_fixture(root)
    if mutation == "unaccepted":
        accepted[0]["status"] = "pending"
    store, spec, registry, grant = prepared(root, formal=True, two_arms=True,
        upstream_inputs=manifest["inputs"], accepted_versions=accepted,
        upstream_dependencies={"affine": ["metadata"], "candidate": []}, upstream_visibility=visibility)
    if mutation == "missing-file":
        path.rename(root / "retained-missing-metadata.json")
    elif mutation == "changed-file":
        path.write_bytes(path.read_bytes().replace(b"accepted-control", b"rejected-control"))
    if mutation == "missing-reference":
        spec["admission"].pop("upstream_snapshot_hash")
    store.register(spec, digest(spec))
    if mutation == "missing-document":
        source = store.path / "manifests" / ("upstream-snapshot-" + spec["admission"]["upstream_snapshot_hash"] + ".json")
        source.rename(root / "retained-missing-snapshot.json")
    if mutation in {"missing-reference", "missing-document"}:
        grant = {**grant, "authorization_id": "explicit-restricted-ui", "visibilities": ["synthetic", "restricted"]}
        store.authorize(grant)
    if mutation != "not-checked":
        try:
            result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(10))
        except ResearchError as exc:
            expected = {"missing-file": "MISSING_ARTIFACT", "changed-file": "IDENTITY_MISMATCH",
                        "unaccepted": "UNACCEPTED_VERSION", "missing-reference": "MISSING_ARTIFACT",
                        "missing-document": "IDENTITY_MISMATCH"}[mutation]
            assert exc.code == expected
        else:
            assert mutation == "ready" and result["state"] == "SUCCEEDED", "actual upstream counterexample was not refused"
    if mutation != "ready":
        assert not any(event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"} for event in store.events())
    return store, spec, grant
