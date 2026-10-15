"""Original-arm derived jobs cannot escape the owner's aggregate4H cap."""
import importlib.util
from pathlib import Path
import sys


def operator():
    path = Path(__file__).resolve().parents[1]/"scripts/pirc27_affine_continuation.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("pirc27_accounting_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Metadata:
    def __init__(self, events):
        self.entries = events

    def manifest(self, name):
        assert name == operator().original.ORIGINAL
        return {"arms": [{"arm_id": "pirc27-affine-exact-mean-x"}]}

    def events(self):
        return self.entries


def event(kind, reservation, *, study="managed-comparison-hash", arm="pirc27-affine-exact-mean-x",
          charged=0, reserved=300000, settled=False):
    return {"event_kind": kind, "payload": {"reservation_id": reservation,
        "study_id": study, "arm_id": arm, "charged_ms": charged,
        "reserved_ms": reserved, "settled": settled}}


def test_nonprefixed_derived_comparison_charged_to_original_arm_counts():
    store = Metadata([event("SETTLE", "comparison", charged=71000, settled=True)])
    assert operator().accounting(store) == {"charged_ms": 71000, "reserved_ms": 0}


def test_nonprefixed_render_reserve_still_holds_aggregate_capacity():
    store = Metadata([event("RESERVE", "render", study="render-case-hash")])
    assert operator().accounting(store) == {"charged_ms": 0, "reserved_ms": 300000}


def test_original_study_prefix_and_original_arm_union_never_double_counts():
    store = Metadata([event("RESERVE", "same", study="pirc27-first-pilot"),
        event("SETTLE", "same", study="pirc27-first-pilot", charged=13, settled=True),
        event("SETTLE", "future", study="pirc27-probability-pilot", arm="pirc27-probability-original", charged=17, settled=True),
        event("SETTLE", "foreign", study="pirc28-fixture", arm="pirc28-original", charged=999, settled=True),
        event("HEARTBEAT", "ignored")])
    assert operator().accounting(store) == {"charged_ms": 30, "reserved_ms": 0}


def test_unknown_cost_conservative_full_charge_and_pending_are_not_free():
    store = Metadata([event("SETTLE", "unknown", charged=300000, settled=True),
                      event("RESERVE", "pending", reserved=120000)])
    assert operator().accounting(store) == {"charged_ms": 300000, "reserved_ms": 120000}
