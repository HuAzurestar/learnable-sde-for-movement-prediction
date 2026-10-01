"""Freeze all-attempt costs, never condition resource use on successful scores."""


def frozen_cost(history, events):
    attempt_ids = {attempt["attempt_id"] for attempt in history}
    latest = {}
    for event in events:
        if event["event_kind"] in {"RESERVE", "SETTLE"} and event["payload"]["attempt_id"] in attempt_ids:
            latest[event["payload"]["reservation_id"]] = event
    sources = list(latest.values())
    entries = [event["payload"] for event in sources]
    missing = sorted(attempt_ids - {entry["attempt_id"] for entry in entries})
    unknown = sorted(entry["attempt_id"] for entry in entries
                     if entry["settled"] and entry["monotonic_elapsed_ms"] is None)
    pending = sorted(entry["attempt_id"] for entry in entries if not entry["settled"])
    return {"unit": "slot-ms", "scope": "all-cell-attempts-including-failures",
            "charged_ms": None if missing else sum(e["charged_ms"] for e in entries if e["settled"]),
            "reserved_ms": sum(e["reserved_ms"] for e in entries if not e["settled"]),
            "measured_ms": None if missing or unknown or pending else sum(e["monotonic_elapsed_ms"] for e in entries),
            "missing_attempt_ids": missing, "unknown_attempt_ids": unknown, "pending_attempt_ids": pending,
            "sources": sources}
