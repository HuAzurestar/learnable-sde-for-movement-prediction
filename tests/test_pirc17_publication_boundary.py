"""Public identifier exceptions must not weaken artifact or secret checks."""
from pathlib import Path

from scripts.check_public_release import is_approved_public_reference


def test_public_contract_identifier_is_narrowly_admitted():
    assert is_approved_public_reference("internal work item", Path("experiments/pirc17/protocol.py"), "NEX326")
    assert is_approved_public_reference("internal work item", Path("tests/test_pirc17_protocol.py"), "nex326")
    assert not is_approved_public_reference("internal work item", Path("experiments/pirc17/protocol.py"), "NEX-" + "99999")
    assert not is_approved_public_reference("internal work item", Path("docs/unrelated.md"), "NEX326")


def test_public_contract_identifier_does_not_allow_secrets_or_paths():
    for label in ("workstation path", "private key", "common access token", "internal role"):
        assert not is_approved_public_reference(label, Path("experiments/pirc17/protocol.py"), "NEX326")
