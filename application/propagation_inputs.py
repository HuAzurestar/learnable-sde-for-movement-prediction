"""Validate frozen numerical inputs at the propagation composition boundary."""

from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage

from .research_registry import implementation_hash


def validate_oracle_input(package, *, expected_package_hash):
    """Bind caller intent, immutable parameters and the current oracle code.

    This is input validation only. Shared runtime admission, budget reservation
    and scientific qualification remain separate requirements for research jobs.
    """
    from inference.affine_oracle import exact_transition
    if not isinstance(package, FrozenDynamicsPackage):
        raise DataValidationError("a frozen oracle package is required")
    package.validate()
    if package.package_hash != expected_package_hash:
        raise DataValidationError("requested frozen package identity differs")
    if package.manifest()["code_hash"] != implementation_hash(exact_transition):
        raise DataValidationError("frozen package oracle code differs from the current implementation")
    return package
