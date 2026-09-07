"""Atomic directory commits for run artifacts and their audit record."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Optional

from domain import (
    ArtifactCommitError,
    ArtifactReference,
    DataValidationError,
    RunRecord,
)

from .artifacts import JsonArtifactStore

ArtifactWriter = Callable[[Path], None]


class AtomicRunStore:
    """Commit one run directory only after every staged artifact validates."""

    record_name = "run_record.json"

    def __init__(
        self,
        root: Path,
        json_store: Optional[JsonArtifactStore] = None,
    ) -> None:
        self.root = Path(root)
        self.json_store = json_store if json_store is not None else JsonArtifactStore()

    def commit(
        self,
        record: RunRecord,
        write_artifacts: Optional[ArtifactWriter] = None,
    ) -> RunRecord:
        record.validate()
        if record.artifacts:
            raise ArtifactCommitError("artifact references are owned by AtomicRunStore")
        self._validate_run_id(record.run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / record.run_id
        if destination.exists() or destination.is_symlink():
            raise ArtifactCommitError(f"run already exists: {record.run_id}")

        staging = Path(
            tempfile.mkdtemp(prefix=f".{record.run_id}.", dir=self.root)
        )
        committed = False
        try:
            if write_artifacts is not None:
                write_artifacts(staging)
            references = self._inventory(record.run_id, staging)
            if record.status == "succeeded" and not references:
                raise ArtifactCommitError(
                    "successful run must commit at least one artifact"
                )
            finalized = replace(record, artifacts=references)
            finalized.validate()
            record_path = staging / self.record_name
            self.json_store.write(finalized.to_dict(), record_path)
            if RunRecord.from_dict(self.json_store.read(record_path)) != finalized:
                raise ArtifactCommitError("staged RunRecord failed round-trip validation")
            staging.replace(destination)
            committed = True
            return finalized
        except ArtifactCommitError:
            raise
        except Exception as exc:
            raise ArtifactCommitError(
                f"atomic commit failed for run {record.run_id}: {exc}"
            ) from exc
        finally:
            if not committed and staging.exists():
                self._remove_staging(staging)

    def read(self, run_id: str) -> RunRecord:
        self._validate_run_id(run_id)
        directory = self.root / run_id
        record = RunRecord.from_dict(
            self.json_store.read(directory / self.record_name)
        )
        if record.run_id != run_id:
            raise DataValidationError(
                f"RunRecord run_id mismatch: expected {run_id}, got {record.run_id}"
            )
        for reference in record.artifacts.values():
            artifact = self._resolve_artifact(run_id, directory, reference.path)
            if not artifact.is_file() or artifact.is_symlink():
                raise DataValidationError(f"missing run artifact: {reference.path}")
            digest, size_bytes = self._digest(artifact)
            if digest != reference.sha256 or size_bytes != reference.size_bytes:
                raise DataValidationError(
                    f"run artifact digest mismatch: {reference.path}"
                )
        return record

    def _inventory(
        self,
        run_id: str,
        staging: Path,
    ) -> dict[str, ArtifactReference]:
        references: dict[str, ArtifactReference] = {}
        for artifact in sorted(staging.rglob("*")):
            if artifact.is_symlink():
                raise ArtifactCommitError(
                    f"staged artifacts must not be symlinks: {artifact.name}"
                )
            if not artifact.is_file():
                continue
            relative = artifact.relative_to(staging).as_posix()
            if relative == self.record_name:
                raise ArtifactCommitError(
                    f"artifact name is reserved: {self.record_name}"
                )
            digest, size_bytes = self._digest(artifact)
            references[relative] = ArtifactReference(
                path=f"{run_id}/{relative}",
                sha256=digest,
                size_bytes=size_bytes,
            )
        return references

    def _resolve_artifact(
        self,
        run_id: str,
        directory: Path,
        recorded_path: str,
    ) -> Path:
        relative = Path(recorded_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DataValidationError(f"invalid run artifact path: {recorded_path}")
        if not relative.parts or relative.parts[0] != run_id:
            raise DataValidationError(f"artifact belongs to another run: {recorded_path}")
        artifact = self.root.joinpath(*relative.parts)
        if artifact.parent != directory and directory not in artifact.parents:
            raise DataValidationError(f"artifact escapes run directory: {recorded_path}")
        return artifact

    def _remove_staging(self, staging: Path) -> None:
        root = self.root.resolve()
        resolved = staging.resolve()
        if resolved.parent != root or not staging.name.startswith("."):
            raise ArtifactCommitError(f"refusing unsafe staging cleanup: {staging}")
        shutil.rmtree(staging)

    @staticmethod
    def _digest(path: Path) -> tuple[str, int]:
        hasher = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
                size_bytes += len(chunk)
        return hasher.hexdigest(), size_bytes

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        first_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        remaining_characters = f"{first_characters}._-"
        if (
            not run_id
            or run_id[0] not in first_characters
            or any(character not in remaining_characters for character in run_id)
        ):
            raise DataValidationError(
                "run_id may contain only letters, digits, dot, underscore, and hyphen"
            )


__all__ = ["ArtifactWriter", "AtomicRunStore"]
