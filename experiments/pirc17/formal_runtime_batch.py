"""Reuse fixed HISTORICAL runtime evidence ONLY within closed restoration.

No live environment check is cached. Every native result/completion/artifact
is still checked by ClosedOutputs. Here the original fixed runtime/start/phase
and constant-size first-floor linkage are checked once per exact scope, then
checked again by the SAME semantic validator before the batch can hand off.
Every consumed runtime/floor/request file is also fully byte-pinned at both
boundaries. Outside the bracketed batch the original strict reader is used.
"""
from pathlib import Path

from .formal_pinned_metadata import OwnedRecord, PinnedMetadata, record_payload
from .protocol_core import canonical, digest, file_hash, under

CONTRACT_SCOPE = ('protocol_sha256', 'execution_sha256', 'matrix_sha256',
                  'runtime_manifest_sha256', 'ledger_directory', 'phase_caps_ns')


def read_closed_runtime(owner, session_directory, *, contract, session_sha256,
                        ledger_root_sha256, worker_pid, phase, first_work_id):
    from .formal_entrypoint import read_runtime_evidence
    args = dict(contract=contract, session_sha256=session_sha256,
                ledger_root_sha256=ledger_root_sha256, worker_pid=worker_pid,
                phase=phase, first_work_id=first_work_id)
    batch = getattr(owner, '_metadata_batch', None)
    if batch is None:
        return read_runtime_evidence(session_directory, **args)
    batch._open()
    # These are ALL contract fields used by the unchanged runtime validator.
    # Compare their complete current values on each access, not object IDs or
    # a caller's claimed contract hash. Exact descriptor/budget checks remain
    # in ClosedOutputs and the original committed ledger state.
    scope = dict(args, contract={k:contract[k] for k in CONTRACT_SCOPE})
    key = ('closed-runtime', id(owner), str(session_directory), canonical(scope))

    def build():
        fixed = OwnedRecord({'payload':scope, 'sha256':digest(scope)})
        expected_args = record_payload(fixed)
        proof = read_runtime_evidence(session_directory, **expected_args)
        owned = OwnedRecord({'payload':proof, 'sha256':digest(proof)})
        root = Path(expected_args['contract']['ledger_directory'])
        session = Path(session_directory)
        runtime_root = under(session, 'runtime')
        paths = dict(binding=runtime_root/'binding.json', startup=runtime_root/'start.json',
                     phase=runtime_root/('bootstrap.json' if phase is None else digest(phase)+'.json'))
        records = {}
        for name, path in paths.items():
            ref = dict(record_payload(owned)[name], path=str(path))
            reader = PinnedMetadata(ref, root=root)
            record, value = batch.read(reader, ref)
            records[name] = value
        # Initial linkage is already CONSTANT size, not full history replay.
        # Preserve its actual bytes as well as the original semantic check.
        floor = records['startup'].get('predecessor_floor_event_sha256')
        extra = []
        if floor is not None:
            extra.append((under(root, 'events/000000.json'), floor))
        if phase is None:
            extra.append((under(session, 'bootstrap-request.json'), records['phase']['request_sha256']))
        for path, checksum in extra:
            ref = dict(path=str(path), content_sha256=checksum, file_sha256=file_hash(path))
            reader = PinnedMetadata(ref, root=root)
            batch.read(reader, ref)

        def closing_check():
            actual = read_runtime_evidence(session_directory, **expected_args)
            if canonical(actual) != canonical(record_payload(owned)):
                raise ValueError('historical runtime changed before metadata batch handoff')

        return record_payload(owned), closing_check

    return batch.validation(key, build)
