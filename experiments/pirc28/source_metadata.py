"""One preflight's pinned immutable metadata, never cached execution authority.

Each first read owns a short, freshly verified store transaction. Computational
validation uses detached values outside the writer lock. Original directory and
member handles remain owned until the final authority transaction closes them;
replacement, same-inode edits and ancestor redirects therefore veto completion.
Attempts, grants, reservations and arm balances must always use the real store.
"""

import copy
from contextlib import ExitStack

from infrastructure.research_json import read_json
from infrastructure.research_store import ResearchError, _opened_metadata_directory, digest, identifier


class SourceMetadata:
    def __init__(self, store, check_budget, maximum_bytes):
        self.store = store
        self.path, self.store_id = store.path, store.store_id
        self.check_budget, self.maximum_bytes = check_budget, maximum_bytes
        self.stack = ExitStack()
        self.values, self.verifiers = {}, {}
        self.consumed = 0
        self.active = False

    def __enter__(self):
        try:
            self.stack.enter_context(_opened_metadata_directory(self.path))
            self.members = self.stack.enter_context(_opened_metadata_directory(self.path/'manifests'))
            self.active = True
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exception):
        self.active = False
        return self.stack.__exit__(*exception)

    def manifest(self, object_id):
        identifier(object_id)
        if not self.active or not object_id.startswith((
                'study-', 'protocol-', 'preregistration-', 'pirc28-study-definition-', 'admission-')):
            raise ResearchError('CONTRACT_MISMATCH', 'immutable source metadata scope required')
        if object_id in self.values:
            try:
                self.verifiers[object_id]()
            except OSError as error:
                raise ResearchError('CORRUPT_ARTIFACT', 'pinned source metadata disappeared') from error
            return copy.deepcopy(self.values[object_id])
        # This callback may take time or hand control to another owner. It must
        # never run while we own the shared writer lock.
        self.check_budget()
        if len(self.values) >= 512:
            raise ResearchError('RESOURCE_LIMIT', 'pre-fit immutable metadata handle quota exceeded')
        try:
            with self.store._read_transaction():
                stream, size, verify = self.stack.enter_context(self.members.opened_member(
                    self.path/'manifests'/(object_id+'.json')))
                if self.consumed+size > self.maximum_bytes:
                    raise ResearchError('RESOURCE_LIMIT', 'pre-fit pinned metadata exceeds quota')
                value = read_json(stream, size, verify_identity=verify)
                event = self.store._manifest_event(object_id, last=True)
                if event is None:
                    raise ResearchError('MISSING_INPUT', 'source metadata is not published')
                if digest(value) != event['payload']['sha256']:
                    raise ResearchError('CORRUPT_ARTIFACT', 'pinned source metadata hash differs')
                # Retain the ordinary manifest boundary as well, including its
                # application-facing validation/refusal semantics.
                observed = self.store.manifest(object_id)
                if observed != value:
                    raise ResearchError('CORRUPT_ARTIFACT', 'source metadata differs at its physical read')
            self.consumed += size
            self.values[object_id], self.verifiers[object_id] = value, verify
            return copy.deepcopy(value)
        except ResearchError:
            raise
        except (OSError, ValueError) as error:
            raise ResearchError('CORRUPT_ARTIFACT', 'invalid original source metadata') from error

    def finish(self):
        """All original physical guards, while final real authority is locked."""
        if self.store._read_snapshot() is None:
            raise ResearchError('CONTRACT_MISMATCH', 'source completion needs current real authority')
        try:
            self.stack.close()
        except OSError as error:
            raise ResearchError('CORRUPT_ARTIFACT', 'original source metadata changed at completion') from error
        finally:
            self.active = False
