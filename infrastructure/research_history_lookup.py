"""Owned pure identity/status facts from freshly hash-verified history reports.

No grants, clocks, permission decisions, manifest bytes or data bytes. Callers
must verify actual metadata again before every use; discard at physical phase
boundaries and when the owning authoritative history prefix changes.
"""
from bisect import bisect_left


_IDENTITY = ('dataset_id', 'release_id', 'source_block_id', 'sha256')


class SourceHistoryLookup:
    def __init__(self):
        self._reports = {}
        self._veto_scopes = set()
        self._veto_keys = None

    def contains(self, report_key):
        return report_key in self._reports

    def add(self, report_key, records):
        """Only after the caller's actual content hash and schema checks."""
        if self.contains(report_key):
            return
        try:
            facts = sorted((tuple(record[field] for field in _IDENTITY), record['status']) for record in records)
            entry = ([key for key, _ in facts], [status for _, status in facts])
            scopes = set()
            for key, status in facts:
                if status in {'exposed', 'unknown'}:
                    dataset, _, source, sha = key
                    # Same hash survives renamed datasets/windows. Same dataset /
                    # source-block survives release/content changes. Reports have
                    # valid hashes by schema; unknown-hash legacy READs remain in
                    # the separate conservative event lookup, not this class.
                    scopes.update({('hash', sha), ('source', dataset, source)})
            self._veto_keys = None
            self._veto_scopes.update(scopes)
            # Publish completion last: contains() must never hide missing vetoes.
            self._reports[report_key] = entry
        except BaseException:
            # A caller can catch a failed update within the same owned phase.
            # Drop all derived facts, including partially mutated containers;
            # its next guard must rebuild from freshly verified actual reports.
            self.__init__()
            raise

    def coverage(self, report_key, identity):
        keys, statuses = self._reports[report_key]
        key = tuple(identity[field] for field in _IDENTITY)
        position = bisect_left(keys, key)
        return statuses[position] if position < len(keys) and keys[position] == key else None

    def prior_exposure(self, identity):
        if self._veto_keys is None:
            # Build once after the actual verified reports have been added;
            # never insert every window into a sorted list quadratically.
            self._veto_keys = sorted(self._veto_scopes)
        for key in [('hash', identity['sha256']),
                    ('source', identity['dataset_id'], identity['source_block_id'])]:
            position = bisect_left(self._veto_keys, key)
            if position < len(self._veto_keys) and self._veto_keys[position] == key:
                return True  # Historical fact only, never current permission.
        return False
