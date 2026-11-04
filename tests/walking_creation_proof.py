"""Pure test-only creation/exit evidence, independent of native stop order.

The caller decodes READY strictly and retains the opened exact owned handle.
No stop callback waits for this observer. Windows wrapper deadline stops are
bounded by the unchanged deadline; parent stop requests are recorded separately.
"""
import time


class CreationProof:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.first_parent_stop = None
        self.created = None

    def parent_stopping(self):
        # No handle/file access or waiting before the real product stop.
        stamp = self.clock()
        if self.first_parent_stop is None:
            self.first_parent_stop = stamp

    def confirm(self, identity, row, nonce, deadline, acknowledge):
        # READY was decoded by creation_receipt; compare the retained identity
        # again here so a different/reused handle cannot inherit that receipt.
        assert self.created is None, 'duplicate creation proof'
        assert (type(row) is dict and set(row) == {'pid', 'birth', 'nonce'}
                and row['nonce'] == nonce and type(row['pid']) is int
                and 1 < row['pid'] < 2**32 and type(row['birth']) is int
                and 0 < row['birth'] < 2**64), 'creation identity unknown'
        assert (identity.pid, identity.birth) == (row['pid'], row['birth']), 'creation identity mismatch'
        assert self.clock() < deadline, 'late creation receipt'
        assert self.first_parent_stop is None, 'creation after parent stop'
        assert identity.state() == 258, 'creation not live'
        acknowledge()
        # Successful ACK alone is insufficient. Prove the same handle still
        # live, then record the completed boundary inside the ORIGINAL window.
        assert identity.state() == 258, 'creation not live after ACK'
        stamp = self.clock()
        assert stamp < deadline, 'late creation ACK'
        assert self.first_parent_stop is None, 'creation crossed parent stop'
        self.created = (identity, identity.pid, identity.birth, stamp)

    def assert_exited(self, identity, whole_job_stopped):
        assert self.created is not None, 'creation not proven before stop'
        original, pid, birth, stamp = self.created
        assert identity is original and (identity.pid, identity.birth) == (pid, birth), 'retained identity mismatch'
        assert self.first_parent_stop is None or stamp < self.first_parent_stop, 'creation crossed parent stop'
        assert whole_job_stopped, 'whole-job stop unconfirmed'
        assert identity.state() == 0, 'exact descendant exit unconfirmed'
