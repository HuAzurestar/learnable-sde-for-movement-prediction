"""Artificial ordered controls only; no original candidate/product/native imports.
Run: python -B tests/test_walking_creation_proof_selftest.py
"""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('owned_creation_proof', Path(__file__).with_name('walking_creation_proof.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Clock:
    def __init__(self): self.now = 1
    def __call__(self): return self.now


class Handle:
    def __init__(self, events):
        self.pid, self.birth, self.value = 123, 987, 258
        self.events = events
    def state(self):
        self.events.append(('state', self.value))
        if isinstance(self.value, Exception): raise self.value
        return self.value
    def cleanup(self):
        self.events.append(('cleanup', self.value))
        self.value = 0


class CreationControls(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.clock = Clock()
        self.proof = module.CreationProof(self.clock)
        self.identity = Handle(self.events)
        self.row = {'pid':123, 'birth':987, 'nonce':'owned'}
    def ack(self): self.events.append(('ACK', 'owned'))
    def confirm(self, ack=None):
        self.proof.confirm(self.identity,self.row,'owned',10,ack or self.ack)
    def exit_proof(self, whole=True):
        try:
            self.proof.assert_exited(self.identity,whole)
            self.events.append(('exit-proof', True))
        finally:
            self.identity.cleanup()

    def test_old_snapshot_rejects_only_fuse_first(self):
        for route in ('parent-first','fuse-first'):
            with self.subTest(route=route):
                identity=Handle([]); owned=[identity]; active=True
                self.assertEqual(identity.state(),258) # verified creation
                if route=='fuse-first': identity.value=0
                before_stop=[bool(owned) and owned[0].state()==258 and active]
                identity.value=0
                old_pass=bool(owned and before_stop and before_stop[0] and identity.state()==0)
                self.assertEqual(old_pass,route=='parent-first')

    def test_parent_first_retains_creation_and_exit_before_cleanup(self):
        self.confirm(); self.clock.now=2
        before=len(self.events); self.proof.parent_stopping()
        self.assertEqual(len(self.events),before) # no handle I/O delays stop
        self.identity.value=0; self.exit_proof()
        self.assertEqual(self.events[-3:], [('state',0),('exit-proof',True),('cleanup',0)])

    def test_fuse_first_retains_same_exact_exit_proof(self):
        self.confirm(); created=self.proof.created
        self.clock.now=10; self.identity.value=0 # original wrapper deadline
        self.clock.now=11; self.proof.parent_stopping()
        self.exit_proof()
        self.assertIs(self.proof.created,created)
        self.assertEqual(self.events[-3:], [('state',0),('exit-proof',True),('cleanup',0)])

    def test_missing_creation_even_if_dead(self):
        self.identity.value=0
        with self.assertRaisesRegex(AssertionError,'creation not proven'): self.exit_proof()

    def test_missing_or_malformed_decoded_identity(self):
        for row in (None,{},[],{'pid':123}, {'pid':True,'birth':987,'nonce':'owned'},
                    {'pid':123,'birth':None,'nonce':'owned'}, {'pid':123,'birth':987,'nonce':'wrong'}):
            with self.subTest(row=row):
                self.row=row
                with self.assertRaisesRegex(AssertionError,'identity unknown'): self.confirm()
        self.assertIsNone(self.proof.created)
        self.assertEqual(self.events,[])

    def test_delayed_creation(self):
        self.clock.now=10
        with self.assertRaisesRegex(AssertionError,'late creation'): self.confirm()
        self.assertEqual(self.events,[])

    def test_late_ack(self):
        def ack(): self.ack(); self.clock.now=10
        with self.assertRaisesRegex(AssertionError,'late creation ACK'): self.confirm(ack)
        self.assertIsNone(self.proof.created)

    def test_unknown_state(self):
        self.identity.value=0xffffffff
        with self.assertRaisesRegex(AssertionError,'not live'): self.confirm()
        self.assertIsNone(self.proof.created)

    def test_denied_observation(self):
        self.identity.value=PermissionError('artificial access denied')
        with self.assertRaises(PermissionError): self.confirm()
        self.assertIsNone(self.proof.created)

    def test_birth_mismatch_reused_pid(self):
        self.identity.birth+=1
        with self.assertRaisesRegex(AssertionError,'identity mismatch'): self.confirm()
        self.assertEqual(self.events,[])

    def test_pid_mismatch(self):
        self.identity.pid+=1
        with self.assertRaisesRegex(AssertionError,'identity mismatch'): self.confirm()

    def test_unconfirmed_whole_job_stop_even_if_exact_handle_dead(self):
        self.confirm(); self.identity.value=0
        with self.assertRaisesRegex(AssertionError,'whole-job stop unconfirmed'): self.exit_proof(False)
        self.assertNotIn(('exit-proof',True),self.events)

    def test_live_descendant_cleanup_cannot_supply_exit_proof(self):
        self.confirm()
        with self.assertRaisesRegex(AssertionError,'exact descendant exit unconfirmed'): self.exit_proof()
        self.assertEqual(self.identity.value,0) # cleanup succeeded, assertion still failed
        self.assertNotIn(('exit-proof',True),self.events)
        self.assertEqual(self.events[-1],('cleanup',258))

    def test_unknown_exit_state_cannot_pass(self):
        self.confirm(); self.identity.value=0xffffffff
        with self.assertRaisesRegex(AssertionError,'exit unconfirmed'): self.exit_proof()

    def test_different_retained_handle_cannot_pass(self):
        self.confirm(); self.identity=Handle(self.events); self.identity.value=0
        with self.assertRaisesRegex(AssertionError,'retained identity mismatch'): self.exit_proof()

    def test_stop_before_creation(self):
        self.proof.parent_stopping()
        with self.assertRaisesRegex(AssertionError,'after parent stop'): self.confirm()

    def test_stop_during_ack(self):
        def ack(): self.ack(); self.proof.parent_stopping()
        with self.assertRaisesRegex(AssertionError,'crossed parent stop'): self.confirm(ack)
        self.assertIsNone(self.proof.created)

    def test_fuse_during_ack(self):
        def ack(): self.ack(); self.identity.value=0
        with self.assertRaisesRegex(AssertionError,'not live after ACK'): self.confirm(ack)
        self.assertIsNone(self.proof.created)

    def test_failed_ack(self):
        def ack(): raise OSError('artificial ACK write failure')
        with self.assertRaises(OSError): self.confirm(ack)
        self.assertIsNone(self.proof.created)

    def test_repeated_stop_keeps_first_boundary(self):
        self.confirm(); self.clock.now=2; self.proof.parent_stopping()
        self.clock.now=3; self.proof.parent_stopping()
        self.assertEqual(self.proof.first_parent_stop,2)


if __name__ == '__main__': unittest.main(verbosity=2)
