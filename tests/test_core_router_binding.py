"""Cold reload must preserve only the router bound to the unchanged live core."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from luban.evolution import load_active_core


class CoreRouterBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Path(self.directory.name)
        self.version = 'core-012345abcdef'
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(2, 2, bias=False)
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([layer])
        self.weight = layer.mlp.down_proj.weight
        self.original = self.weight.detach().clone()
        self.runtime = SimpleNamespace(model=model, decoder=model.model, identity='base',
                                       store=self.store, active_core=None, router_meta={})
        weights = self.store / (self.version + '.safetensors')
        save_file({'model.layers.0.mlp.down_proj.weight': self.original + 1}, str(weights))
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.metadata = {'identity': 'base', 'version': self.version, 'sha256': digest,
                         'report': 'evolution-012345abcdef.json'}
        (self.store / 'active-core.json').write_text(json.dumps(self.metadata))
        (self.store / (self.version + '.json')).write_text(json.dumps(self.metadata))
        (self.store / self.metadata['report']).write_text(json.dumps({'gates': {}}))

    def route(self, previous, bound, identity='base'):
        self.runtime.active_core = previous
        self.runtime.router_meta = {'calibrated': True, 'identity': identity, 'core_version': bound}

    def test_rejected_core_preserves_only_unchanged_original_binding(self):
        for previous, bound, identity, expected in (
                (None, None, 'base', True),
                (self.version, None, 'base', False),
                (None, self.version, 'base', False),
                (None, None, 'other-base', False)):
            with self.subTest(previous=previous, bound=bound, identity=identity):
                self.route(previous, bound, identity)
                self.assertIsNone(load_active_core(self.runtime))
                self.assertIsNone(self.runtime.active_core)
                self.assertEqual(self.runtime.router_meta['calibrated'], expected)
                self.assertEqual(self.runtime.core_requires_review['version'], self.version)
                self.assertTrue(torch.equal(self.weight, self.original))

    def test_missing_pointer_fallback_checks_the_same_binding(self):
        (self.store / 'active-core.json').unlink()
        with patch('luban.foundation.install_foundation', return_value=None):
            for previous, bound, expected in ((None, None, True), (self.version, None, False),
                                              (None, self.version, False)):
                self.route(previous, bound)
                self.assertIsNone(load_active_core(self.runtime))
                self.assertEqual(self.runtime.router_meta['calibrated'], expected)

    def test_accepted_same_core_preserves_router_but_a_change_invalidates_it(self):
        self.metadata.pop('report')  # Existing legacy compatibility acceptance.
        (self.store / (self.version + '.json')).write_text(json.dumps(self.metadata))
        for previous, bound, expected in ((self.version, self.version, True),
                                          (None, self.version, False), (self.version, None, False)):
            self.route(previous, bound)
            self.assertEqual(load_active_core(self.runtime)['version'], self.version)
            self.assertEqual(self.runtime.router_meta['calibrated'], expected)

    def test_missing_binding_does_not_keep_a_calibrated_router(self):
        self.runtime.router_meta = {'calibrated': True}
        load_active_core(self.runtime)
        self.assertFalse(self.runtime.router_meta['calibrated'])


if __name__ == '__main__':
    unittest.main()
