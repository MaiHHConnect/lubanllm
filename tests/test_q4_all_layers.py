"""Independent audit must detect packing corruption separately from weight loss."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_q4_all_layers import bf16_ulp, compare_projection, unpack_words
from diagnose_native_q4 import bf16, quantize


class AllLayerAuditTests(unittest.TestCase):
    def fixture(self, group_size=32):
        original=bf16(np.random.default_rng(42).normal(0,0.04,(2,64)).astype(np.float32))
        q=quantize(original.reshape(-1,group_size))
        words=np.zeros((2,8),dtype=np.uint32)
        codes=q[4].astype(np.uint32).reshape(2,8,8)
        for shift in range(8):
            words|=codes[:,:,shift]<<(4*shift)
        scales=q[7].reshape(2,64//group_size);biases=q[9].reshape(2,64//group_size)
        inputs=bf16(np.random.default_rng(43).normal(0,1,(9,64)).astype(np.float32))
        observed=bf16((inputs.astype(np.float64)@q[0].reshape(2,64).astype(np.float64).T).astype(np.float32))
        return original,words,scales,biases,inputs,observed

    def test_literal_packing_and_detect_corrupt_native_code(self):
        np.testing.assert_array_equal(unpack_words(np.array([[0x76543210,0xfedcba98]],dtype=np.uint32)),[list(range(16))])
        args=self.fixture()
        result=compare_projection(*args)
        self.assertTrue(result['conversion']['passed'])
        self.assertTrue(result['kernel']['passed'])
        self.assertGreater(result['weight_quantization_loss']['relative_l2'],0)
        args[1][0,0]^=np.uint32(1)
        result=compare_projection(*args)
        self.assertFalse(result['conversion']['passed'])
        self.assertEqual(result['conversion']['code_mismatches'],1)

    def test_group16_and_unsupported_group(self):
        args=self.fixture(16)
        result=compare_projection(*args,group_size=16)
        self.assertTrue(result['conversion']['passed'])
        self.assertTrue(result['kernel']['passed'])
        with self.assertRaises(ValueError):
            compare_projection(*args,group_size=8)

    def test_correct_conversion_does_not_hide_bad_gpu_result(self):
        args=self.fixture()
        args[-1][0,0]+=10
        result=compare_projection(*args)
        self.assertTrue(result['conversion']['passed'])
        self.assertFalse(result['kernel']['passed'])
        self.assertEqual(result['kernel']['outside_tolerance'],1)

    def test_nonfinite_and_wrong_shapes_fail_closed(self):
        args=self.fixture()
        args[-1][0,0]=np.nan
        with self.assertRaises(ValueError):
            compare_projection(*args)
        args=self.fixture()
        with self.assertRaises(ValueError):
            compare_projection(*args[:-1],args[-1][:,:1])
        np.testing.assert_array_equal(bf16_ulp([0,1,-2]),[2.**-133,2.**-7,2.**-6])


if __name__=='__main__':
    unittest.main()
