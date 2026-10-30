import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import q4_packed_format as packed
import verify_q4_paired_recovery as recovery


class PairedRecoveryTests(unittest.TestCase):
    def test_rebind_changes_only_core_header_and_preserves_ordered_records(self):
        manifest, core, mask = 'a'*64, 'b'*64, 1
        records = [struct.pack('<II',n,k)+bytes([i+1])*4
                   for i,(_,_,n,k) in enumerate(packed.projections(mask))]
        original = packed.encode_header(manifest,'0'*64,mask)+b''.join(records)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'original.bin',Path(directory)/'new.bin'
            source.write_bytes(original)
            # Existing projection validation has separate real-size tests; use
            # tiny records here to test the stream and source-binding boundary.
            with patch.object(recovery,'PAYLOAD_BYTES',4), patch.object(packed,'PAYLOAD_BYTES',4), \
                    patch.object(packed,'validate_projection') as validate:
                result = recovery.rebind_carrier(source,target,core,manifest,mask)
                self.assertEqual(validate.call_count,3)
                for call,raw,(_,_,n,k) in zip(validate.call_args_list,records,packed.projections(mask),strict=True):
                    self.assertEqual(call.args,(raw,n,k))
                self.assertEqual(target.read_bytes(),packed.encode_header(manifest,core,mask)+b''.join(records))
                self.assertEqual(source.read_bytes(),original)
                self.assertEqual(result['quantized_projections'],3)
                with self.assertRaises(FileExistsError):
                    recovery.rebind_carrier(source,target,core,manifest,mask)
                with self.assertRaises(ValueError):
                    recovery.rebind_carrier(source,source,core,manifest,mask)

    def test_wrong_original_core_and_trailing_or_truncated_records_are_rejected(self):
        manifest, mask = 'a'*64,1
        records=b''.join(struct.pack('<II',n,k)+b'abcd' for _,_,n,k in packed.projections(mask))
        good=packed.encode_header(manifest,'0'*64,mask)+records
        wrong=packed.encode_header(manifest,'b'*64,mask)+records
        with tempfile.TemporaryDirectory() as directory, patch.object(recovery,'PAYLOAD_BYTES',4):
            path=Path(directory)/'source.bin'
            for raw in (wrong,good[:-1],good+b'x'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    list(recovery.source_payloads(path,manifest,mask))

    def test_candidate_contract_requires_completed_original_base_and_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); core=root/'core.safetensors';core.write_bytes(b'placeholder')
            proxy=[]
            for step in (0,1):
                row={'step':step}
                for kind in ('down','logits'):
                    path=root/f'{step}-{kind}.bf16';path.write_bytes(b'\0\0');row[kind+'_path']=str(path)
                proxy.append(row)
            candidate=dict(schema='q4-paired-tail-recovery-v1',completed=True,trainingcompleted=True,originalbase=True,
                core_path=str(core),core_file_sha256='b'*64,source_receipt={'source':'fixed'},final_proxy=proxy)
            self.assertEqual(recovery.candidate_contract(candidate,{'source':'fixed'},root),core)
            for key,value in [('originalbase',False),('trainingcompleted',False),('core_file_sha256','0'*64),
                              ('source_receipt',{}),('final_proxy',list(reversed(proxy)))]:
                bad=copy.deepcopy(candidate);bad[key]=value
                with self.assertRaises(ValueError):
                    recovery.candidate_contract(bad,{'source':'fixed'},root)
            alias=root/'alias';alias.symlink_to(core)
            with self.assertRaises(ValueError):
                recovery.local_artifact(str(alias),root)

    def test_loaded_candidate_requires_actual_core_and_exact_plain_packed_scope(self):
        base={'checkpoint_sha256':'c'*64,'original_down_sha256':'d'*64}
        loaded=dict(base_identity=recovery.IDENTITY,manifest_sha256=recovery.MANIFEST,
            inference_precision='mixed-q4-g16-packed',core_loaded=True,core_file_sha256='b'*64,
            core_tensor_sha256='e'*64,retained_original_weight_bytes=0,packed_version=2,residual_rank=0,
            active_residual_projections=0,quantized_projections=32,research_sequential=False,
            research_capture_inputs=False,research_intervention=False,**base)
        recovery.candidate_loaded(loaded,'b'*64,base,'e'*64)
        for key,value in [('core_loaded',False),('core_file_sha256','f'*64),('core_tensor_sha256','f'*64),
                          ('packed_version',3),('quantized_projections',33),('residual_rank',8),
                          ('research_intervention',True),('original_down_sha256','f'*64)]:
            bad={**loaded,key:value}
            with self.assertRaises(ValueError):
                recovery.candidate_loaded(bad,'b'*64,base,'e'*64)

    def test_upstream_exact_bytes_but_down_may_change(self):
        original=np.zeros(10240,np.float32);actual=original.copy();actual[8192:]=1
        recovery.unchanged_upstream(actual,original)
        actual[0]=-0.0
        with self.assertRaises(ValueError):
            recovery.unchanged_upstream(actual,original)
        with self.assertRaises(ValueError):
            recovery.unchanged_upstream(actual[:-1],original)
        actual=original.copy();actual[-1]=np.inf
        with self.assertRaises(ValueError):
            recovery.unchanged_upstream(actual,original)


if __name__ == '__main__':
    unittest.main()
