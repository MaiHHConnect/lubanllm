"""CPU metadata inventory only; admission boundary mocked, no actual F/native quality."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import test_fresh_guard_binding as old
import verify_goal_product_guard as guard
import verify_goal_foundation as foundation
import smoke_product450 as smoke

PHASE='evolution-goal-new-data-fixed450-v1'
class DataPhaseGuardTests(unittest.TestCase):
    def fixture(self):
        original=old.FreshGuardBindingTests();args,plan,formal,binding=original.fixture();self.addCleanup(original.doCleanups)
        run=args.candidate_acceptance.parent;wanted=[]
        def member(name,value=None):
            path=run/name;guard.write_new(path,value or {'CPUmetadata':name});wanted.append(path)
            return {'path':str(path),'sha256':guard.sha(path)}
        def row(sample_id):return {'id':sample_id,'prompt':sample_id,'answer':'A','provenance':{'messages':[{'role':'user','content':sample_id}],'generation_check':{'type':'exact','expected':'A'}}}
        plan['groups']['new_training'] += [row('phaseT'+str(i)) for i in range(32)]
        plan['groups']['new_validation'] += [row('phaseV'+str(i)) for i in range(4)]
        plan['retention_regressions'] += [row('phaseA'+str(i)) for i in range(4)]
        original_proto=member('original-v1-formal.json',formal)
        v2=member('parent-v2-wrapper.json',{'parent_source_snapshot':{'old1':member('parent-v1-source.json')}})
        v3=member('parent-v3-wrapper.json',{'files':{'parent_protocol':v2},'parent_source_snapshot':{'old2':member('parent-v2-source.json')}})
        v4=member('parent-v4-wrapper.json',{'files':{'parent_protocol':v3},'parent_source_snapshot':{'old3':member('parent-v3-source.json')}})
        lineage={'files':{'parent_report':member('parent-v3-report.json'),'parent_protocol':v3},'ancestor_lineage':{
            'files':{'parent_report':member('parent-v2-report.json'),'parent_protocol':v2},'ancestor_lineage':{'files':{'parent_report':member('parent-v1-report.json')}}}}
        formal.update(protocol=PHASE,files={'parent_protocol':v4,'parent_report':member('parent-v4-preVALID-report.json'),
            'parent_result':member('parent-v4-failedVALID-result.json')},parent_source_snapshot={'old4':member('parent-v4-source.json')},
            ancestor_lineage=lineage,original_formal_protocol=original_proto,original_formal_files=copy.deepcopy(plan['formal_files']),
            material_manifest=member('new-phase-manifest.json'))
        args.formal_protocol.write_text(json.dumps(formal));plan.update(protocol=PHASE,goal_sequence_protocol={'path':str(args.formal_protocol),'sha256':guard.sha(args.formal_protocol)})
        (run/'plan.json').write_text(json.dumps(plan))
        new=plan['groups']['new_training']+plan['groups']['new_validation']+plan['retention_regressions']
        binding=(formal,new,plan['protection'],binding[3])
        return args,plan,formal,binding,wanted
    def prepare(self,args,formal,binding):
        with patch.object(guard,'bound_materials',return_value=binding),patch.object(foundation,'fresh_protocol_binding',return_value=(formal,{},{})),patch('luban.retention.read_retention_file'):
            return guard.prepare(args)
    def test_phase80_all_parent_snapshots_and_separate_failure_archive_inventory(self):
        args,plan,formal,binding,wanted=self.fixture();prepared=self.prepare(args,formal,binding)
        self.assertTrue(prepared['data_phase']);self.assertEqual(len(prepared['new80_ids']),80)
        self.assertEqual(len(prepared['old89_ids']),89);self.assertTrue(set(wanted)<=set(prepared['source_paths']))
        self.assertFalse(args.output.exists());self.assertIs(smoke.prepare,guard.prepare)
        self.assertEqual(len(guard.PRODUCT_FILES),25);self.assertIn('product_foundation_proof.py',guard.PRODUCT_FILES)
        self.assertNotIn('product_foundation_proof.py',formal['new_implementation_sha256']);self.assertEqual(len(formal['new_implementation_sha256']),22)
    def test_missing_phase_records_or_extra_ancestor_rejected(self):
        for change in ('train','validation','acceptance','ancestor'):
            args,plan,formal,binding,_=self.fixture()
            if change=='train':plan['groups']['new_training'].pop()
            elif change=='validation':plan['groups']['new_validation'].pop()
            elif change=='acceptance':plan['retention_regressions'].pop()
            else:formal['ancestor_lineage']['ancestor_lineage']['ancestor_lineage']['ancestor_lineage']={'files':{}}
            (args.candidate_acceptance.parent/'plan.json').write_text(json.dumps(plan))
            self.assertRaises(ValueError,self.prepare,args,formal,binding)

if __name__=='__main__':unittest.main(verbosity=2)
