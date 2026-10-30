import {test,expect} from 'bun:test';
import {validateNativeEvolutionConversionQualification as validate,type NativeEvolutionConversionQualification} from '../backend/native-evolution-graph-qualification.ts';
const binding={training_binary_sha256:'1'.repeat(64),training_source_proof_sha256:'2'.repeat(64),graph_binary_sha256:'5'.repeat(64)};
const valid=():NativeEvolutionConversionQualification=>({method:'gptq-bf16-refine-v1',evidence_id:'cpu-and-actual-reference',...binding,shape:[2048,6144],passes:2,optimizer_steps_in_conversion:0,calibration:'known-old-and-actual-q4-replay-training',new_validation_used:false,
 independent_reference:{result_id:'real-shape/reference',execution:'real-shape-independent-python-reference',candidate_tensor_sha256:'6'.repeat(64),pairs:['converted-candidate','converted-master'].map(name=>({name,elements:2048*6144,max_abs_error:0,allowed_max_abs_error:0}))},
 boundary:{result_id:'actual-compiled-cpu-448-450',step449_unmodified:true,step450_applied_once:true,seven_state_resume_equal:true,reference_training_basis_preserved:true,cancellation_before_commit:true,parent_rng_restored:true},
 actual_restore:{result_id:'actual-real-model/restore',actual_execution:'native-real-model',graph_binary_sha256:binding.graph_binary_sha256,candidate_tensor_sha256:'6'.repeat(64),restored_tensor_sha256:'7'.repeat(64),pid:123,engine_closed:true,lease_released:true,product_pointers_unchanged:true}});
test('numerical conversion report validation does not itself issue adoption or graph capability',()=>{expect(validate(valid(),binding)).toBeUndefined();});
test('conversion requires real shape, exact full matrices, fixed passes and no NEW VALID',()=>{
 const mutations=[(v:any)=>v.shape=[3,160],(v:any)=>v.passes=3,(v:any)=>v.new_validation_used=true,(v:any)=>v.independent_reference.pairs.pop(),(v:any)=>v.independent_reference.pairs[0].elements=1,(v:any)=>v.independent_reference.pairs[1].allowed_max_abs_error=1e-3,(v:any)=>v.independent_reference.pairs[1].max_abs_error=1e-8];
 for(const change of mutations){const v=valid();change(v);expect(()=>validate(v,binding)).toThrow();}
});
test('changed source, missing restore and cancellation/boundary failures deny conversion qualification',()=>{
 const mutations=[(v:any)=>v.training_binary_sha256='3'.repeat(64),(v:any)=>v.training_source_proof_sha256='4'.repeat(64),(v:any)=>v.boundary.step449_unmodified=false,(v:any)=>v.boundary.cancellation_before_commit=false,(v:any)=>v.boundary.seven_state_resume_equal=false,(v:any)=>v.boundary.reference_training_basis_preserved=false,(v:any)=>v.actual_restore.engine_closed=false,(v:any)=>v.actual_restore.graph_binary_sha256='8'.repeat(64)];
 for(const change of mutations){const v=valid();change(v);expect(()=>validate(v,binding)).toThrow();}
});
test('current pre-conversion product role cannot acquire the new qualification from CPU evidence alone',async()=>{
 const {prepareNativeConversionQualification}=await import('../scripts/prepare-native-conversion-qualification.ts');
 const {readFile}=await import('node:fs/promises');
 const {join}=await import('node:path');
 const runtime=JSON.parse(await readFile(join(import.meta.dir,'../build/native-evolution/runtime-authority.json'),'utf8'));
 if(runtime.roles.trainer.sha256!=='dea39ec08e4deeb2378a66c7460cafcfdb6a364b1332e66438fc53f14acb72ae')return;
 await expect(prepareNativeConversionQualification('/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/identity-native450-v19')).rejects.toThrow('Install the tested conversion trainer role');
});
