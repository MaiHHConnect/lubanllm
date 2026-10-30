import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {Database} from 'bun:sqlite';
import {createHash} from 'node:crypto';
import {patchNativeEvolutionTrainingHistory} from '../scripts/native-evolution-training-history-patches.ts';
import {core450ControlMetadata} from '../.native-migration/research/native-core450-owner-controller/metadata.ts';
import {coreCheckpointField,coreCheckpointMetadata,coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {nativePartialCorrectIDs} from '../.native-migration/research/native-core-controller-candidate/progress.ts';
import {CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary} from '../.native-migration/research/native-core-controller-candidate/segments.ts';
const source=await readFile(new URL('../.native-migration/research/native-core450-runner/training-safety.ts',import.meta.url),'utf8');
const replace=(s:string,a:string,b:string)=>{assert.equal(s.split(a).length,2);return s.replace(a,b);};
const scope={assert,coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory,nativePartialCorrectIDs,CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary};
const safety=new Function(...Object.keys(scope),new Bun.Transpiler({loader:'ts'}).transformSync(patchNativeEvolutionTrainingHistory(source,replace).replace(/^import .*?;$/gm,'').replaceAll('export ','')+'\nreturn {nextCore450TrainingStep,deriveCore450TrainingCheck};'))(...Object.values(scope));
const code=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
const begin=code.indexOf('async function prepareCore450SegmentHeld('),end=code.indexOf('\n}',begin);assert(begin>=0&&end>begin);
const consumer=code.slice(begin,end+2).replace(/\b(assert|hash|readFile|join)\d+\b/g,'$1');
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
async function segment(payload:any,groups:any){
 let next:number|undefined,admissions=0;const stop=new Error('next boundary reached actual trainer admission');
 const cap={approval:{step:coreCheckpointMetadata(coreCheckpointField(payload,'step'))},checkpoint:{payload},binding:{records_sha256:'records'}};
 const approval={protocol:{base:{}},graph:{records_sha256:'records'},actual:{fixture:{directory:'/unused',manifest_sha256:hash('fixture')}},manifest:'fixture'};
 const env={assert,hash,core450ControlMetadata,assertNativeExpertScopeGuard(){},validateCore450Checkpoint(_e:any,value:any){assert.equal(value,cap);},assertNativeOrdinaryPreparation(){},nextCore450TrainingStep(p:any,g:any){next=safety.nextCore450TrainingStep(p,g);return next;},async admitReceipt(){admissions++;throw stop;}};
 const run=new Function(...Object.keys(env),consumer+'\nreturn prepareCore450SegmentHeld;')(...Object.values(env));
 try{await run({}, {},approval,cap,'/unused',{},groups);}catch(error){if(error!==stop)throw error;}
 return {next,admissions};
}
const proof=(id:string,passed:boolean)=>({id,answer:passed?'正确':'错误',passed,complete:true,branch_enabled:false,reference_used:false});
const record=(id:string)=>({id,answer:'正确',provenance:{generation_check:{type:'exact',expected:'正确',max_new_tokens:16}}});
const groups={new_training:[record('new')],replay_training:[record('known')]};
function typed(v:any):any{if(v===null)return{kind:'none'};if(Array.isArray(v))return{kind:'list',items:v.map(typed)};if(typeof v==='object')return{kind:'dict',entries:Object.entries(v).map(([k,x])=>[{kind:'string',value:k},typed(x)])};return{kind:typeof v==='number'?(Number.isInteger(v)?'int':'float'):typeof v==='boolean'?'bool':'string',value:typeof v==='number'?String(v):v};}
function payload(step:number){
 const results:any[]=[];let previous=null;for(const n of CORE450_CHECKS.filter(n=>n<=step)){
  const generated=[proof('new',false)],r=safety.deriveCore450TrainingCheck({step:n,new_records:groups.new_training,replay_records:groups.replay_training,baseline:[proof('new',false)],previous,new_generation:generated,replay_generation:[proof('known',true)],baseline_replay_loss:1,replay_loss:1,previous_streak:0,prior_safety_failed:false,constraint_residual:{quantized_frobenius:0,quantized_max_abs:0,quantized_relative_to_delta:0,master_frobenius:0,weight_dtype:'torch.bfloat16'}});results.push(r.row);previous=generated as any;
 }
 const p=typed({step,consecutive_passes:0,new_exposures:coreNewExposures(step,groups.new_training),report:{steps:450,training_safety_policy:'closed-book-growth-450-train-safety-v1',partial_safety_failed:false,partial_training_baseline:[proof('new',false),proof('known',true)],baseline_training_replay_generation:[proof('known',true)],baseline_training_replay_loss:1,baseline_correct_replay_training_ids:['known'],training_selection:{checkpoints:[...CORE450_CHECKS],policy_version:'closed-book-growth-450-train-safety-v1',required_consecutive_passes:2,required_new_exposures:2,minimum_replay_generation:0,replay_generation_ids:['known'],results,selected_step:null}}});
 p.entries.push([{kind:'string',value:'optimizer'},{kind:'dict',entries:[[{kind:'string',value:'state'},{kind:'dict',entries:[[{kind:'int',value:'0'},{kind:'dict',entries:[[{kind:'string',value:'exp_avg'},{kind:'tensor',name:'exp_avg'}]]}]]}]]}]);return p;
}
test('actual segment consumer projects control fields, preserves integer Adam keys and durable boundaries',async()=>{
 for(const [step,next] of [[0,16],[48,50],[50,64],[448,450]]){const p=payload(step),before=JSON.stringify(p);expect(()=>coreCheckpointMetadata(p)).toThrow('恢复metadata需要string字典');expect(await segment(p,groups)).toEqual({next,admissions:1});expect(JSON.stringify(p)).toBe(before);}
 await expect(segment(payload(450),groups)).rejects.toThrow('Training is terminal');
 const bad=payload(50),report=coreCheckpointField(bad,'report')!;const selection=coreCheckpointField(report,'training_selection')!;(coreCheckpointField(selection,'results') as any).items=[];await expect(segment(bad,groups)).rejects.toThrow();
});
test.skipIf(process.env.LUBAN_NATIVE_CP0_ACTUAL!=='1')('actual v10 CP0 metadata reaches real segment trainer boundary without opening PT/raw weights',async()=>{
 const path='/Users/linhao/Library/Application Support/Luban/native-evolution-work/identity-EAnyqC/current-dc74e691830749ffa66b7fda18b6931f/checkpoint-tree.json';
 const tree=JSON.parse(await readFile(path,'utf8')),p=coreCheckpointField(tree,'payload')!;
 const db=new Database('/Volumes/移动数据库/lubanllm/app/memory.sqlite3',{readonly:true});let row:any;try{row=db.query('SELECT body FROM training_runs WHERE id=?').get('51ede0d58379');}finally{db.close();}
 const body=JSON.parse(row.body);expect(body.checkpoint).not.toBeNull();expect(()=>coreCheckpointMetadata(p)).toThrow('恢复metadata需要string字典');expect(await segment(p,body.records.groups)).toEqual({next:16,admissions:1});
});
