import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {checkpointApprovedBinding as typed,equalCheckpointValues as same} from '../.native-migration/research/native-unified-service/checkpoint-owner.ts';
import {coreCheckpointMetadata as plain} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {parseJSON} from '../.native-migration/research/native-product-runtime/backend/wire.ts';
import {encoded,equal} from '../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
import evidence from '../research/native_evolution_final_binding_20261017.json';
const worker=await readFile(resolve('build/native-evolution/worker.js'),'utf8');
const binding=()=>parseJSON(evidence.original_binding_json);
const reconstructed=()=>plain(typed(binding()));
function consumerPrefix(source:string){
 const start=source.indexOf('async function observeCore450ProductFinal('),end=source.indexOf('  const control =',start);
 assert(start>=0&&end>start);
 return source.slice(start,end)+'\nreturn true;\n}';
}
const normalize=(s:string)=>s.replace(/\bequal3\b/g,'equalCheckpointValues').replace(/\bapprovedBinding2\b/g,'checkpointApprovedBinding').replace(/\b(assert|need|ledgerParse|equal|hash|encoded|object|trainingProcessStarted)\d+\b/g,'$1');
const finalCode=normalize(consumerPrefix(worker));
assert(finalCode.includes('FINAL checkpoint typed binding differs'));
const runFinal=new Function('assert','assertNativeExpertScopeGuard','equalCheckpointValues','checkpointApprovedBinding',finalCode+';return observeCore450ProductFinal;')(assert,()=>{},same,typed);
function request(value:any=binding()){return{expected:{id:evidence.run_id,kind:'evolution',binding:value}};}
function cap(){return{approval:{step:450,id:evidence.run_id},binding:reconstructed(),checkpoint:{binding:typed(reconstructed())}};}
const variants:{name:string;change:(v:any)=>void}[]=[
 {name:'wrong value',change:v=>v.arguments.learning_rate=0.00002},
 {name:'wrong numeric type',change:v=>v.arguments.learning_rate='0.00001'},
 {name:'int versus float',change:v=>v.arguments.steps=new Number(450)},
 {name:'wrong source',change:v=>v.implementation_sha256[Object.keys(v.implementation_sha256)[0]]='f'.repeat(64)},
 {name:'missing source',change:v=>delete v.active_files_sha256['active-core.json']},
 {name:'wrong identity',change:v=>v.identity='f'.repeat(64)},
];
test('actual FINAL consumer rejects old spelling equality and accepts identical typed full v18 binding',async()=>{
 expect(equal(binding(),reconstructed())).toBe(false);
 expect(encoded(binding().arguments.learning_rate)).toBe('0.00001');
 expect(encoded(reconstructed().arguments.learning_rate)).toBe('1e-05');
 expect(same(typed(binding()),typed(reconstructed()))).toBe(true);
 expect(await runFinal({}, {}, {}, cap(), {}, request(), {}, {})).toBe(true);
 for(const {change}of variants){const b=binding();change(b);await expect(runFinal({}, {}, {}, cap(), {}, request(b), {}, {})).rejects.toThrow();}
 for(const field of ['id','kind']){const r=request();(r.expected as any)[field]='wrong';await expect(runFinal({}, {}, {}, cap(), {}, r, {}, {})).rejects.toThrow('step/kind/run');}
 const c=cap();c.approval.step=448;await expect(runFinal({}, {}, {}, c, {}, request(), {}, {})).rejects.toThrow('step/kind/run');
});

test('final compiled commit row accepts equivalent float spelling and rejects changed typed bindings',async()=>{
 const start=worker.indexOf('function row2(db, r) {'),end=worker.indexOf('\nfunction cas2(',start);assert(start>=0&&end>start);
 const revised=normalize(worker.slice(start,end));
 assert(revised.includes('Commit checkpoint typed binding differs'));
 const hash=(x:any)=>createHash('sha256').update(x).digest('hex');
 const env={need:assert,ledgerParse:parseJSON,equal,hash,encoded,object:(x:any)=>x!==null&&typeof x==='object',trainingProcessStarted:()=>null,checkpointApprovedBinding:typed,equalCheckpointValues:same};
 const make=(src:string)=>new Function(...Object.keys(env),src+';return row2;')(...Object.values(env));
 const body={id:evidence.run_id,kind:'evolution',scope:'product',binding:binding(),arguments:binding().arguments,records:evidence.records,started:1,owner:null};
 const row={id:evidence.run_id,identity:evidence.identity,kind:'evolution',state:'paused',body:encoded(body)};
 const db={query:(sql:string)=>({get:()=>sql.startsWith('SELECT *')?row:null,all:()=>[{id:row.id}]})};
 const r={expected:{id:row.id,binding:reconstructed()},base:{identity:row.identity}};
 expect(make(revised)(db,r).value).toBe(row);
 for(const {change}of variants){const v=reconstructed();change(v);expect(()=>make(revised)(db,{...r,expected:{...r.expected,binding:v}})).toThrow();}
 expect(revised.includes('equal(body.arguments, body.binding.arguments)')).toBe(true);
});
