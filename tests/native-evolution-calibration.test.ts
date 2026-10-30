import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {predictorInputs,q4ConstraintFrame,validateConstraintFixture,FINAL_CONVERSION} from '../backend/native-evolution-constraint-inputs.ts';
import {patchConstraintFixtureParser} from '../scripts/native-evolution-calibration-patches.ts';
const replace=(s:string,a:string,b:string)=>{assert(s.includes(a),a);return s.replace(a,b);};
test('OLD causal predictor rows include final EOS and never prefix/last token states',()=>{
 const data=Buffer.alloc(5*2*2);for(let i=0;i<10;i++)data.writeUInt16LE(0x3f80+i,i*2);
 const result=predictorInputs(data,[-100,-100,2,3,4],2);expect(result.length).toBe(3*2*4);
 expect(result.readUInt32LE()).toBe((0x3f80+2)*65536);expect(result.readUInt32LE(result.length-4)).toBe((0x3f80+7)*65536);
 data.writeUInt16LE(0x7f80,4);expect(()=>predictorInputs(data,[-100,-100,2,3,4],2)).toThrow('Nonfinite');
});
test('Q4 capture checks actual full-depth predicted token and only selects last down activation',()=>{
 const tail=Buffer.alloc(20480),logits=Buffer.alloc(496640);tail.writeUInt16LE(0x4000,2048*2);logits.writeUInt16LE(0x3f80,42*2);
 const row={id:'train',input_tokens:[1,2,3],tokens:[42,248044]},event={request_id:'train',step:0,committed_tokens:3,tail_only:true,layer:23,tail_values:10240,vocab:248320,top1:42};
 const out=q4ConstraintFrame(event,row,0,tail,logits);expect(out.length).toBe(6144*4);expect(out.readFloatLE()).toBe(2);
 for(const change of [{layer:22},{request_id:'valid'},{top1:41},{committed_tokens:4}])expect(()=>q4ConstraintFrame({...event,...change},row,0,tail,logits)).toThrow();
 expect(()=>q4ConstraintFrame(event,{...row,tokens:[41]},0,tail,logits)).toThrow('differs');
});
test('both actual V3 parsers accept one static matrix and reject missing/surplus/unbound/type-invalid assets',async()=>{
 const source=await readFile('.native-migration/research/native-evolution-fixture/parser-readback.ts','utf8');
 const body=patchConstraintFixtureParser(source,replace,'helper').replace(/^import .*;\n/gm,'').replace('export function parseV3Fixture','function parseV3Fixture');
 const js=new Bun.Transpiler({loader:'ts'}).transformSync(body+'\nreturn parseV3Fixture;');
 const parse=new Function('validateConstraintFixture','need','sha',js)(validateConstraintFixture,(v:any,m:string)=>assert(v,m),(s:string)=>/^[a-f0-9]{64}$/.test(s));
 const hash='a'.repeat(64),tensors:any={};
 const t=(name:string,dtype:string,shape:number[])=>{tensors[name]={dtype,shape,bytes:shape.reduce((a,b)=>a*b,1)*({f32:4,bf16:2,u8:1,i64:8}[dtype]!),sha256:hash};};
 for(const name of ['candidate','master','reference','exp_avg','exp_avg_sq'])t(name,name==='candidate'?'bf16':'f32',[2,3]);
 t('basis','f32',[3,1]);t('adam_step','f32',[]);t('norm','bf16',[2]);t('head','bf16',[4,2]);t('cpu_rng','u8',[8]);t('constraint_inputs','f32',[5,3]);
 for(const name of ['r0','r1']){t(name+'_x','bf16',[1,2,3]);t(name+'_residual','bf16',[1,2,2]);t(name+'_ids','i64',[1,2]);t(name+'_labels','i64',[1,2]);}
 const config={identity:hash,device:'cpu',steps:'1',norm_eps:'1e-6',train_eos:'1',checkpoint_steps:'1',rng_mode:'fresh',final_conversion:FINAL_CONVERSION,constraint_old_positions:'2'};
 const manifest=(c:any=config,ts:any=tensors)=>['LUBAN_NATIVE_TRAINING_V3',...Object.entries(c).map(([k,v])=>`config\t${k}\t${v}`),...Object.entries(ts).map(([k,v]:any)=>`tensor\t${k}\t${v.dtype}\t${v.shape.join(',')||'-'}\t${k}.raw\t${v.bytes}\t${v.sha256}`),`row\tr0\tnew\t${hash}\t2\t1`,`row\tr1\treplay\t${hash}\t2\t1`,'step\t1\tnew\tr0\tr1'].join('\n')+'\n';
 const run=(c?:any,t?:any)=>parse(manifest(c,t),{identity:hash,device:'cpu',core_shape:[2,3],reference_sha256:hash,basis_sha256:hash},{cpu_sha256:hash,mps_sha256:null});
 expect(run().tensors.constraint_inputs.shape).toEqual([5,3]);expect(run().rows).not.toHaveProperty('constraint_inputs');
 for(const old of ['0','5','1.5'])expect(()=>run({...config,constraint_old_positions:old})).toThrow();
 expect(()=>run({...config,final_conversion:'other'})).toThrow();
 expect(()=>run(config,{...tensors,constraint_inputs:{...tensors.constraint_inputs,dtype:'bf16',bytes:30}})).toThrow();
 const missing={...tensors};delete missing.constraint_inputs;expect(()=>run(config,missing)).toThrow();
 const absent={...config} as any;delete absent.final_conversion;delete absent.constraint_old_positions;expect(()=>run(absent,tensors)).toThrow();
 const output=await readFile('.native-migration/backend/native-training-output.ts','utf8');expect(()=>patchConstraintFixtureParser(output,replace,'helper')).not.toThrow();
});
test('cached OLD bootstrap baseline retains exact IDs, complete EOS and original semantic check',async()=>{
 const {preparedOldBaseline}=await import('../backend/native-evolution-constraint-inputs.ts');
 const records=[{id:'old',answer:'',provenance:{generation_check:{type:'exact',expected:'K',max_new_tokens:8}}}],row={id:'old',answer:'K',passed:true,complete:true,branch_enabled:false,reference_used:false,finish_reason:'stop',raw_completion_token_ids:[42,248044]};
 const meta={method:FINAL_CONVERSION,training:false,new_validation_used:false,old_baseline:[row]};
 expect(preparedOldBaseline(meta,records)).toEqual([row]);
 for(const change of [{id:'other'},{answer:'wrong'},{passed:false},{complete:false},{branch_enabled:true},{reference_used:true},{raw_completion_token_ids:[42]},{finish_reason:'length'}])expect(()=>preparedOldBaseline({...meta,old_baseline:[{...row,...change}]},records)).toThrow();
 expect(()=>preparedOldBaseline({...meta,new_validation_used:true},records)).toThrow();
});
