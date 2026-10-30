import {expect,test} from 'bun:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';
import {encoded} from '../backend/canonical-json';
import {patchNativeEvolutionSampleFields} from '../scripts/native-evolution-sample-patches';
const source=patchNativeEvolutionSampleFields(readFileSync('.native-migration/research/native-unified-expert/native-engine.ts','utf8'),(s,a,b)=>{assert.equal(s.split(a).length,2);return s.replace(a,b);});
const start=source.indexOf('  private trainingSampleFields('),end=source.indexOf('\n  private trainingSampleResult(',start);
const js=new Bun.Transpiler({loader:'ts'}).transformSync('class ActualFields{'+source.slice(start,end)+'}');
const record=(v:any)=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const fields=new Function('need','record','encoded','MAX_FRAME',js+';return new ActualFields();')(assert,record,encoded,4<<20);
const result=spawnSync(resolve('.venv/bin/python'),['-c','import json; from scripts.current_q4_identity_material import records; print(json.dumps(records(),ensure_ascii=False))'],{env:{...process.env,PYTHONPATH:'src:.'},encoding:'utf8',timeout:30000});assert.equal(result.status,0,result.stderr);const material=JSON.parse(result.stdout);
test('all current authorized identity and boundary rows pass actual Engine encoding with full contexts unchanged',()=>{
 const training=[...material.groups.new_training,...material.groups.replay_training],holdout=[...material.groups.new_validation,...material.groups.replay_validation,...material.retention_regressions],all=[...training,...holdout];
 expect(training).toHaveLength(36);expect(holdout).toHaveLength(17);expect(new Set(training.map((r:any)=>r.id)).size).toBe(36);
 expect(holdout.some((r:any)=>training.some((t:any)=>t.id===r.id))).toBe(false);
 for(const row of all){const before=JSON.stringify(row),value=fields.trainingSampleFields(row);expect(value.operation).toBe('prepare_training_sample');expect(value.sample.answer).toBe(row.answer);expect(JSON.parse(value.sample.provenance)).toEqual(row.provenance);expect(JSON.stringify(row)).toBe(before);expect(value.sample.provenance).toBe(fields.trainingSampleFields({...row,provenance:encoded(row.provenance)}).sample.provenance);}
 for(const row of material.retention_regressions){expect(row.training_allowed).toBe(false);expect(JSON.parse(fields.trainingSampleFields(row).sample.provenance).messages).toHaveLength(3);}
});
test('malformed provenance, missing answer and oversized context still fail the original fields contract',()=>{
 for(const provenance of [null,[],42,true])expect(()=>fields.trainingSampleFields({answer:'ok',provenance})).toThrow(/格式/);
 expect(()=>fields.trainingSampleFields({provenance:{}})).toThrow(/格式/);
 expect(()=>fields.trainingSampleFields({answer:'ok',provenance:{messages:[{role:'user',content:'x'.repeat(4<<20)}]}})).toThrow(/限制/);
});
