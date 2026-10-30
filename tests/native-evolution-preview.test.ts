import {afterEach,expect,test} from 'bun:test';
import {mkdtempSync,realpathSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {Memory,ledgerStringify} from '../backend/memory';
import {nativeEvolutionPreviewMemory} from '../backend/native-evolution-preview';
import {withNativeEvolutionMaterial,selectIndependentNativeEvolutionCohort} from '../backend/native-evolution-cohort';
const directories:string[]=[],memories:Memory[]=[];
afterEach(()=>{for(const m of memories.splice(0))m.close();for(const p of directories.splice(0))rmSync(p,{recursive:true,force:true});});
const identity='a'.repeat(64),other='b'.repeat(64),tensor='c'.repeat(64);
function fixture(){
 const root=realpathSync(mkdtempSync(join(tmpdir(),'luban-native-preview-')));directories.push(root);const memory=new Memory(root);memories.push(memory);
 const store=join(root,'modules');mkdirSync(store);
 const row=(id:string,purpose='training',role='new')=>({id,prompt:id,answer:'ok',source:'human',purpose,approved:true,training_allowed:true,provenance:{base_identity:identity,evolution_role:role,verified_by:'local requester',split_family:id,generation_check:{type:'exact',expected:'ok'}}});
 const records=[...Array.from({length:350},(_,i)=>row('new-'+i,i<345?'training':'validation')),row('replay-training','training','replay'),row('replay-validation','validation','replay'),row('reserved')];
 memory.connect(db=>{const q=db.query('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?)');for(const r of records)q.run(r.id,r.prompt,r.answer,r.source,r.purpose,0,1,1,ledgerStringify(r.provenance));for(const [id,base_identity]of [['other',other],['unbound',null]])q.run(id!,id!,'ok','human','training',0,1,1,ledgerStringify({base_identity,evolution_role:'new',verified_by:'local requester'}));});
 mkdirSync(join(root,'app/growth'));writeFileSync(join(root,'app/growth/rounds.json'),JSON.stringify({reserved_prompts:['reserved']}));
 const cases=Array.from({length:8},(_,i)=>row('product-ordinary-'+i,'validation','replay'));
 writeFileSync(join(store,'original-core-protection.json'),JSON.stringify({schema_version:1,protocol:'original-core-protection-v1',identity,passed:true,original_tensor_sha256:tensor,cases,generation:cases.map(r=>({id:r.id,answer:r.answer,passed:true,complete:true,branch_enabled:false}))}));
 return{root,store,memory};
}
test('real shared backend Memory supports identity-scoped cohort preview without changing its identity or filters',async()=>{
 const {root,store,memory}=fixture();expect(memory.model_identity).toBeNull();
 const view=await nativeEvolutionPreviewMemory(memory,root,identity);
 expect(memory.model_identity).toBeNull();expect(Object.isFrozen(view)).toBe(true);expect(view.samples()).toHaveLength(352);
 expect(view.samples().some(r=>['other','unbound','reserved'].includes(r.id))).toBe(false);
 await withNativeEvolutionMaterial(view,null,async()=>{const cohort=selectIndependentNativeEvolutionCohort(view,store,{identity,original_down_sha256:tensor});expect(cohort.unspent_count).toBe(350);expect(cohort.records.groups.new_training).toHaveLength(57);expect(cohort.records.protection).toHaveLength(8);cohort.assertUnchanged();});
 expect(memory.model_identity).toBeNull();
 memory.model_identity=other;
 const [a,b]=await Promise.all([nativeEvolutionPreviewMemory(memory,root,identity),nativeEvolutionPreviewMemory(memory,root,other)]);
 expect(a.samples()).toHaveLength(352);expect(b.samples().map(r=>r.id)).toEqual(['other']);expect(memory.model_identity).toBe(other);expect(memory.samples().map(r=>r.id)).toEqual(['other']);
});
test('preview refuses a different ledger root or malformed identity and preserves shared state on read failure',async()=>{
 const {root,memory}=fixture();memory.model_identity=other;
 await expect(nativeEvolutionPreviewMemory(memory,join(root,'another'),identity)).rejects.toThrow(/作用域/);
 await expect(nativeEvolutionPreviewMemory(memory,root,'wrong')).rejects.toThrow(/作用域/);
 writeFileSync(join(root,'app/growth/rounds.json'),'bad json');
 await expect(nativeEvolutionPreviewMemory(memory,root,identity)).rejects.toThrow();expect(memory.model_identity).toBe(other);
});
