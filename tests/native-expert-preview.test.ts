import {test,expect,afterEach} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,mkdir,rm,realpath} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {patchNativeExpertCohortPreview} from '../scripts/native-expert-source-patches.ts';
const root=resolve(import.meta.dir,'..'),directories:string[]=[],memories:any[]=[];
afterEach(async()=>{for(const m of memories.splice(0))m.close();for(const path of directories.splice(0))await rm(path,{recursive:true,force:true});});
const replace=(body:string,before:string,after:string)=>{assert.equal(body.split(before).length,2);return body.replace(before,after);};
async function compiled(patched:boolean){
 const work=await realpath(await mkdtemp(join(tmpdir(),'luban-expert-preview-')));directories.push(work);
 const entry=join(work,'entry.ts');
 await writeFile(entry,`export {Memory,ledgerStringify} from ${JSON.stringify(join(root,'backend/memory.ts'))};
 export {nativeEvolutionPreviewMemory} from ${JSON.stringify(join(root,'backend/native-evolution-preview.ts'))};
 export {selectLearningCohort} from ${JSON.stringify(join(root,'.native-migration/research/native-learning-product/cohort.ts'))};`);
 const build=await Bun.build({entrypoints:[entry],target:'bun',plugins:patched?[{name:'actual-preview-admission',setup(b){
  b.onLoad({filter:/native-product-runtime\/backend\/native-evolution-cohort\.ts$/},async args=>({loader:'ts',contents:patchNativeExpertCohortPreview(await readFile(args.path,'utf8'),replace,root)}));
 }}]:[]});assert(build.success,build.logs.map(String).join('\n'));
 const output=join(work,'compiled.js');await writeFile(output,await build.outputs[0]!.arrayBuffer());return{api:await import(output),work};
}
async function fixture(patched:boolean){
 const {api,work}=await compiled(patched),storage=join(work,'data'),store=join(storage,'modules');await mkdir(store,{recursive:true});
 const memory=new api.Memory(storage);memories.push(memory);const identity='a'.repeat(64),other='b'.repeat(64);
 const base=Object.fromEntries(['identity','model_manifest_sha256','config_sha256','checkpoint_sha256','tokenizer_sha256','original_down_sha256','engine_binary_sha256'].map(key=>[key,identity]));
 Object.assign(base,{core_parameter:'model.layers.23.mlp.down_proj.weight',core_shape:[2,3],dtype:'BF16',device:'mps'});
 memory.connect((db:any)=>{const insert=db.query('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?)');
  for(let i=0;i<19;i++){const id='row-'+i,purpose=i<16?'training':'validation';
   insert.run(id,id,'ok','human',purpose,0,1,1,api.ledgerStringify({base_identity:identity,verified_by:'local requester',evolution_role:'new',split_family:id,generation_check:{type:'exact',expected:'ok'}}));}
  insert.run('other','other','ok','human','training',0,1,1,api.ledgerStringify({base_identity:other,verified_by:'local requester'}));
 });memory.model_identity=other;
 return{...api,memory,base,identity,other,storage,store};
}
test('actual frozen expert cohort reproduces old instanceof failure and accepts only registered identity preview',async()=>{
 const old=await fixture(false),oldView=await old.nativeEvolutionPreviewMemory(old.memory,old.storage,old.identity);
 expect(()=>old.selectLearningCohort(oldView,old.store,old.base)).toThrow('进化资料库模型身份不一致');
 const f=await fixture(true),view=await f.nativeEvolutionPreviewMemory(f.memory,f.storage,f.identity);
 const cohort=f.selectLearningCohort(view,f.store,f.base);
 expect(cohort.records.train).toHaveLength(16);expect(cohort.records.valid).toHaveLength(3);
 expect(view.samples('training')).toHaveLength(16);expect(view.samples('validation')).toHaveLength(3);
 expect(cohort.records.train.some((row:any)=>row.id==='other')).toBe(false);cohort.assertUnchanged();
 expect(f.memory.model_identity).toBe(f.other);
 // Structural copies cannot acquire registered preview authority, even with the correct identity.
 expect(()=>f.selectLearningCohort({...view},f.store,f.base)).toThrow('进化资料库模型身份不一致');
 const wrong=await f.nativeEvolutionPreviewMemory(f.memory,f.storage,f.other);
 expect(()=>f.selectLearningCohort(wrong,f.store,f.base)).toThrow('进化资料库模型身份不一致');
 expect(()=>f.selectLearningCohort(view,f.store,{...f.base,original_down_sha256:'bad'})).toThrow('快照基座绑定无效');
});
test('registered expert preview preserves insufficient-material wait without mutating shared identity',async()=>{
 const f=await fixture(true),view=await f.nativeEvolutionPreviewMemory(f.memory,f.storage,f.other);
 expect(()=>f.selectLearningCohort(view,f.store,{...f.base,identity:f.other})).toThrow('等待足够的已核验新知识及独立检验资料');
 expect(f.memory.model_identity).toBe(f.other);
});
