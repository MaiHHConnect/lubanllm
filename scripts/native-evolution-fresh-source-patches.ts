/** Use the current retained source inventory; physical journal pins are a projection, not the full inventory. */
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
type Replace=(body:string,before:string,after:string)=>string;
export function patchNativeEvolutionFreshSource(body:string,replace:Replace,helper:string){
 const start=body.indexOf('export async function prepareCore450ProductFresh(');if(start<0)throw Error('Actual product fresh function missing');
 let product=body.slice(start);
 product=replace(product,'const files=collectLearningModelFiles(r.store),accepted=validateModelFiles(files,r.base,readNativeFoundationAdmission(r.store,files)),pins=nativeAcceptedHashes(r.store);',
  'const heldSource=await readNativeEvolutionHeldSource(engine,guard,r.preparation),{accepted,pins}=heldSource;');
 product=replace(product,'assert(equal(r.binding.active_files_sha256,pins)&&equal(r.previous.active_files_sha256,pins));',
  "assert(equal(r.binding.active_files_sha256,pins),'Fresh physical journal inventory differs from retained source');assert(equal(r.previous.active_files_sha256,peer.hashes),'Fresh RNG belongs to a different retained full source');");
 product=replace(product,'assert(equal(nativeAcceptedHashes(r.store),pins));','await heldSource.assertUnchanged();');
 return 'import {readNativeEvolutionHeldSource} from '+JSON.stringify(helper)+';\n'+body.slice(0,start)+product;
}
export async function applyNativeEvolutionFreshSourcePatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(path:string)=>string;replace:Replace;root:string}){
 const path=target(join(root,'.native-migration/research/native-core450-runner/fresh.ts'));
 bodies.set(path,patchNativeEvolutionFreshSource(bodies.get(path)??await readFile(path,'utf8'),replace,join(root,'backend/native-evolution-held-source.ts')));
}
