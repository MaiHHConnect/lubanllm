/** Replace observation snapshots with the native round's in-memory rollback and checkpoint RNG. */
import {join} from 'node:path';
import {readFile} from 'node:fs/promises';
type Context={bodies:Map<string,string>;target:(path:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export async function applyNativeEvolutionAmbientPatches({bodies,target,replace,root}:Context){
 const research=join(root,'.native-migration/research'),ambient=JSON.stringify(join(root,'backend/native-evolution-ambient.ts')),seed=JSON.stringify(join(root,'backend/native-evolution-seed.ts'));
 const edit=async(name:string,change:(body:string)=>string)=>{const path=target(join(research,name));bodies.set(path,change(bodies.get(path)??await readFile(path,'utf8')));};
 await edit('native-unified-expert/native-engine.ts',body=>{
  body=replace(body,'prepareRetainedCoreReload:(owners:NativeCore450Owner)=>','prepareRetainedCoreReload:(owners:NativeCore450Owner,prepared?:NativeShallowPreparation)=>');
  body=replace(body,'prepareRetainedCoreReload:async(owners:NativeCore450Owner)=>','prepareRetainedCoreReload:async(owners:NativeCore450Owner,prepared?:NativeShallowPreparation)=>');
  return replace(body,'prepareRetainedCoreReload(this,guard,owners)','prepareRetainedCoreReload(this,guard,owners,prepared)');
 });
 await edit('native-unified-expert/expert-process.ts',body=>{
  body=replace(body,"await this.command('close');this.child.stdin!.end();","const closed=await this.command('close');assert(closed.response.event==='closed'&&closed.response.resident_parameters===0&&closed.response.parent_restored===true,'Actual native process close ACK differs');this.child.stdin!.end();");
  body=replace(body,"for(const kind of(this.options.device==='mps'?['','_mps']:['']))assert((await readFile(join(this.options.directory,'parent'+kind+'_rng.raw'))).equals(await readFile(join(this.options.directory,'restored'+kind+'_rng.raw'))));",'');
  return replace(body,' async seedFresh(){',` async beginEvolutionRound(){return this.command('evolution-begin');}
 async restoreEvolutionRNG(){return this.command('evolution-end');}
 async exportEvolutionRNG(name:string){this.name(name);return this.command('evolution-rng '+name);}
 async seedFresh(){`);
 });
 await edit('native-unified-core-control/install.ts',body=>{
  body=replace(body,'router:Readonly<Record<string,any>>;closed:boolean','router?:Readonly<Record<string,any>>;closed:boolean');
  body=replace(body,"const before=await snapshotNativeRouter(peer.worker,'router-before-'+id);peer.features.add({close:()=>before.close()});await worker.loadCoreOwner(owner,cap.approval.step,cap);","const loaded=await worker.loadCoreOwner(owner,cap.approval.step,cap);assert(loaded.response.operation==='core-current'&&loaded.response.step===cap.approval.step&&loaded.response.core_tensor_sha256===cap.state.candidate.sha256,'Actual complete Core installation ACK differs');");
  body=replace(body,'const exported=await worker.exportCoreState(id);await verifyCoreExport(cap,exported.directory,worker.pid);','');
  const start=body.indexOf("const router=Object.freeze(Object.fromEntries(['projection.weight','projection.bias']"),end=body.indexOf('const state:Installed=',start);if(start<0||end<start)throw Error('Installation router audit boundary missing');body=body.slice(0,start)+body.slice(end);
  body=replace(body,'actual_artifacts:string;production_model_qualified:false}>','checkpoint_path:string;production_model_qualified:false}>');
  body=replace(body,'actual_artifacts:exported.directory,production_model_qualified:false});const state:Installed','checkpoint_path:path,production_model_qualified:false});const state:Installed');
  body=replace(body,'cap,owner,receipt,router,closed:false','cap,owner,receipt,closed:false');
  const residual=body.indexOf('export async function coreConstraintResidual('),after=body.indexOf('export async function restoreCorePrevious(',residual);let part=body.slice(residual,after);part=replace(part,'await verifyCoreExport(cap,exported.directory,worker.pid);','');body=body.slice(0,residual)+part+body.slice(after);
  return body+nativeEvolutionRestoreCoreSource;
 });
 await edit('native-evolution-producer/producer.ts',body=>{
  body='import {beginNativeEvolutionRound,restoreNativeEvolutionRound,type NativeEvolutionAmbient} from '+ambient+';\n'+body;
  body=replace(body,"const previous=await control.captureRouterInitial(join(r.work,'ambient-previous')),branch=peer.worker.isBranchEnabled;","const previous=await beginNativeEvolutionRound(r.engine,guard,r.owners),branch=peer.worker.isBranchEnabled;");
  body=replace(body,'r.owners.own(previous,()=>previous.close());','');
  const start=body.indexOf('const records=selectIndependentNativeEvolutionCohort(r.memory,store,base).records,first='),end=body.indexOf('let failure',start);
  if(start<0||end<start)throw Error('Producer route snapshot preflight boundary absent');body=body.slice(0,start)+body.slice(end);
  body=replace(body,'peer.worker.hasInstalledCore?await control.restoreRetainedCorePrevious(previous,routePrepared,r.owners):await control.restoreRouter(previous)','await restoreNativeEvolutionRound(r.engine,guard,previous)');
  body=replace(body,'ambient:Awaited<ReturnType<typeof control.captureRouterInitial>>','ambient:NativeEvolutionAmbient');
  body=replace(body,'ambient.snapshot.cpu_rng','ambient.rng.cpu_rng');body=replace(body,'ambient.snapshot.mps_rng!','ambient.rng.mps_rng');return body;
 });
 await edit('native-core450-runner/fresh.ts',body=>{
  const start=body.indexOf('export async function prepareCore450ProductFresh(');if(start<0)throw Error('Actual fresh product function absent');let product=body.slice(start);
  product=replace(product,'previous:NativeRouterEngineSnapshot','previous:NativeEvolutionRNG');
  product=product.replaceAll('validateNativeRouterEngineSnapshot(engine,guard,r.previous)','validateNativeEvolutionRNG(engine,guard,r.previous)');
  product=replace(product,'const previous=r.previous.snapshot;','const previous=r.previous;');
  return 'import {validateNativeEvolutionRNG,type NativeEvolutionRNG} from '+ambient+';\n'+body.slice(0,start)+product;
 });
 await edit('native-core450-owner-controller/bootstrap.ts',body=>{
  const start=body.indexOf('export async function createCore450ProductFreshCheckpoint(');let product=body.slice(start);
  product=replace(product,'previous:Awaited<ReturnType<typeof control.captureRouterInitial>>|undefined','previous:NativeEvolutionAmbient|undefined');
  product=replace(product,"previous,preparation:r.preparation,signal:r.signal});","previous:previous.rng,preparation:r.preparation,signal:r.signal});");
  const a=product.indexOf('const seedReceipt=await (control as any).seedCoreFresh(fresh),seedSnapshot='),b=product.indexOf('const trainingFresh=',a);
  if(a<0||b<a)throw Error('Original fresh seed boundary absent');
  product=product.slice(0,a)+'const {receipt:seedReceipt,rng:seedSnapshot}=await seedNativeEvolutionFresh(engine,guard,fresh,owners);\n   '+product.slice(b);
  return 'import {type NativeEvolutionAmbient} from '+ambient+';\nimport {seedNativeEvolutionFresh} from '+seed+';\n'+body.slice(0,start)+product;
 });
 await edit('native-core450-owner-controller/runner.ts',body=>{
  body='import {type NativeEvolutionAmbient} from '+ambient+';\n'+body;
  const marker=body.indexOf('export type ProductRunRequest');if(marker<0)throw Error('Product runner boundary absent');
  body=replace(body,";previous:Awaited<ReturnType<Control['captureRouterInitial']>>;final_evidence:",';previous:NativeEvolutionAmbient;final_evidence:');
  return replace(body,'previous:Awaited<ReturnType<Control["captureRouterInitial"]>>','previous:NativeEvolutionAmbient');
 });
}

export const nativeEvolutionRestoreCoreSource=`
/** Actual core-drop ACK restores the held accepted tensor. Retire its installation only after that ACK. */
export async function restoreNativeEvolutionCore(engine:NativeEngine,guard:SnapshotGuard){
 assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard),s=installations.get(engine),expected=peer.source.runtime.core_raw?.sha256??peer.base.original_down_sha256;
 const result=await peer.worker.dropCoreState(),ack=result.response;assert(ack.operation==='core-drop'&&ack.step===0&&ack.observation==='current');
 assert(ack.backbone_resident?ack.core_tensor_sha256===expected:!s&&ack.core_tensor_sha256===null,'Actual accepted down restoration ACK differs');
 if(s){await s.owner.close();assert.equal(s.owner.handle.fd,-1);s.closed=true;installations.delete(engine);}restored.delete(engine);
 assert(!peer.worker.hasInstalledCore);return Object.freeze({pid:peer.worker.pid,step:0,core_tensor_sha256:ack.core_tensor_sha256,graph_resident:ack.backbone_resident,actual_source_restored:true,production_model_qualified:false});
}
`;
