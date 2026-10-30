/** Cancellation stops new work, not the actual withdrawal of the already-installed Core. */
import assert from 'node:assert/strict';
import {join} from 'node:path';
type Replace=(body:string,before:string,after:string)=>string;
export function patchNativeEvolutionCancelCleanup(body:string,replace:Replace){
 body=replace(body,'private async check(){','private async check(restoringCore=false){');
 if(body.includes('if(await observationFastCheck(this))return;'))
  body=replace(body,'if(await observationFastCheck(this))return;','if(!restoringCore&&await observationFastCheck(this))return;');
 body=replace(body,'await this.coreSource?.assertUnchanged();',`if(restoringCore){
   // This owner was admitted by loadCoreOwner, without the cancelled training signal.
   // Its frozen tensor descriptors and original FD/stat remain valid for withdrawal.
   const owner=this.coreSourceOwner;
   if(this.coreSource){assert(owner,'Installed Core withdrawal requires its held tensor owner');
    assert.deepEqual(owner.tensor_keys.toSorted(),['candidate','master','exp_avg','exp_avg_sq','adam_step','reference','basis'].toSorted());
    const candidate=owner.tensors.candidate;assert(candidate?.dtype==='bf16'&&candidate.shape.length===2&&candidate.shape[0]===this.options.hidden&&candidate.shape[1]>0,'Installed Core withdrawal shape differs');
    await owner.assertUnchanged();
   }else assert(!owner,'Unbound Core tensor owner cannot be withdrawn');
  }else await this.coreSource?.assertUnchanged();`);
 const start=body.indexOf(' private command('),end=body.indexOf(' private sourceFields(',start);
 assert(start>=0&&end>start,'Actual retained Process command boundary missing');
 const command=body.slice(start,end);
 assert.equal(command.split('await this.check();').length,3,'Actual command must check before and after native ACK');
 return body.slice(0,start)+command.replaceAll('await this.check();',"await this.check(line==='core-drop');")+body.slice(end);
}
export async function applyNativeEvolutionCancelCleanupPatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(path:string)=>string;replace:Replace;root:string}){
 const path=target(join(root,'.native-migration/research/native-unified-expert/expert-process.ts'));
 const body=bodies.get(path);assert(body,'Current composed retained Process is required');
 bodies.set(path,patchNativeEvolutionCancelCleanup(body,replace));
 const installPath=target(join(root,'.native-migration/research/native-unified-core-control/install.ts'));
 const install=bodies.get(installPath);assert(install,'Current composed Core installer is required');
 bodies.set(installPath,patchNativeEvolutionInstallOwnership(install,replace));
}

/** A native command may have taken the FD even when its ACK is invalid or unknown. */
export function patchNativeEvolutionInstallOwnership(body:string,replace:Replace){
 const start=body.indexOf('export async function installCoreCheckpoint('),end=body.indexOf('export async function coreConstraintResidual(',start);
 assert(start>=0&&end>start,'Actual Core installation boundary missing');let install=body.slice(start,end);
 install=replace(install,'let installed=false;','let installationAttempted=false;');
 install=install.replaceAll('!installed','!installationAttempted');
 install=replace(install,'const loaded=await worker.loadCoreOwner(owner,cap.approval.step,cap);',
  'installationAttempted=true;const loaded=await worker.loadCoreOwner(owner,cap.approval.step,cap);');
 install=replace(install,'installed=true;','');
 return body.slice(0,start)+install+body.slice(end);
}
