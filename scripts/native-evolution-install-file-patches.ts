import assert from 'node:assert/strict';
import {join} from 'node:path';
type Replace=(body:string,before:string,after:string)=>string;
export function patchNativeEvolutionInstallFile(body:string,replace:Replace){
 const start=body.indexOf('export async function installCoreCheckpoint('),end=body.indexOf('export async function coreConstraintResidual(',start);
 assert(start>=0&&end>start);let part=body.slice(start,end);
 part=replace(part,'let installationAttempted=false;', 'owner=await retainNativeEvolutionInstallFile(owner);let installationAttempted=false;');
 // A prior close may have released the FD but failed unlink; retain its cleanup retry.
 part=replace(part,"if(owner.handle.fd===-1)return;assert(peer.worker.isClosed||!installationAttempted,", "assert(owner.handle.fd===-1||peer.worker.isClosed||!installationAttempted,");
 return body.slice(0,start)+part+body.slice(end);
}
export function applyNativeEvolutionInstallFilePatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(path:string)=>string;replace:Replace;root:string}){
 const path=target(join(root,'.native-migration/research/native-unified-core-control/install.ts'));
 const body=bodies.get(path);assert(body);
 bodies.set(path,'import {retainNativeEvolutionInstallFile} from '+JSON.stringify(join(root,'backend/native-evolution-install-file.ts'))+';\n'+patchNativeEvolutionInstallFile(body,replace));
}
