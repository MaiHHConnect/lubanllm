/** Reuse actual source creation without routing it through ordinary chat or taking an audit snapshot. */
import {join} from 'node:path';
import {readFile} from 'node:fs/promises';
export async function applyNativeEvolutionPeerPatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(p:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string}){
 const research=join(root,'.native-migration/research'),peer=target(join(research,'native-unified-peer/engine-peer.ts'));let body=bodies.get(peer)!;
 const start=body.indexOf('export async function captureRouterEngineInitial('),end=body.indexOf('\n const held=peer;',start);if(start<0||end<start)throw Error('Retained source initialization boundary missing');
 const original=body.slice(start,end),signature=original.slice(0,original.indexOf('{\n')+1);
 const ensure=original.replace('captureRouterEngineInitial(','ensureNativeEnginePeer(').replace('Promise<NativeRouterEngineSnapshot>','Promise<Peer>')+'\n return peer;\n}\n';
 body=body.slice(0,start)+ensure+signature+'\n const peer=await ensureNativeEnginePeer(engine,guard,o);'+body.slice(end);
 body+=`\n/** Read only this Engine's issued live source. It grants no ordinary routing qualification. */
export function readNativeRetainedSource(engine:NativeEngine){const peer=peers.get(engine);if(!peer||peer.closed)return null;peer.worker.assertLive();assert(engine instanceof NativeEngine&&engine.config.routing?.store===peer.store&&engine.config.identity===peer.base.identity&&equal(engine.config.aten?.base,peer.base)&&equal(peer.source.runtime.base,peer.base));return{base:structuredClone(peer.base),worker_pid:peer.worker.pid,binary_sha256:peer.build.binary_sha256,core_version:peer.source.runtime.core_label??null,core_tensor_sha256:peer.source.runtime.core_raw?.sha256??peer.base.original_down_sha256,expert_id:peer.previous?.id??null,graph_resident:peer.worker.status.backbone_resident===true,native_status:structuredClone(peer.worker.status),ordinary_qualified:false as const};}
`;bodies.set(peer,body);
 const descriptor=target(join(research,'native-peer-descriptor-v2/descriptor.ts'));
 bodies.set(descriptor,replace(bodies.get(descriptor)!,".replace('LUBAN_QWEN35_CACHE_NONE_INPUT_V1','LUBAN_QWEN35_CACHE_CORE_INPUT_V1')", ".replace('LUBAN_QWEN35_CACHE_NONE_INPUT_V1','LUBAN_QWEN35_CACHE_CORE_INPUT_V1').replace('config\\tinput_limit\\t512\\n','config\\tinput_limit\\t32768\\n').replace('config\\ttotal_limit\\t2560\\n','config\\ttotal_limit\\t32768\\n')"));
 const engine=target(join(research,'native-unified-expert/native-engine.ts'));body=bodies.get(engine)!;
 body="import {ensureNativeEnginePeer} from '../native-unified-peer/engine-peer.ts';\n"+body;
 body=replace(body,'    captureRouterInitial:(directory:string)=>Promise<NativeRouterEngineSnapshot>;','    openRetainedSource:(directory:string)=>Promise<void>;\n    captureRouterInitial:(directory:string)=>Promise<NativeRouterEngineSnapshot>;');
 body=replace(body,'captureRouterInitial:async(directory:string)=>{',`openRetainedSource:async(directory:string)=>{guard.assertQuiescent();need(captureRouting==="source"&&!expertScope&&!scopedGenerationBusy&&this.config.aten&&this.config.routing&&this.config.expert_build&&!this.workerPID,"Evolution initialization requires the exclusive source scope");controlled=true;hold=true;await ensureNativeEnginePeer(this,guard,{directory,model:this.config.model,store:this.config.routing.store,base:this.config.aten.base,build:this.config.expert_build});},captureRouterInitial:async(directory:string)=>{`);bodies.set(engine,body);
 const producer=target(join(research,'native-evolution-producer/producer.ts'));body=bodies.get(producer)!;
 body=patchNativeEvolutionProducerScope(body,replace);bodies.set(producer,body);
 const base=target(join(research,'native-ordinary-snapshot-base-v1/resolver.ts'));body=bodies.get(base)??await readFile(base,'utf8');
 body="import {readNativeRetainedSource} from "+JSON.stringify(peer)+";\n"+body;
 body=replace(body,'  const { signal } = options, interrupted = () => need(!signal?.aborted, "快照基座核验已停止");','  const { signal } = options, interrupted = () => need(!signal?.aborted, "快照基座核验已停止");\n  interrupted();const retained=readNativeRetainedSource(engine);if(retained){validateBase(retained.base);return retained.base;}');bodies.set(base,body);
}

/** The existing borrow is the only authority allowed to rebind a paused peer to the new Engine guard. */
export function patchNativeEvolutionProducerScope(body:string,replace:(body:string,before:string,after:string)=>string){
 body=replace(body,' const base=await resolveNativeSnapshotBase(r.engine,{signal:r.signal});\n','');
 return replace(body,"const peer=borrowNativeEnginePeer(r.engine,guard);need(!peer.worker.hasInstalledCore,","const peer=borrowNativeEnginePeer(r.engine,guard);const base=await resolveNativeSnapshotBase(r.engine,{signal:r.signal});need(!peer.worker.hasInstalledCore,");
}
