/** Daily completed-state transport reuses its actual FD owners; numerical migration probes remain independent. */
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
type Context={bodies:Map<string,string>;target:(p:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export function patchNativeEvolutionTrainingOutput(body:string,replace:Context['replace'],helper:string){
 body='import {isNativeEvolutionClone,retainNativeEvolutionVerifiedRaw} from '+JSON.stringify(helper)+';\n'+body;
 body=replace(body,'need(await checkpointRawSHA(file.handle,file.bytes,file.dtype)===spec.sha256,"Native training raw SHA/finite mismatch");await unchanged(path,file.handle,file.stat);',
 'need(root===r.input&&isNativeEvolutionClone(path,file.stat,spec.sha256)||await checkpointRawSHA(file.handle,file.bytes,file.dtype)===spec.sha256,"Native training raw SHA/finite mismatch");await retainNativeEvolutionVerifiedRaw(file,file.stat);');
 body=replace(body,'await unchanged(file.path,file.handle,file.stat);need(await checkpointRawSHA(file.handle,file.bytes,file.dtype)===file.sha256,"Native training raw changed after read");','await unchanged(file.path,file.handle,file.stat);');
 body=replace(body,'await unchanged(file.path,file.handle,file.stat);need(await checkpointRawSHA(file.handle,file.bytes)===file.sha256,"Native training metadata changed after read");','await unchanged(file.path,file.handle,file.stat);');
 body=replace(body,'await stateInvariant(Object.fromEntries(FINAL.map(key=>[key,inputFiles[key]])) as NativeTrainingState,integer(input.config.training_offset??"0"));',
 'const productInputs=version===3&&!live&&Object.values(inputFiles).every(file=>isNativeEvolutionClone(file.path,file.stat,file.sha256));\n    if(productInputs)need((await bytes(inputFiles.adam_step!.handle,4)).readFloatLE(0)===integer(input.config.training_offset??"0"),"Native training Adam cumulative step differs");else await stateInvariant(Object.fromEntries(FINAL.map(key=>[key,inputFiles[key]])) as NativeTrainingState,integer(input.config.training_offset??"0"));');
 body=replace(body,'outputFiles[name]=await tensor(r.output,spec);','if(!productInputs||STATE.includes(key as any))outputFiles[name]=await tensor(r.output,spec);');
 body=replace(body,'const offset=integer(input.config.training_offset??"0");for(const step of selected)await stateInvariant(state(step),offset+step);if(!live)await stateInvariant(state(),offset+input.steps.length);',
 'const offset=integer(input.config.training_offset??"0");for(const step of selected){if(productInputs&&step===input.steps.length){for(const key of STATE)need(state(step)[key].sha256===state()[key].sha256,"Native training last checkpoint differs from final state");}else await stateInvariant(state(step),offset+step);}if(!live)await stateInvariant(state(),offset+input.steps.length);');
 return body;
}
export async function applyNativeEvolutionOutputPatches({bodies,target,replace,root}:Context){
 const helper=join(root,'backend/native-evolution-segment-files.ts');
 async function edit(relative:string,change:(body:string)=>string){const path=target(join(root,'.native-migration',relative));bodies.set(path,change(bodies.get(path)??await readFile(path,'utf8')));}
 await edit('backend/native-training-output.ts',body=>patchNativeEvolutionTrainingOutput(body,replace,helper));
 await edit('backend/native-training-mps-output.ts',body=>{
  body=replace(body,"await unchanged(f.path,f.handle,f.stat);need(await checkpointRawSHA(f.handle,f.bytes,'u8')===f.sha256,'V3 captured RNG changed');",'await unchanged(f.path,f.handle,f.stat);');
  return replace(body,"await unchanged(f.path,f.handle,f.stat);need(await checkpointRawSHA(f.handle,f.bytes)===f.sha256,'V3 RNG metadata changed');",'await unchanged(f.path,f.handle,f.stat);');
 });
 await edit('research/native-core450-runner/candidate.ts',body=>replace(body,'for(const raw of [...Object.values(cap.state),cap.cpu_rng,cap.mps_rng])assert.equal(await checkpointRawSHA(raw.handle,raw.bytes,raw.dtype),raw.sha256);',''));
 for(const relative of new Set(['backend/native-checkpoint.ts','research/native-product-runtime/backend/native-checkpoint.ts','research/native-unified-service/checkpoint-owner.ts'].map(p=>target(join(root,'.native-migration',p))))) {
 bodies.set(relative,patchNativeEvolutionCheckpointRead(bodies.get(relative)??await readFile(relative,'utf8'),replace,helper));}
 await edit('research/native-core450-runner/checkpoint.ts',body=>{
  body='import {isNativeEvolutionVerifiedRaw} from '+JSON.stringify(helper)+';\n'+body;
  return replace(body,'for(const f of cp.files)assert.equal(await checkpointRawSHA(f.handle,f.bytes,f.dtype),f.sha256);','for(const f of cp.files)assert(await isNativeEvolutionVerifiedRaw(f),"Checkpoint raw owner was not admitted");');
 });
 for(const path of new Set(['backend/native-checkpoint-save.ts','research/native-learning-product/checkpoint-save-product.ts'].map(p=>target(join(root,'.native-migration',p)))))bodies.set(path,patchNativeEvolutionCheckpointSave(bodies.get(path)??await readFile(path,'utf8'),replace,helper));
 await edit('research/native-unified-service/checkpoint-journal.ts',body=>patchNativeEvolutionCheckpointSourceSHA(body,replace));
}

export function patchNativeEvolutionCheckpointSave(body:string,replace:Context['replace'],helper:string){
  body='import {isNativeEvolutionVerifiedRaw,retainNativeEvolutionVerifiedRaw} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,'need(await checkpointRawSHA(f.handle,f.bytes,f.dtype)===f.sha256,"Checkpoint raw source SHA/finite differs");bound.push({...f,stat});',
   'if(!await isNativeEvolutionVerifiedRaw(f)){need(await checkpointRawSHA(f.handle,f.bytes,f.dtype)===f.sha256,"Checkpoint raw source SHA/finite differs");await retainNativeEvolutionVerifiedRaw(f,stat);}bound.push({...f,stat});');
  return replace(body,'await unchanged(f.path,f.handle,f.stat);need(await checkpointRawSHA(f.handle,f.bytes,f.dtype)===f.sha256,"Checkpoint source changed during save");','await unchanged(f.path,f.handle,f.stat);');
}

export function patchNativeEvolutionCheckpointRead(body:string,replace:Context['replace'],helper:string){
  body='import {retainNativeEvolutionVerifiedRaw,retainNativeEvolutionScratch,removeNativeEvolutionScratch} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,'async function fileSHA(handle: FileHandle, bytes: number, dtype?: CheckpointTensor["dtype"]) {','async function fileSHA(handle: FileHandle, bytes: number, dtype?: CheckpointTensor["dtype"], signal?: AbortSignal) {');
  body=replace(body,'  while (position < bytes) {','  while (position < bytes) {\n    signal?.throwIfAborted();');
  body=replace(body,'await unchanged(path, file.handle, file.stat); files.push({ ...item, path, handle: file.handle, byte_order: "little" });','await retainNativeEvolutionVerifiedRaw({...item,path,handle:file.handle},file.stat); files.push({ ...item, path, handle: file.handle, byte_order: "little" });');
  // The native reader exclusively created this output; retain it only after complete validation.
  body=replace(body,'    let closed = false;','    await retainNativeEvolutionScratch(directory);\n    let closed = false;');
  const old=body.includes('"Checkpoint owner FD close unconfirmed"')
   ? 'close: async () => { if (!closed) { const results = await Promise.allSettled(handles.filter(handle => handle.fd !== -1).map(handle => handle.close())); need(results.every(result => result.status === "fulfilled") && handles.every(handle => handle.fd === -1), "Checkpoint owner FD close unconfirmed"); closed = true; } }'
   : 'close: async () => { if (!closed) { closed = true; await Promise.all(handles.map(handle => handle.close())); } }';
  return replace(body,old,'close: async () => { if (!closed) { const results = await Promise.allSettled(handles.filter(handle => handle.fd !== -1).map(handle => handle.close())); need(results.every(result => result.status === "fulfilled") && handles.every(handle => handle.fd === -1), "Checkpoint owner FD close unconfirmed"); await removeNativeEvolutionScratch(directory); closed = true; } }');
}

/** Keep the journal's complete PT check, using its existing bounded explicit-read implementation. */
export function patchNativeEvolutionCheckpointSourceSHA(body:string,replace:Context['replace']){
 body='import {checkpointRawSHA} from "./checkpoint-owner.ts";\n'+body;
 const begin=body.indexOf('async function sourceSHA('),end=body.indexOf('\n}',begin)+2;
 if(begin<0||end<=begin)throw Error('Checkpoint journal sourceSHA consumer absent');
 let source=body.slice(begin,end);
 source=replace(source,'  const file = await openNativeFile(path, Number(stat.size));','  signal?.throwIfAborted();\n  const file = await openNativeFile(path, Number(stat.size));');
 const readStart=source.indexOf('    const hash = createHash("sha256");'),readEnd=source.indexOf('    await unchangedNativeFile',readStart);
 if(readStart<0||readEnd<readStart)throw Error('Checkpoint journal stream consumer absent');
 source=source.slice(0,readStart)+'    const digest = await checkpointRawSHA(file.handle, Number(stat.size), undefined, signal);\n    signal?.throwIfAborted();\n'+source.slice(readEnd);
 source=replace(source,'hash.digest("hex") === sha256','digest === sha256');
 source=replace(source,'finally { await file.handle.close(); }','finally { await file.handle.close(); need(file.handle.fd === -1, "Checkpoint source FD close unconfirmed"); }');
 return body.slice(0,begin)+source+body.slice(end);
}
