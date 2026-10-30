import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
export async function applyNativeEvolutionSegmentPatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(p:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string}){
 const research=join(root,'.native-migration/research'),helper=JSON.stringify(join(root,'backend/native-evolution-segment-files.ts'));
 const exporter=target(join(research,'native-evolution-fixture/exporter.ts'));let body=bodies.get(exporter)??await readFile(exporter,'utf8');
 body=replace(body,'let closed=false;return {directory:destination,','let closed=false;return {tensorOwner(name:string){lock();need(!closed,"Fixture owner closed");const spec=tensors[name];need(spec,"Unknown held fixture tensor");const file=outputFiles.find(file=>file.path===join(destination,spec.file));need(file,"Fixture tensor owner missing");return file;},directory:destination,');bodies.set(exporter,body);
 const segment=target(join(research,'native-core450-runner/segment.ts'));body=bodies.get(segment)!;
 body="import {core450ControlMetadata} from '../native-core450-owner-controller/metadata.ts';\n"+'import {cloneNativeEvolutionInput} from '+helper+';\n'+body;
 body=replace(body,'nextCore450TrainingStep(metadata(cap.checkpoint.payload),groups)','nextCore450TrainingStep(core450ControlMetadata(cap.checkpoint.payload),groups)');
 const begin=body.indexOf(' for(const[name,spec]of Object.entries(approval.fixture.tensors)){'),end=body.indexOf('\n const raw=deriveCore450SegmentManifest',begin);if(begin<0||end<begin)throw Error('Segment copy loop missing');
 body=body.slice(0,begin)+` for(const[name,spec]of Object.entries(approval.fixture.tensors)){
  signal?.throwIfAborted();const original=updated[name]??(legacy?approval.files.get(join(fixtureDirectory,spec.file)):approval.fixture.tensorOwner(name));assert(original);
  const dtype=updated[name]?.dtype??spec.dtype,bytes=updated[name]?.bytes??spec.bytes,sha256=original.sha256,shape=updated[name]?.shape??spec.shape;assert.equal(bytes,spec.bytes);assert.deepEqual(shape,spec.shape);assert.equal(dtype,spec.dtype);
  const before=await original.handle.stat({bigint:true});await unchangedNativeFile(original.path,original.handle,before);
  if(!updated[name]&&!legacy)await unchangedNativeFile(original.path,original.handle,approval.fixture.tensorOwner(name).stat);
  const path=join(input,spec.file);await cloneNativeEvolutionInput({path:original.path,handle:original.handle,stat:before,sha256},path,signal);
  mapping.push({name,path,source:original.path,bytes,dtype,shape:[...shape],sha256});
 }`+body.slice(end);bodies.set(segment,body);
 const processPath=target(join(research,'native-core450-runner/process.ts'));body=bodies.get(processPath)!;
 body='import {isNativeEvolutionClone} from '+helper+';\n'+body;
 body=replace(body,'need(await digest(f.handle, size) === pin,','need(isNativeEvolutionClone(path,f.stat,pin)||await digest(f.handle, size) === pin,');
 body=replace(body,'need(await digest(f.handle, Number(f.stat.size)) === f.sha256,','need(isNativeEvolutionClone(f.path,f.stat,f.sha256)||await digest(f.handle, Number(f.stat.size)) === f.sha256,');bodies.set(processPath,body);
 const runner=target(join(research,'native-core450-owner-controller/runner.ts'));body=bodies.get(runner)!;
 body='import {retainNativeEvolutionScratch,removeNativeEvolutionScratch} from '+helper+';\n'+body;
 body=replace(body,"export type ProductRunRequest=Omit<Request,'preparation'>&{preparation:NativeEvolutionPrepared;", "export type ProductRunRequest=Omit<Request,'preparation'>&{onProgress?:(progress:{step:number;steps:number;phase:string})=>void;preparation:NativeEvolutionPrepared;");
 const start=body.indexOf('export async function runCore450ProductFromCheckpoint(');if(start<0)throw Error('Product runner absent');let product=body.slice(start);
 product=replace(product,"const entry=join(r.directory,'entry');await mkdir(entry,{mode:0o700});", "const entry=join(r.directory,'entry');await mkdir(entry,{mode:0o700});await retainNativeEvolutionScratch(entry);let currentDirectory=entry;");
 product=replace(product,"await mkdir(directory,{mode:0o700});\n    const request=", "await mkdir(directory,{mode:0o700});await retainNativeEvolutionScratch(directory);\n    const request=");
 product=replace(product,"await mkdir(loadedDirectory,{mode:0o700});", "await mkdir(loadedDirectory,{mode:0o700});await retainNativeEvolutionScratch(loadedDirectory);");
 product=replace(product,'await owners.release(oldCandidate);candidate=undefined;await owners.release(oldCurrent);','await owners.release(oldCandidate);candidate=undefined;await owners.release(oldCurrent);\n    await removeNativeEvolutionScratch(directory);await removeNativeEvolutionScratch(currentDirectory);currentDirectory=loadedDirectory;');
 product=replace(product,"if(current.approval.step===450)phase='awaiting_quality';", "r.onProgress?.({step:current.approval.step,steps:450,phase:current.approval.step===450?'awaiting_quality':'training'});if(current.approval.step===450)phase='awaiting_quality';");
 body=body.slice(0,start)+product;bodies.set(runner,body);
 const job=target(join(research,'native-core-product-job-source-v1/job.ts'));body=bodies.get(job)!;
 body=replace(body,"onFinal?:ProductRunRequest['onFinal']", "onProgress?:ProductRunRequest['onProgress'];onFinal?:ProductRunRequest['onFinal']");
 body=replace(body,'const common={automatic:r.automatic,','const common={onProgress:r.onProgress,automatic:r.automatic,');bodies.set(job,body);
}
