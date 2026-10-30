/** CPU-only actual product fixture + current release admission; no model, trainer or GPU execution. */
import assert from 'node:assert/strict';
import {join,resolve,dirname} from 'node:path';
import {mkdtemp,realpath,readFile,writeFile,unlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {constants,openSync,closeSync} from 'node:fs';
import {dlopen} from 'bun:ffi';
import {nativeEvolutionRuntimeResources,NativeCore450Owner} from '../backend/native-evolution-worker.ts';
import {withNativeEvolutionRelease,closeNativeEvolutionRelease} from '../backend/native-evolution-release.ts';
import {admitReceipt,verifyNativeTrainingMPSPackage} from '../.native-migration/research/native-unified-service/training-programs-v6.ts';
import {exportNativeEvolutionFixture} from '../.native-migration/research/native-evolution-fixture/exporter.ts';
import {parseV3Fixture} from '../.native-migration/research/native-evolution-fixture/parser-readback.ts';
import {openCheckpointRaw,checkpointRawSHA} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
const project=process.cwd(),work=await realpath(await mkdtemp(join(tmpdir(),'native-real-fixture-'))),source=resolve(project,'.native-migration/research/native-evolution-fixture/literal-fixture.ts');
let literal=await readFile(source,'utf8');literal=literal.replace(/from\s*(["'])([^"']+)\1/g,(_all,_q,p)=>'from '+JSON.stringify(p.startsWith('.')?resolve(dirname(source),p):p));await writeFile(join(work,'literal.ts'),literal);
const{fixtureCase}=await import(join(work,'literal.ts')),c=await fixtureCase(3,0,450),identity=c.request.expected.base.identity;
const lockPath='/tmp/luban-model-locks-'+process.getuid!()+'/'+identity+'-mps.lock',lock=openSync(lockPath,constants.O_CREAT|constants.O_EXCL|constants.O_RDWR|constants.O_NOFOLLOW,0o600),system=dlopen('/usr/lib/libSystem.B.dylib',{flock:{args:['i32','i32'],returns:'i32'}});assert.equal(system.symbols.flock(lock,2|4),0);
const owner={},owners=new NativeCore450Owner(),resources=nativeEvolutionRuntimeResources(),handles:Awaited<ReturnType<typeof openCheckpointRaw>>[]=[];let result:any;
try{await withNativeEvolutionRelease(owner,async()=>{
 const first=await admitReceipt(resources.release,'trainer');
 try{
 const execution={package_receipt:resources.release.path,package_receipt_sha256:resources.release.sha256,binary:first.binary,binary_sha256:first.binary_sha256,...resources.training};
 for(const key of ['package_receipt_sha256','binary_sha256','source_build_receipt_sha256','training_source_sha256','rng_source_sha256','cpu_rng_helper_sha256','mps_rng_helper_sha256']as const)await assert.rejects(verifyNativeTrainingMPSPackage({...execution,[key]:'0'.repeat(64)}));
 const rng:any={mode:'fresh'};
 for(const[kind,bytes]of [['cpu',5056],['mps',44]]as const){const path=join(c.root,'sources',kind+'_rng.raw');await writeFile(path,Buffer.alloc(bytes,kind==='cpu'?17:23),{mode:0o600,flag:'wx'});const held=await openCheckpointRaw(path);handles.push(held);const sha256=await checkpointRawSHA(held.handle,bytes,'u8');const raw={path,handle:held.handle,dtype:'u8',shape:[bytes],bytes,sha256,assertUnchanged:()=>unchangedNativeFile(path,held.handle,held.stat)};rng[kind]=raw;c.request.expected.tensors[kind+'_rng']={dtype:'u8',shape:[bytes],sha256,source_sha256:sha256};}
 const request={...c.request,owners,guard:{modelLockFd:lock,assertQuiescent(){c.request.guard.assertQuiescent();assert.equal(system.symbols.flock(lock,2|4),0);}},expected:{...c.request.expected,scope:'prepare-only',training_binary_sha256:execution.binary_sha256},device:'mps',rng,execution:{...execution,cpu_rng_sha256:rng.cpu.sha256,mps_rng_sha256:rng.mps.sha256}};
 result=await exportNativeEvolutionFixture(request);
 const parsed=parseV3Fixture(await readFile(join(result.directory,'manifest.tsv'),'utf8'),{input:result.directory,output:join(work,'never-executed'),manifest_sha256:result.manifest_sha256,binary_sha256:execution.binary_sha256,identity,device:'mps',core_shape:[2,3],reference_sha256:result.tensors.reference.sha256,basis_sha256:result.tensors.basis.sha256},{cpu_sha256:rng.cpu.sha256,mps_sha256:rng.mps.sha256});
 assert.equal(parsed.steps.length,450);assert.equal(parsed.tensors.cpu_rng.bytes,5056);assert.equal(parsed.tensors.mps_rng.bytes,44);assert.equal(result.package_validation.commands,first.commands);await result.assertUnchanged();
 await result.close();result=undefined;await first.assertUnchanged();
 const report={passed:true,work,fixture:c.root,real_release:resources.release,trainer:execution.binary_sha256,actual_package_commands:first.commands.length,shared_release_commands:true,wrong_package_pins_rejected:7,steps:parsed.steps.length,model_loaded:false,GPU:false,optimizer_steps:0};await writeFile(join(work,'result.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await result?.close();await first.close();}
});}finally{await owners.close();await closeNativeEvolutionRelease(owner);for(const h of handles)await h.handle.close();closeSync(lock);await c.close();await unlink(lockPath);await unlink('/tmp/luban-model-locks-'+process.getuid!()+'/'+identity+'-cpu.lock');}
