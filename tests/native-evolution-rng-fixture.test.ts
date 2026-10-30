import {test,expect} from 'bun:test';
import {spawnSync} from 'node:child_process';

test('actual captured RNG owners pass the original complete fixture consumer and preserve all bytes',()=>{
 const run=spawnSync(process.execPath,['--eval',String.raw`
import assert from 'node:assert/strict';import{mock}from'bun:test';import{resolve,dirname,join}from'node:path';import{mkdtemp,realpath,readFile,writeFile,mkdir,rm,unlink}from'node:fs/promises';import{constants,openSync,closeSync}from'node:fs';import{tmpdir}from'node:os';import{dlopen}from'bun:ffi';
const root=process.cwd(),old='.native-migration/research/',stub=(p,value)=>mock.module(resolve(p),()=>value),temp=await realpath(await mkdtemp(join(tmpdir(),'native-rng-fixture-')));
let peer,guard,scopeActive=true;const engine={};
stub(old+'native-unified-expert/native-engine.ts',{NativeEngine:class{},assertNativeExpertScopeGuard:(e,g)=>{assert.equal(e,engine);assert.equal(g,guard);assert(scopeActive,'scope ended');}});
stub(old+'native-unified-peer/engine-peer.ts',{borrowNativeEnginePeer:()=>peer});
stub(old+'native-unified-core-control/install.ts',{restoreNativeEvolutionCore:async()=>{throw Error('not a round restoration test');}});
let packageChecks=0;stub('.native-migration/backend/native-training-mps-package.ts',{verifyNativeTrainingMPSPackage:async()=>{packageChecks++;return{commands:[],files:[],assertUnchanged:async()=>{},close:async()=>{}};}});
const{captureNativeEvolutionTrainingRNG}=await import(resolve('backend/native-evolution-ambient.ts'));
const{NativeCore450Owner}=await import(resolve(old+'native-core450-owner-controller/owner.ts'));
const{exportNativeEvolutionFixture}=await import(resolve(old+'native-evolution-fixture/exporter.ts'));
const{parseV3Fixture}=await import(resolve(old+'native-evolution-fixture/parser-readback.ts'));
// Reuse the existing tiny CPU fixture generator, relocating only its temporary output directory.
const source=resolve(old+'native-evolution-fixture/literal-fixture.ts');let code=await readFile(source,'utf8');code=code.replace(/from\s*(["'])([^"']+)\1/g,(all,q,p)=>'from '+JSON.stringify(p.startsWith('.')?resolve(dirname(source),p):p));await writeFile(join(temp,'literal.ts'),code);
const{fixtureCase}=await import(join(temp,'literal.ts')),fixture=await fixtureCase(3,0,450),identity=fixture.request.expected.base.identity;
const system=dlopen('/usr/lib/libSystem.B.dylib',{flock:{args:['i32','i32'],returns:'i32'}}),lockPath='/tmp/luban-model-locks-'+process.getuid()+'/'+identity+'-mps.lock';
const lock=openSync(lockPath,constants.O_CREAT|constants.O_EXCL|constants.O_RDWR|constants.O_NOFOLLOW,0o600);assert.equal(system.symbols.flock(lock,2|4),0);
guard={modelLockFd:lock,assertQuiescent(){assert(scopeActive);fixture.request.guard.assertQuiescent();assert.equal(system.symbols.flock(lock,2|4),0);}};
const cpu=Buffer.alloc(5056,17),mps=Buffer.alloc(44,23),owners=new NativeCore450Owner();
peer={hashes:{'active-core.json':null},serial:0,worker:{pid:17,options:{directory:temp},async exportEvolutionRNG(name){const path=join(temp,name);await mkdir(path);await writeFile(join(path,'cpu_rng.raw'),cpu,{mode:0o600});await writeFile(join(path,'mps_rng.raw'),mps,{mode:0o600});}}};
let result;
try{
 const captured=await captureNativeEvolutionTrainingRNG(engine,guard,owners);
 for(const [name,raw]of [['cpu_rng',captured.cpu_rng],['mps_rng',captured.mps_rng]])fixture.request.expected.tensors[name]={dtype:raw.dtype,shape:[...raw.shape],sha256:raw.sha256,source_sha256:raw.sha256};
 const request={...fixture.request,guard,expected:{...fixture.request.expected,scope:'prepare-only'},device:'mps',rng:{mode:'fresh',cpu:captured.cpu_rng,mps:captured.mps_rng},execution:{binary_sha256:fixture.request.expected.training_binary_sha256,cpu_rng_sha256:captured.cpu_rng.sha256,mps_rng_sha256:captured.mps_rng.sha256}};
 const invalid=[
  {...request,execution:{...request.execution,cpu_rng_sha256:'0'.repeat(64)}},
  {...request,execution:{...request.execution,mps_rng_sha256:'0'.repeat(64)}},
  ...[{assertUnchanged:undefined},{dtype:'f32'},{shape:[1]},{bytes:1},{sha256:'0'.repeat(64)},{byte_offset:1},{file_sha256:'0'.repeat(64)}].map(change=>({...request,rng:{...request.rng,cpu:{...captured.cpu_rng,...change}}}))
 ];
 for(const input of invalid)await assert.rejects(exportNativeEvolutionFixture(input),/V3 complete borrowed RNG owner/);
 assert.equal(packageChecks,0);result=await exportNativeEvolutionFixture(request);assert.equal(packageChecks,1);
 assert((await readFile(join(result.directory,'cpu_rng.raw'))).equals(cpu));assert((await readFile(join(result.directory,'mps_rng.raw'))).equals(mps));
 const parsed=parseV3Fixture(await readFile(join(result.directory,'manifest.tsv'),'utf8'),{input:result.directory,output:join(temp,'not-run'),manifest_sha256:result.manifest_sha256,binary_sha256:request.execution.binary_sha256,identity,device:'mps',core_shape:[2,3],reference_sha256:result.tensors.reference.sha256,basis_sha256:result.tensors.basis.sha256},{cpu_sha256:captured.cpu_rng.sha256,mps_sha256:captured.mps_rng.sha256});
 assert.equal(parsed.steps.length,450);assert.equal(parsed.tensors.cpu_rng.bytes,5056);assert.equal(parsed.tensors.mps_rng.bytes,44);
 await captured.cpu_rng.assertUnchanged();await captured.mps_rng.assertUnchanged();scopeActive=false;await assert.rejects(captured.cpu_rng.assertUnchanged(),/scope ended/);scopeActive=true;
 await result.close();result=undefined;await owners.close();await assert.rejects(captured.cpu_rng.assertUnchanged(),/owner closed/);
 console.log(JSON.stringify({passed:true,negative_contracts:invalid.length,full_fixture:true,steps:450,GPU:false}));
}finally{await result?.close();await owners.close();closeSync(lock);await fixture.close();await unlink(lockPath);await unlink('/tmp/luban-model-locks-'+process.getuid()+'/'+identity+'-cpu.lock');await rm(temp,{recursive:true,force:true});}
`],{cwd:process.cwd(),encoding:'utf8',timeout:30000});
 expect(run.status,run.stderr).toBe(0);
 expect(JSON.parse(run.stdout)).toEqual({passed:true,negative_contracts:9,full_fixture:true,steps:450,GPU:false});
});
