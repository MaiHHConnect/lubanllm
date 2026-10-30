import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,open,lstat,realpath} from 'node:fs/promises';
import {basename,dirname,resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
import * as files from '../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
import {openNativeFile,unchangedNativeFile} from '../.native-migration/research/native-product-runtime/backend/native-features.ts';
import {floatValue,parseJSON} from '../.native-migration/research/native-product-runtime/backend/wire.ts';
import {coreCheckpointField as field,coreCheckpointMetadata as plain,coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {nativePartialCorrectIDs} from '../.native-migration/research/native-core-controller-candidate/progress.ts';
import {CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary} from '../.native-migration/research/native-core-controller-candidate/segments.ts';
import {validateGenerationCheck,isFiniteGenerationCheck,evaluateGeneration} from '../.native-migration/research/native-unified-service/generation-checks-v4.ts';
import {finishNativeEvolutionQuality,nativeEvolutionProgress,PARTIAL_GATES} from '../backend/native-evolution-quality.ts';
import {uniqueNativeEvolutionRecords} from '../backend/native-evolution-material.ts';
import {encoded} from '../backend/canonical-json.ts';
import evidence from '../research/native_evolution_final_policy_20261017.json';
const code=await readFile(resolve('build/native-evolution/worker.js'),'utf8');
function fn(name:string){const start=code.search(new RegExp('(?:async )?function '+name+'\\(')),end=code.indexOf('\n}',start);assert(start>=0&&end>start);return code.slice(start,end+2);}
const source=code.slice(code.indexOf('// .native-migration/research/native-core450-runner/resume.ts'),code.indexOf('// .native-migration/research/native-unified-service/checkpoint-owner.ts')).replace(/^import .*\n/gm,'');
const safety=code.slice(code.indexOf('var CORE450_SAFETY_POLICY ='),code.indexOf('// .native-migration/research/native-core450-runner/resume.ts'));
const env={assert12:assert,assert17:assert,lstat7:lstat,realpath6:realpath,basename8:basename,dirname13:dirname,resolve8:resolve,createHash9:createHash,
 encoded2:files.encoded,hash2:files.hash,need6:files.need,object6:files.object,readSafe2:files.readSafe,digest2:files.digest,ledgerParse2:parseJSON,floatValue2:floatValue,openRegular2:openNativeFile,unchanged3:unchangedNativeFile,
 validateGenerationCheck,isFiniteGenerationCheck,evaluateGeneration,CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary,nativePartialCorrectIDs,exposures:coreNewExposures,eq:pythonMetadataEqual,ordinary:validateOrdinaryCoreHistory,
 finishNativeEvolutionQuality,nativeEvolutionProgress,PARTIAL_GATES,uniqueNativeEvolutionRecords,encoded3:encoded};
const policyLine=source.split('\n').find(l=>l.includes('Core450 safety source policy differs'))!;assert(policyLine);
const runtime=new Function(...Object.keys(env),source+'\n'+safety+'\n'+fn('validateCore450DurableFinal')+'\nreturn {admitNativeCoreResume,validateCore450DurableFinal,validateCore450TrainingHistory,checkPolicy(expected,report){'+policyLine+'}};')(...Object.values(env));
const metadata=()=>Object.fromEntries((evidence.payload_metadata.entries as any[]).map(([k,v])=>[k.value,plain(v)]));
const constructionSource=await readFile(resolve('backend/native-evolution-final.ts'),'utf8');
const begin=constructionSource.indexOf('  const finalReport ='),end=constructionSource.indexOf('\n  const tree =',begin);
const construct=new Function('report','quality','progress','floatValue','baselineReplay','baselineNew','validationReplay','validationNew','lossRecords',constructionSource.slice(begin,end)+'\nreturn finalReport;');
function fixedSaved(){const saved=metadata(),original={...saved.report,partial_policy:'closed-book-growth-450-v1'},quality={...saved.report};
 saved.report=construct(original,quality,saved.report.learning_progress,floatValue,Number(saved.report.baseline_replay_loss),saved.report.baseline_new_loss===null?null:Number(saved.report.baseline_new_loss),Number(saved.report.validation_replay_loss),saved.report.validation_new_loss===null?null:Number(saved.report.validation_new_loss),[...evidence.records.protection,...evidence.records.retention_regressions]);return saved;}

test('actual FINAL construction preserves fixed450 policy through original reader and complete durable validation',()=>{
 const old=metadata();expect(()=>runtime.checkPolicy({steps:450},old.report)).toThrow('Core450 safety source policy differs');
 const saved=fixedSaved();expect(saved.report.partial_policy).toBe('closed-book-growth-450-v1');
 runtime.checkPolicy({steps:450},saved.report);
 const result=runtime.validateCore450DurableFinal(saved,evidence.records,evidence.summary.run_id,evidence.candidate.sha256,evidence.reference.sha256);
 expect(result.gates).toEqual(evidence.summary.gates);expect(result.promoted).toBe(false);expect(result.training_safety.complete_training_safety).toBe(true);
 expect(result.partial_policy).toBe('closed-book-growth-v1'); // The standalone quality report keeps its own contract.
 for(const policy of ['unknown','closed-book-growth-v1',null])expect(()=>runtime.checkPolicy({steps:450},{...saved.report,partial_policy:policy})).toThrow();
 const damaged=metadata();damaged.report={...saved.report,training_safety_policy:'unknown'};expect(()=>runtime.checkPolicy({steps:450},damaged.report)).toThrow();
 const forged=fixedSaved();forged.report.gates.protected_core_retention_loss=true;expect(()=>runtime.validateCore450DurableFinal(forged,evidence.records,evidence.summary.run_id,evidence.candidate.sha256,evidence.reference.sha256)).toThrow();
 const omitted=fixedSaved();omitted.report.training_selection.results.pop();expect(()=>runtime.validateCore450DurableFinal(omitted,evidence.records,evidence.summary.run_id,evidence.candidate.sha256,evidence.reference.sha256)).toThrow();
});

test.skipIf(!process.env.LUBAN_FINAL_RAW)('actual native-reader FINAL raw passes complete resume admission after only in-memory producer policy correction',async()=>{
 const directory=process.env.LUBAN_FINAL_RAW!,tree=JSON.parse(await readFile(join(directory,'checkpoint-tree.json'),'utf8')),manifest=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));
 const payload=field(tree,'payload')!,reportNode=field(payload,'report')!;
 assert(reportNode.kind==='dict');const policy=field(reportNode,'partial_policy')!;assert(policy.kind==='string');expect(policy.value).toBe('closed-book-growth-v1');policy.value=fixedSaved().report.partial_policy;
 const filesOpened:any[]=[];let admitted:any;
 try{
  for(const descriptor of manifest.tensors){const path=join(directory,descriptor.file);filesOpened.push({...descriptor,path,handle:await open(path,'r'),byte_order:'little'});}
  const checkpoint={tree,binding:field(tree,'binding')!,payload,files:filesOpened,rng:field(tree,'rng'),mps_rng:field(tree,'mps_rng')};
  const native=field(payload,'native_state')!,original=field(payload,'learning_reference') as any,report=plain(reportNode);
  admitted=await runtime.admitNativeCoreResume({checkpoint,expected:{id:evidence.summary.run_id,binding:parseJSON(evidence.original_binding_json),report,groups:evidence.records.groups,steps:450,parameter:{name:report.parameter,shape:original.shape},original:{path:join(directory,original.file),sha256:original.sha256,shape:original.shape,dtype:'bf16'},frozen:{basis_sha256:(field(native,'basis')as any).sha256,reference_f32_sha256:(field(native,'reference')as any).sha256}}});
  expect(admitted.semantic_admitted).toBe(true);expect(admitted.step).toBe(450);expect(admitted.native.reasons).toEqual([]);expect(admitted.native.compatible).toBe(true);
  expect(admitted.native.state.candidate.sha256).toBe(evidence.candidate.sha256);
  const saved={...metadata(),report};const quality=runtime.validateCore450DurableFinal(saved,evidence.records,evidence.summary.run_id,evidence.candidate.sha256,evidence.reference.sha256);expect(quality.promoted).toBe(false);expect(quality.gates.protected_core_retention_loss).toBe(false);
 }finally{await admitted?.close();for(const file of filesOpened)await file.handle.close();}
},120000);
