import {test,expect} from 'bun:test';
import {mkdtemp,readFile,writeFile,mkdir,symlink,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import assert from 'node:assert/strict';
import {encoded} from '../backend/canonical-json.ts';
import {NativeQ4Observer,nativeQ4CalibrationInputs,assertNativeQ4Observation} from '../backend/native-evolution-q4.ts';
const sha=(data:Uint8Array)=>Bun.CryptoHasher.hash('sha256',data,'hex');
test('actual Swift tokenizer and resident Worker send generation then forced trace without loading a second model',async()=>{
 const work=await mkdtemp(join(tmpdir(),'native-q4-calibration-')),binary=join(work,'fixture'),observer=join(work,'observe');
 try{
  let source=await readFile('tests/native_worker_fixture.swift','utf8');
  source=source.replace('let workerArgs = Array(args.dropFirst(4))','let tracePath = args.first(where: { $0.hasPrefix("--research-trace=") }).map { String($0.dropFirst("--research-trace=".count)) }\n        let workerArgs = Array(args.dropFirst(4)).filter { !$0.hasPrefix("--research-trace=") }');
  source=source.replace('            var cancelled = false',`            if let forced = request["forced_tokens"] as? [Int], let tracePath {
                let prompt = request["tokens"] as! [Int]
                for (step, token) in [42, 248044].enumerated() {
                    var logits = Data(repeating: 0, count: 496640); logits[token*2] = 128; logits[token*2+1] = 63
                    var tail = Data(repeating: 0, count: 20480); tail[4096] = 0; tail[4097] = 64
                    let prefix = tracePath + "/" + id + "-step" + String(format: "%04d", step)
                    try tail.write(to: URL(fileURLWithPath: prefix + ".tail.bf16")); try logits.write(to: URL(fileURLWithPath: prefix + ".logits.bf16"))
                    emit(["event":"trace", "id":id,"step":step,"top1":mode == "bad-trace" ? 41 : token,"tail_path":prefix + ".tail.bf16","logits_path":prefix + ".logits.bf16","tail_only":true,"layer":23,"tail_values":10240,"vocab":248320,"committed_tokens":prompt.count+step])
                }
                emit(["event":"result","id":id,"tokens":[42,248044],"forced_tokens":forced,"generated_tokens":2,"finish_reason":"trace","complete":true,"trace_only":true]);continue
            }
            var cancelled = false`);
  source=source.replaceAll('[42, 43]','[42, 248044]');
  await writeFile(join(work,'Fixture.swift'),source);
  for(const args of [['/usr/bin/swiftc','-parse-as-library',join(work,'Fixture.swift'),'-o',binary],['/usr/bin/swiftc','native/EvolutionObserve.swift','native/Worker.swift','native/Tokenizer.swift','-o',observer]]){
   const child=Bun.spawn(args,{stdout:'pipe',stderr:'pipe'});expect(await child.exited,await new Response(child.stderr).text()).toBe(0);
  }
  await writeFile(binary+'.json',JSON.stringify({protocol:'luban-worker-jsonl-v1',binary_sha256:sha(await readFile(binary))}));
  const original='/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1',manifestSHA=sha(await readFile(join(original,'upstream-file-manifest.json')));
  for(const mode of ['normal','bad-trace']){await mkdir(join(work,mode));for(const file of ['upstream-file-manifest.json','config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'])await symlink(join(original,file),join(work,mode,file));}
  const row={id:'authorized-replay',answer:'K',prompt:'K',approved:true,training_allowed:true,purpose:'training',provenance:{evolution_role:'replay',generation_check:{type:'exact',expected:'K',max_new_tokens:8}}};
  const request={executable:observer,binary,model:join(work,'normal'),manifestSHA,identity:'b'.repeat(64),records:[row],work};
  const owner=new NativeQ4Observer(),result=await owner.captureTraining(request);await owner.close();
  expect(result.rows).toEqual([{id:row.id,positions:2}]);expect(nativeQ4CalibrationInputs(result,[row]).length).toBe(2*6144*4);expect(nativeQ4CalibrationInputs(result,[row]).readFloatLE()).toBe(2);
  expect(()=>nativeQ4CalibrationInputs({...result},[row])).toThrow("actual capture");expect(()=>nativeQ4CalibrationInputs(result,[{...row,answer:"changed"}])).toThrow("材料一致");expect(result.baseline.rows[0]!.tokens).toEqual([42,248044]);
  const preparation=await readFile('backend/native-evolution-calibration.ts','utf8'),begin=preparation.indexOf('export async function prepareNativeEvolutionCalibration('),end=preparation.indexOf(' const check=async',begin);
  const actualPrefix=preparation.slice(begin,end).replace('export async function','async function')+'return q4Raw;}\nreturn prepareNativeEvolutionCalibration;';
  const records={groups:{replay_training:[row]},protection:[{id:'old'}]},frozen={active:true,capture:result,records:encoded(records)};
  const verify=new Function('assert','scope','encoded','nativeQ4CalibrationInputs','assertNativeQ4Observation','borrowResidentNativeEnginePeer',new Bun.Transpiler({loader:'ts'}).transformSync(actualPrefix))(assert,{getStore:()=>frozen},encoded,nativeQ4CalibrationInputs,assertNativeQ4Observation,()=>({base:{identity:request.identity,model_manifest_sha256:manifestSHA},worker:{}}));
  expect((await verify({}, {}, {records,core:null})).length).toBe(2*6144*4);
  await expect(verify({}, {}, {records:{...records,protection:[{id:'changed-old'}]},core:null})).rejects.toThrow('materials changed');
  await expect(verify({}, {}, {records,core:{sha256:'c'.repeat(64)}})).rejects.toThrow('Core changed');
  frozen.active=false;await expect(verify({}, {}, {records,core:null})).rejects.toThrow('actual current');frozen.active=true;
  await expect(owner.captureTraining({...request,records:[{...row,purpose:'validation'}]})).rejects.toThrow('authorized');
  await expect(owner.captureTraining({...request,model:join(work,'bad-trace')})).rejects.toThrow('differs');await owner.close();
  expect((await owner.captureTraining(request)).rows.length).toBe(1);await owner.close();
 }finally{await rm(work,{recursive:true,force:true});}
},60000);
