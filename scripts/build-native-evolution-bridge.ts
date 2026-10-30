/** Rebuild only the Swift source-selection adapter; graph/trainer bytes are not touched. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join,resolve,basename} from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {checkReleaseMachO} from './release-macho.ts';
const root=resolve(import.meta.dir,'..'),output=join(root,'build/native-evolution');
const runtime=JSON.parse(await readFile(join(output,'runtime-authority.json'),'utf8'));
const manifestPath=join(runtime.runtimeRoot,runtime.authority.manifest_relative),manifest=JSON.parse(await readFile(manifestPath,'utf8'));
const argv=manifest.programs.bridge.proof.compile.argv as string[],records:any[]=[];
const sha=(data:string|Uint8Array)=>createHash('sha256').update(data).digest('hex');
async function overlay(path:string,edit:(replace:(before:string,after:string)=>void)=>void){
 const original=await readFile(path,'utf8');let body=original;const changes:any[]=[];
 const replace=(before:string,after:string)=>{assert.equal(body.split(before).length,2,before);changes.push({before,after,at:body.indexOf(before)});body=body.replace(before,after);};edit(replace);
 let inverse=body;for(const c of changes.toReversed()){assert.equal(inverse.slice(c.at,c.at+c.after.length),c.after);inverse=inverse.slice(0,c.at)+c.before+inverse.slice(c.at+c.after.length);}assert.equal(inverse,original);
 const target=join(output,'Evolution'+basename(path));await writeFile(target,body);records.push({path,target,before_sha256:sha(original),after_sha256:sha(body),changes,inverse_exact:true});return target;
}
const router=argv.find(p=>p.endsWith('/Router.swift'))!,cli=argv.find(p=>p.endsWith('/TextCLI.swift'))!;
const routerOutput=await overlay(router,replace=>{
 replace('struct NativeRouter {',`func nativeEvolutionCoreQualified(_ meta: [String: Any], _ files: RouterFiles) throws -> Bool {
    guard let name = meta["report"] as? String, routerMatches(name, "^evolution-[a-f0-9]{12}\\\\.json$") else { return true }
    guard let bytes = try files.read(name, limit: 16 << 20, optional: true), let report = try? routerJSON(bytes),
          let gates = report["gates"] as? [String: Any] else { return false }
    for key in ["protected_core_retention_generation", "protected_core_retention_loss", "protected_replay_generation", "protected_replay_loss_retained"] {
        guard let value = gates[key] as? Bool, value else { return false }
    }
    return (report["core_retention_loss"] as? [String: Any])?["passed"] as? Bool == true && report["core_retention_generation"] is [Any]
}
struct NativeRouter {`);
 replace('var coreVersion: String?, coreReceipt: [String: Any]?,','var coreReview: [String: Any]?, physicalCore: [String: Any]?\n        var coreVersion: String?, coreReceipt: [String: Any]?,');
 replace('\n                && (try routerBoolean(loaded["core_loaded"])) && loaded["core_file_sha256"] as? String == sha\n                && loaded["core_tensor_sha256"] as? String == tensors[coreParameter]?.sha,',',');
 replace('            coreVersion = version; coreReceipt = ["version": version,', '            physicalCore = ["version": version,');
 replace('"metadata_sha256": routerSHA(metadataData), "pointer_sha256": routerSHA(pointer)]',`"metadata_sha256": routerSHA(metadataData), "pointer_sha256": routerSHA(pointer)]
            if try nativeEvolutionCoreQualified(meta, files) {
                try routerNeed((try routerBoolean(loaded["core_loaded"])) && loaded["core_file_sha256"] as? String == sha
                    && loaded["core_tensor_sha256"] as? String == tensors[coreParameter]?.sha, "有效核心与实际装载不一致")
                coreVersion = version; coreReceipt = physicalCore
            } else {
                try routerNeed(!(try routerBoolean(loaded["core_loaded"])) && (loaded["core_file_sha256"] == nil || loaded["core_file_sha256"] is NSNull)
                    && (loaded["core_tensor_sha256"] == nil || loaded["core_tensor_sha256"] is NSNull), "需复验核心必须实际回退原始模型")
                coreReview = ["version": version, "qualified": false, "reason": "核心缺少当前逐条保护复验"]
            }`);
 replace('"validated_expert": expertReceipt as Any? ?? NSNull(),','"physical_core": physicalCore as Any? ?? NSNull(), "core_requires_review": coreReview as Any? ?? NSNull(),\n            "validated_expert": expertReceipt as Any? ?? NSNull(),');
});
const cliOutput=await overlay(cli,replace=>replace('[("current_core", "version"), ("router", "id")]','[("physical_core", "version"), ("router", "id")]'));
const path=join(runtime.runtimeRoot,runtime.roles.bridge.path),commands:any[]=[];
function run(args:string[]){const p=spawnSync(args[0]!,args.slice(1),{encoding:'utf8',timeout:90000});commands.push({argv:args,exit_code:p.status});assert.equal(p.status,0,p.stderr);}
run(argv.map((value,index)=>value===router?routerOutput:value===cli?cliOutput:index===argv.indexOf('-o')+1?path:value));
const originalRaw=await readFile(path),original={bytes:originalRaw.length,sha256:sha(originalRaw)};
run(['/usr/bin/strip','-S','-x',path]);const strippedRaw=await readFile(path),stripped={bytes:strippedRaw.length,sha256:sha(strippedRaw)};
run(['/usr/bin/codesign','--force','--sign','-',path]);run(['/usr/bin/codesign','--verify','--strict',path]);
const raw=await readFile(path),final={bytes:raw.length,sha256:sha(raw)};checkReleaseMachO(raw,path);
const build={schema:'luban-native-effective-core-bridge-v1',sources:records,commands,graph_changed:false,trainer_changed:false,model_loaded:false};
await writeFile(join(output,'effective-core-bridge-source-build.json'),JSON.stringify(build,null,2)+'\n');
const proof={kind:'bridge',source_build_sha256:sha(JSON.stringify(build)),original_binary_sha256:original.sha256,approved_binary_sha256:original.sha256,compile:commands[0],inverse_verified:true};
manifest.programs.bridge.proof=proof;runtime.authority.sources.bridge=sha(JSON.stringify(proof));
manifest.payload[runtime.roles.bridge.path]={...manifest.payload[runtime.roles.bridge.path],original,copied:original,stripped,relocated:stripped,final,commands:commands.slice(1),debug_error:null};
runtime.roles.bridge.sha256=final.sha256;
const text=JSON.stringify(manifest,null,2)+'\n';await writeFile(manifestPath,text);runtime.manifestBytes=Buffer.byteLength(text);runtime.authority.manifest_sha256=sha(text);
await writeFile(join(output,'runtime-authority.json'),JSON.stringify(runtime,null,2)+'\n');console.log(JSON.stringify({path,final,graph_changed:false,trainer_changed:false,model_loaded:false}));
