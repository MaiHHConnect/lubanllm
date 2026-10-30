/** Small actual exFAT/APFS native CPU fixture; never reads a production model or starts Metal. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmod, copyFile, lstat, mkdir, mkdtemp, readFile, realpath, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
import { closeSync } from "node:fs";
import { dlopen, ptr } from "bun:ffi";
import { writeEvolutionSourcePermissions } from "./native-evolution-source-permissions";
import { nativeStorageFilesystem } from "../backend/native-evolution-storage";

const root=resolve(import.meta.dir,".."), sha=(bytes:Uint8Array|string)=>createHash("sha256").update(bytes).digest("hex");
assert.equal(nativeStorageFilesystem("/Volumes/移动数据库"),"exfat");
const work=await realpath(await mkdtemp(join(tmpdir(),"luban-source-permission-CPU-")));
const evidence=await mkdtemp("/Volumes/移动数据库/lubanllm/native-source-permission-CPU-");
const overlays=await writeEvolutionSourcePermissions(root,work), binary=join(work,"source-permission-cpu"), records:any[]=[];
const prior=JSON.parse(await readFile(join(root,".native-migration/research/native-canonical-core-source-close-role-v2/build.json"),"utf8"));
const original=prior.commands.find((entry:any)=>entry.argv[0]==="/usr/bin/clang++"&&entry.argv.includes("-o")).argv as string[];
const command=original.map((arg,index)=>arg.endsWith("/PersistentMPSV3.cpp")?join(root,"native/AcceptedSourcePermissionsCPU.cpp"):index===original.indexOf("-o")+1?binary:arg);
command.push("-DLUBAN_CPU_BASE_HEADER="+JSON.stringify(join(root,".native-migration/research/native-qwen35-cache/Base.h")),
  "-DLUBAN_EXPERT_RESOURCE_HEADER="+JSON.stringify(overlays.expert),"-DLUBAN_ROUTER_RESOURCE_HEADER="+JSON.stringify(overlays.router));
async function run(argv:string[],pipeFD?:number){
  const child=spawn(argv[0]!,argv.slice(1),{stdio:pipeFD===undefined?["ignore","pipe","pipe"]:["ignore","pipe","pipe",pipeFD]}), stdout:Buffer[]=[],stderr:Buffer[]=[];
  child.stdout!.on("data",b=>stdout.push(b));child.stderr!.on("data",b=>stderr.push(b));
  const timeout=setTimeout(()=>child.kill("SIGKILL"),90000);
  const code=await new Promise<number|null>((done,reject)=>{child.once("error",reject);child.once("close",done);}).finally(()=>clearTimeout(timeout));
  const result={argv,code,stdout:Buffer.concat(stdout).toString(),stderr:Buffer.concat(stderr).toString()};records.push(result);return result;
}
const system=dlopen("/usr/lib/libSystem.B.dylib",{pipe:{args:["ptr"],returns:"i32"}}), pipe=new Int32Array([-1,-1]);
try{
  const built=await run(command);assert.equal(built.code,0,built.stderr);
  assert.equal((await run(["/usr/bin/codesign","--force","--sign","-",binary])).code,0);
  assert.equal(system.symbols.pipe(ptr(pipe)),0);
  const weights=Buffer.alloc(6144);for(let at=0;at<weights.length;at+=2)weights.writeUInt16LE(0x3b80,at);
  const router=Buffer.alloc(204);for(let at=0;at<router.length;at+=4)router.writeFloatLE(.125,at);
  const inputs=async(directory:string,mode:number)=>{await mkdir(directory,{mode:0o700});
    const expert=join(directory,"expert.raw"),head=join(directory,"router.raw");
    await writeFile(expert,weights,{mode});await writeFile(head,router,{mode});await chmod(expert,mode);await chmod(head,mode);
    return [expert,sha(weights),...Array.from({length:3},(_,i)=>sha(weights.subarray(i*2048,(i+1)*2048))),head,sha(router),sha(router.subarray(0,192)),sha(router.subarray(192))];};
  const local=await inputs(join(work,"private"),0o600), external=await inputs(join(evidence,"external"),0o600);
  assert.equal((await lstat(local[0]!)).mode&0o7777,0o600);assert.equal((await lstat(external[0]!)).mode&0o7777,0o700);
  const goodLocal=await run([binary,...local,"normal"],pipe[0]),goodExternal=await run([binary,...external,"normal"],pipe[0]);
  assert.equal(goodLocal.code,0,goodLocal.stderr);assert.equal(goodExternal.code,0,goodExternal.stderr);
  assert.deepEqual(JSON.parse(goodExternal.stdout),JSON.parse(goodLocal.stdout));
  const loose=await inputs(join(work,"loose"),0o700),permission=await run([binary,...loose,"normal"],pipe[0]);
  assert.equal(permission.code,1);assert.match(permission.stderr,/ownership\/permissions/);
  const badSHA=[...external];badSHA[1]="0".repeat(64);const corrupt=await run([binary,...badSHA,"normal"],pipe[0]);
  assert.equal(corrupt.code,1);assert.match(corrupt.stderr,/complete file SHA/);
  const badRouter=[...external];badRouter[6]="0".repeat(64);const routerCorrupt=await run([binary,...badRouter,"normal"],pipe[0]);
  assert.equal(routerCorrupt.code,1);assert.match(routerCorrupt.stderr,/Router complete file SHA/);
  const mutable=await inputs(join(evidence,"mutation"),0o600),mutated=await run([binary,...mutable,"mutate"],pipe[0]);
  assert.equal(mutated.code,1);assert.match(mutated.stderr,/Bound source changed|Wake source complete file SHA/);
  const nonfinite=await inputs(join(evidence,"nonfinite"),0o600),badBytes=Buffer.from(weights);badBytes.writeUInt16LE(0x7f80,0);
  await writeFile(nonfinite[0]!,badBytes);nonfinite[1]=sha(badBytes);nonfinite[2]=sha(badBytes.subarray(0,2048));
  const finite=await run([binary,...nonfinite,"normal"],pipe[0]);assert.equal(finite.code,1);assert.match(finite.stderr,/finite/i);
  await copyFile(binary,join(evidence,"source-permission-cpu"));
  await writeFile(join(evidence,"result.json"),JSON.stringify({passed:true,work,evidence,CPU:true,GPU:false,model_loaded:false,optimizer_steps:0,
    checks:["APFS0600 bind/wake/cold-wake","exFAT0700 bind/wake/cold-wake exact CPU output","exFAT router bind exact weights","APFS0700 rejected","expert SHA rejected","router SHA rejected","bound source mutation rejected","nonfinite expert rejected"],
    output:JSON.parse(goodExternal.stdout),records},null,2)+"\n");
  console.log(JSON.stringify({passed:true,evidence,checks:8}));
}catch(error){await writeFile(join(evidence,"failure.json"),JSON.stringify({error:String(error),work,records},null,2)+"\n");throw error;}
finally{for(const fd of pipe)if(fd>=0)closeSync(fd);}
