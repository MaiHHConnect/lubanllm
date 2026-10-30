/** Real exFAT files and CPU native serialization. No model, GPU, trainer, teacher, or active pointer. */
import { beforeAll, afterAll, expect, test } from "bun:test";
import { createHash, randomBytes } from "node:crypto";
import { chmod, lstat, mkdir, mkdtemp, open, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { dlopen } from "bun:ffi";
import assert from 'node:assert/strict';
import { assertNativePersistentDirectory, nativeStorageFilesystem, persistNativeGeneration } from "../backend/native-evolution-storage";
import {patchNativeEvolutionCheckpointSave,patchNativeEvolutionCheckpointRead} from "../scripts/native-evolution-output-patches";
import { patchNativeCheckpointStorage } from "../scripts/native-evolution-storage-patches";

const root = resolve(import.meta.dir, ".."), sha = (v: Uint8Array | string) => createHash("sha256").update(v).digest("hex");
let local: string, external: string, api: any, memory: any, store: string, lock: any, identity: string, id: string, binding: any;
let first: any, rawFiles: any[] = [], rawHandles: any[] = [];
const checks: string[] = [], rawReadCounts=[0,0];
const system = dlopen("/usr/lib/libSystem.B.dylib", { flock: { args: ["i32", "i32"], returns: "i32" } });
beforeAll(async () => {
  expect(nativeStorageFilesystem("/Volumes/移动数据库")).toBe("exfat");
  local = await realpath(await mkdtemp(join(tmpdir(), "luban-external-save-CPU-")));
  external = await mkdtemp("/Volumes/移动数据库/lubanllm/native-evolution-storage-CPU-");
  expect(nativeStorageFilesystem(local)).toBe("apfs");
  const canonical = join(root, ".native-migration/backend"), entry = join(local, "entry.ts");
  const exports = { "native-checkpoint-save.ts": ["saveNativeTrainingCheckpoint"], "native-checkpoint.ts": ["checkpointApprovedBinding", "readNativeCheckpoint", "equalCheckpointValues"],
    "memory.ts": ["Memory", "ledgerParse"], "canonical-json.ts": ["encoded"], "native-training-journal.ts": ["trainingProcessStarted"],
    "native-model-write.ts": ["writeNativeModelTensors"], "native-model-tensor.ts": ["readNativeModelTensors"] };
  await writeFile(entry, Object.entries(exports).map(([file, names]) => `export {${names.join(",")}} from ${JSON.stringify(file === "native-checkpoint-save.ts" ? join(root,".native-migration/research/native-learning-product/checkpoint-save-product.ts") : join(canonical, file))};`).join("\n")+'\nexport {isNativeEvolutionVerifiedRaw} from '+JSON.stringify(join(root,"backend/native-evolution-segment-files.ts"))+';');
  const result = await Bun.build({ entrypoints: [entry], target: "bun", plugins: [{ name: "external-storage", setup(builder) {
    builder.onLoad({filter:/\/native-checkpoint\.ts$/},async()=>({loader:"ts",contents:patchNativeEvolutionCheckpointRead(await readFile(join(root,'.native-migration/research/native-unified-service/checkpoint-owner.ts'),"utf8"),(body,from,to)=>{expect(body.split(from)).toHaveLength(2);return body.replace(from,to);},join(root,"backend/native-evolution-segment-files.ts"))}));
    builder.onLoad({filter:/\/process-fds-v4\.ts$/},async args=>({loader:"ts",contents:(await readFile(args.path,"utf8")).replace("resolve(dirname(process.execPath),'../Resources',A.paths.processFD)",JSON.stringify(join(root,"build/native-evolution/Resources/native/bin/lib-process-fd.dylib")))}));
    builder.onResolve({filter:/native-product-runtime\/backend\/|native-unified-service\/checkpoint-owner\.ts/}, args=>({path:join(canonical,args.path.endsWith("checkpoint-owner.ts")?"native-checkpoint.ts":args.path.split("/").at(-1)!.replace(/(?:\.ts)?$/, ".ts"))}));
    builder.onLoad({ filter: /\/checkpoint-save-product\.ts$/ }, async args => ({ loader: "ts", contents: patchNativeEvolutionCheckpointSave(patchNativeCheckpointStorage(await readFile(args.path, "utf8"),
      join(root, "backend/native-evolution-storage.ts"), (body, from, to) => { expect(body.split(from)).toHaveLength(2); return body.replace(from, to); }), (body,from,to)=>{expect(body.split(from)).toHaveLength(2);return body.replace(from,to);},join(root,"backend/native-evolution-segment-files.ts")) }));
  } }] });
  expect(result.success, result.logs.map(String).join("\n")).toBe(true);
  const bundled = join(local, "storage-bundle.js"); await writeFile(bundled, Buffer.from(await result.outputs[0]!.arrayBuffer())); api = await import(bundled);
  store = join(external, "modules/luban0-v1"); await mkdir(store, { recursive: true, mode: 0o700 });
  memory = new api.Memory(external); identity = sha("CPU external checkpoint " + local); id = randomBytes(6).toString("hex"); memory.model_identity = identity;
  binding = { protocol: "durable-training-v1", identity, scope: "research", core_version: null, expert_id: null,
    device: "cpu", dtype: "torch.bfloat16", torch_version: "CPU-storage-fixture", arguments: { steps: 450 },
    packages: { transformers: "not-loaded", safetensors: "native" }, active_files_sha256: {}, implementation_sha256: {}, records_sha256: sha(api.encoded({})) };
  const body = { id, kind: "evolution", scope: "research", binding, arguments: binding.arguments, records: {},
    owner: { pid: process.pid, started: api.trainingProcessStarted(process.pid) }, progress: { step: 0, steps: 450, phase: "CPU_storage" },
    started: Date.now() / 1000, checkpoint: null, commit: null };
  memory.connect((db: any) => {
    db.exec("CREATE TABLE training_runs(id TEXT PRIMARY KEY,identity TEXT NOT NULL,kind TEXT NOT NULL,state TEXT NOT NULL,body TEXT NOT NULL,updated REAL NOT NULL); CREATE UNIQUE INDEX one_pending_training_run ON training_runs(identity) WHERE state IN ('queued','running','paused','committing');");
    db.query("INSERT INTO training_runs VALUES(?,?,?,?,?,?)").run(id, identity, "evolution", "queued", api.encoded(body), Date.now() / 1000);
  });
  const lockDirectory = "/tmp/luban-model-locks-" + process.getuid!(); await mkdir(lockDirectory, { recursive: true, mode: 0o700 });
  lock = await open(join(lockDirectory, identity + "-cpu.lock"), "wx+", 0o600); expect(system.symbols.flock(lock.fd, 6)).toBe(0);
  for (const [index, dtype, bytes] of [[0, "bf16", Buffer.from([0, 0x3f, 0, 0x40])], [1, "u8", Buffer.alloc(32, 42)]] as const) {
    const file = `tensor${index}.raw`, path = join(local, file); await writeFile(path, bytes, { flag: "wx", mode: 0o600 });
    const handle = await open(path, "r"); rawHandles.push(handle);
    const actualRead=handle.read.bind(handle);handle.read=((...args:any[])=>{rawReadCounts[index]++;return (actualRead as any)(...args);}) as any;
    rawFiles.push({ kind: "tensor", file, dtype, shape: [dtype === "bf16" ? 2 : bytes.length], bytes: bytes.length, sha256: sha(bytes), path, handle });
  }
});
afterAll(async () => {
  await first?.close(); await Promise.allSettled(rawHandles.map(handle => handle.close())); memory?.close();
  if (lock) { system.symbols.flock(lock.fd, 8); await lock.close(); await rm(`/tmp/luban-model-locks-${process.getuid!()}/${identity}-cpu.lock`); }
  if (external) await writeFile(join(external, "result.json"), JSON.stringify({ schema: "luban-native-external-storage-CPU-v1", filesystem: "exfat",
    checks, native_cpu_checkpoint: first ? { path: first.source, descriptor: first.descriptor, receipt: first.receipt } : null,
    GPU: false, model_loaded: false, optimizer_steps: 0, actual_2B_checkpoint_qualified: false }, null, 2) + "\n");
  if (local) await rm(local, { recursive: true, force: true });
  console.log("External storage CPU evidence: " + external);
});
function tree(step: number) {
  const value = (text: string) => ({ kind: "string", value: text });
  const tensor = (index: number) => { const { path, handle, ...descriptor } = rawFiles[index]; return descriptor; };
  return { kind: "dict", entries: [[value("binding"), api.checkpointApprovedBinding(binding)],
    [value("payload"), { kind: "dict", entries: [[value("id"), value(id)], [value("step"), { kind: "int", value: String(step) }], [value("candidate"), tensor(0)]] }],
    [value("rng"), tensor(1)], [value("mps_rng"), { kind: "none" }]] };
}
async function save(step: number, assertUnchanged = async () => {}) {
  const runtime = JSON.parse(await readFile(join(root, "build/native-evolution/runtime-authority.json"), "utf8"));
  return api.saveNativeTrainingCheckpoint({ memory, store, guard: { modelLockFd: lock.fd, assertQuiescent() { expect(system.symbols.flock(lock.fd, 6)).toBe(0); } },
    expected: { id, kind: "evolution", binding }, snapshot: { tree: tree(step), files: rawFiles, assertUnchanged },
    progress: { step, steps: 450, phase: "CPU_storage" }, outputParent: local,
    writer: join(runtime.runtimeRoot, runtime.roles.writer.path), writer_sha256: runtime.roles.writer.sha256,
    importer: join(runtime.runtimeRoot, runtime.roles.reader.path), importer_sha256: runtime.roles.reader.sha256 });
}
test("only real exFAT can lack private directory bits; POSIX scratch remains strict", async () => {
  await assertNativePersistentDirectory(external);
  const loose = join(local, "loose"); await mkdir(loose, { mode: 0o700 }); await chmod(loose, 0o755);
  await expect(assertNativePersistentDirectory(loose)).rejects.toThrow("Private directory"); checks.push("exFAT accepted; APFS privacy preserved");
});
test("exclusive external write rejects conflict and changed bytes, cleaning only its own failed generation", async () => {
  const source = rawFiles[0].path, destination = join(external, "exclusive.pt"), expectedSHA = rawFiles[0].sha256;
  await persistNativeGeneration({ source, destination, expectedSHA }); const original = await readFile(destination);
  await expect(persistNativeGeneration({ source, destination, expectedSHA })).rejects.toThrow(); expect(await readFile(destination)).toEqual(original);
  const failed = join(external, "failed.pt"); await expect(persistNativeGeneration({ source, destination: failed, expectedSHA: "0".repeat(64) })).rejects.toThrow("changed");
  await expect(lstat(failed)).rejects.toThrow();
  // macOS adds com.apple.provenance sidecars even to plain open/write files; only declared generation paths are read.
  expect(await readFile(destination)).toEqual(original);
  checks.push("exclusive conflict preserved original; invalid copy removed; OS sidecars do not alter declared file bytes");
});
test("native safetensors formatter stays private while its actual external file passes native readback", async () => {
  const output = join(local, "core.partial"), raw = rawFiles[0];
  const owner = await api.writeNativeModelTensors({ path: output, tensors: [{ tensor_key: "core", dtype: "BF16", shape: [2], path: raw.path, bytes: raw.bytes, sha256: raw.sha256 }] });
  try {
    const destination = join(external, "core.safetensors"); await persistNativeGeneration({ source: output, destination, expectedSHA: owner.file_sha256 });
    const readback = await api.readNativeModelTensors({ path: destination, file_sha256: owner.file_sha256, tensors: [{ tensor_key: "core", dtype: "BF16", shape: [2] }] });
    try { expect(readback.tensors.core.sha256).toBe(raw.sha256); await readback.assertUnchanged(); } finally { await readback.close(); }
  } finally { await owner.close(); await rm(output); }
  checks.push("native BF16 safetensors external readback exact");
});
test("actual native CPU writer persists exFAT checkpoint and native reader finishes before journal advances", async () => {
  first = await save(16); expect(rawReadCounts).toEqual([1,1]); expect(first.source.startsWith(store + "/training-runs/" + id + "/")).toBe(true);
  for(const raw of first.checkpoint.files)expect(await api.isNativeEvolutionVerifiedRaw(raw)).toBe(true);
  expect(first.receipt.python_invocations).toBe(0); expect(first.receipt.device).toBe("cpu"); expect(api.equalCheckpointValues(first.checkpoint.tree, tree(16))).toBe(true);
  const row = memory.connect((db: any) => db.query("SELECT body FROM training_runs WHERE id=?").get(id));
  expect(api.ledgerParse(row.body).checkpoint).toEqual(first.descriptor);
  expect((await readdir(first.work)).some(name => name.endsWith(".pt"))).toBe(false);
  await first.close();await expect(lstat(first.work)).rejects.toThrow();checks.push("actual CPU .pt external readback and durable journal advance; closed local readback reclaimed");
}, 30000);
test("actual native saves keep only current and previous external generations",async()=>{
 const old=first.source,second=await save(32);await second.close();first=await save(48);await first.close();expect(rawReadCounts).toEqual([1,1]);
 await expect(lstat(old)).rejects.toThrow();expect(await Bun.file(second.source).exists()).toBe(true);expect(await Bun.file(first.source).exists()).toBe(true);
 const row=memory.connect((db:any)=>db.query('SELECT body FROM training_runs WHERE id=?').get(id)),body=api.ledgerParse(row.body);
 expect(body.checkpoint).toEqual(first.descriptor);expect(body.previous_checkpoint).toEqual(second.descriptor);checks.push('actual native 16/32/48 save retains only current and previous durable generations');
},30000);
test("journal race after readback preserves the preceding durable generation and removes local staged pt", async () => {
  let count = 0;
  await expect(save(64, async () => { if (++count === 3) memory.connect((db: any) => db.query("UPDATE training_runs SET updated=updated+1 WHERE id=?").run(id)); })).rejects.toThrow("CAS");
  const row = memory.connect((db: any) => db.query("SELECT body FROM training_runs WHERE id=?").get(id));
  expect(api.ledgerParse(row.body).checkpoint).toEqual(first.descriptor); expect(sha(await readFile(first.source))).toBe(first.descriptor.sha256);
  for (const name of (await readdir(local)).filter(name => name.startsWith("checkpoint-save-"))) expect((await readdir(join(local, name))).some(file => file.endsWith(".pt"))).toBe(false);
  checks.push("post-readback CAS conflict kept prior checkpoint and no local pt duplicate");
}, 30000);
test("actual native reader closes and reclaims successive current directories; failed FD close retains the same owner for retry",async()=>{
 const runtime=JSON.parse(await readFile(join(root,'build/native-evolution/runtime-authority.json'),'utf8'));
 const source=first.source,descriptor=first.descriptor;
 for(let i=0;i<6;i++){
  const output=join(local,'current-lifecycle-'+i),cp=await api.readNativeCheckpoint({binary:join(runtime.runtimeRoot,runtime.roles.reader.path),binary_sha256:runtime.roles.reader.sha256,source,checkpoint_sha256:descriptor.sha256,output,expected:{id,binding}});
  expect(cp.files.every((f:any)=>f.handle.fd>=0)).toBe(true);expect((await readdir(local)).filter(n=>n.startsWith('current-lifecycle-'))).toEqual(['current-lifecycle-'+i]);
  if(i===2){const handle=cp.files[0].handle,close=handle.close.bind(handle);let failed=false;handle.close=async()=>{if(!failed){failed=true;throw Error('injected FD close failure');}await close();};
   await expect(cp.close()).rejects.toThrow('FD close unconfirmed');expect(handle.fd).toBeGreaterThanOrEqual(0);expect((await lstat(output)).isDirectory()).toBe(true);
  }
  await cp.close();await cp.close();expect(cp.files.every((f:any)=>f.handle.fd===-1)).toBe(true);await expect(lstat(output)).rejects.toThrow();
 }
 expect((await readdir(local)).filter(n=>n.startsWith('current-lifecycle-'))).toEqual([]);expect(await Bun.file(source).exists()).toBe(true);
 const body=api.ledgerParse(memory.connect((db:any)=>db.query('SELECT body FROM training_runs WHERE id=?').get(id)).body);
 expect(body.checkpoint).toEqual(descriptor);expect(await Bun.file(join(store,'training-runs',id,body.previous_checkpoint.name)).exists()).toBe(true);
 checks.push('6 actual native reader lifecycles leave zero closed current directories; failed FD close retained files until same-owner retry; durable current and previous untouched');
},30000);
test('compiled complete checkpoint closes borrowed original/resume FDs before reclaiming raw directory, retrying a failed borrower',async()=>{
 const runtime=JSON.parse(await readFile(join(root,'build/native-evolution/runtime-authority.json'),'utf8')),output=join(local,'current-complete-owner');
 const cp=await api.readNativeCheckpoint({binary:join(runtime.runtimeRoot,runtime.roles.reader.path),binary_sha256:runtime.roles.reader.sha256,source:first.source,checkpoint_sha256:first.descriptor.sha256,output,expected:{id,binding}});
 const borrowed=await open(cp.files[0].path,'r'),reference=await open(cp.files[0].path,'r');let fail=true,readerCloses=0;
 const original={handle:reference,async close(){if(fail){fail=false;throw Error('borrowed FD close failure');}await reference.close();}},resume={close:()=>borrowed.close()},s={closed:false};
 const loaded={async close(){expect(borrowed.fd).toBe(-1);expect(reference.fd).toBe(-1);readerCloses++;await cp.close();}};
 const bundle=await readFile(join(root,'build/native-evolution/worker.js'),'utf8'),start=bundle.indexOf('async function loadCore450CheckpointHeld('),end=bundle.indexOf('async function ',start+25),fn=bundle.slice(start,end<0?undefined:end);
 const begin=fn.indexOf('}, async close() {')+'}, async close() {'.length,last=fn.indexOf('\n    } });',begin);expect(begin).toBeGreaterThan(17);expect(last).toBeGreaterThan(begin);
 const body=fn.slice(begin,last).replace(/\bassert\d+\b/g,'assert');
 const close=new Function('s','assertObservationMutationAllowed','engine','resume','original','loaded','official','cp','assert','return async()=>{'+body+'\n};')(s,()=>{}, {},resume,original,loaded,undefined,cp,assert);
 try{await expect(close()).rejects.toThrow('Complete Core owner closure unconfirmed');expect(readerCloses).toBe(0);expect((await lstat(output)).isDirectory()).toBe(true);expect(cp.files.every((f:any)=>f.handle.fd>=0)).toBe(true);
  await close();await close();expect(readerCloses).toBe(1);expect(s.closed).toBe(true);await expect(lstat(output)).rejects.toThrow();
 }finally{await borrowed.close();await reference.close();await cp.close();}
 checks.push('actual compiled complete checkpoint refuses deletion until all borrowed FDs closed; failed borrower retry succeeds');
},30000);
