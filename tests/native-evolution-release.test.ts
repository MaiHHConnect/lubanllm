import { expect, test } from "bun:test";
import { mkdtemp, open, realpath, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { borrowNativeEvolutionRelease, closeNativeEvolutionRelease, withNativeEvolutionRelease, retainNativeReleaseFailure } from "../backend/native-evolution-release.ts";
import { unchangedNativeFile } from "../.native-migration/research/native-product-runtime/backend/native-features.ts";

async function fixture() {
  const root = await realpath(await mkdtemp(join(tmpdir(), "native-release-"))), path = join(root, "program");
  await writeFile(path, "already admitted program"); const engine = {}, authority = { manifest_sha256: "a".repeat(64) };
  let admissions = 0, closes = 0, failClose = false;
  const handles: Awaited<ReturnType<typeof open>>[] = [];
  const verify = async () => {
    admissions++; const handle = await open(path, "r"), stat = await handle.stat({ bigint: true }); handles.push(handle);
    return { manifest: { role: "trainer" }, files: [{ path }], async assertUnchanged() { await unchangedNativeFile(path, handle, stat); },
      async close() { closes++; if (failClose) { failClose = false; throw Error("close injected"); } if (handle.fd !== -1) await handle.close(); } };
  };
  return { root, path, engine, authority, handles, verify, admissions: () => admissions, closes: () => closes,
    failClose() { failClose = true; }, borrow(signal?: AbortSignal) { return withNativeEvolutionRelease(engine, () => borrowNativeEvolutionRelease(root, authority, verify, signal)); },
    async cleanup() { await closeNativeEvolutionRelease(engine); for (const handle of handles) if (handle.fd !== -1) await handle.close(); await rm(root, { recursive: true }); } };
}

test("same live Engine verifies once across sequential/concurrent program borrowers and releases only at final ownership close", async () => {
  const r = await fixture();
  try {
    const [first, second] = await Promise.all([r.borrow(), r.borrow()]);
    expect(r.admissions()).toBe(1); await first.assertUnchanged(); await first.close();
    await expect(closeNativeEvolutionRelease(r.engine)).rejects.toThrow(/仍持有/);
    expect(r.handles[0]!.fd).not.toBe(-1); await second.close(); await closeNativeEvolutionRelease(r.engine);
    expect(r.handles[0]!.fd).toBe(-1); expect(r.closes()).toBe(1);
    expect(() => r.borrow()).toThrow(/关闭/);
  } finally { await r.cleanup(); }
});

test("sequential segments reuse held FDs while a changed path stat rejects without another full verification", async () => {
  const r = await fixture();
  try {
    const first = await r.borrow(); await first.close(); const next = await r.borrow(); await next.close();
    expect(r.admissions()).toBe(1); expect(r.closes()).toBe(0);
    await writeFile(r.path, "mutated source"); await expect(r.borrow()).rejects.toThrow(/变化/);
    expect(r.admissions()).toBe(1);
  } finally { await r.cleanup(); }
});

test("source switch and new Engine bind separately; final FD-close failure retries the same owner", async () => {
  const r = await fixture(), other = {};
  try {
    const first = await r.borrow(); await first.close();
    const different = await withNativeEvolutionRelease(r.engine, () => borrowNativeEvolutionRelease(r.root, { manifest_sha256: "b".repeat(64) }, r.verify)); await different.close();
    const next = await withNativeEvolutionRelease(other, () => borrowNativeEvolutionRelease(r.root, r.authority, r.verify)); await next.close();
    expect(r.admissions()).toBe(3); r.failClose();
    await expect(closeNativeEvolutionRelease(r.engine)).rejects.toThrow(/保留/);
    expect(r.handles[0]!.fd).not.toBe(-1); expect(r.handles[1]!.fd).toBe(-1);
    await closeNativeEvolutionRelease(r.engine); expect(r.handles[0]!.fd).toBe(-1);
    expect(r.admissions()).toBe(3);
  } finally { await closeNativeEvolutionRelease(other); await r.cleanup(); }
});

test("cancellation adds no borrower and failed initial verification cleanup remains retryable", async () => {
  const r = await fixture();
  try {
    const first = await r.borrow(); await first.close();
    await expect(r.borrow(AbortSignal.abort())).rejects.toThrow(); expect(r.admissions()).toBe(1);
    let attempts = 0;
    await expect(withNativeEvolutionRelease(r.engine, () => borrowNativeEvolutionRelease(r.root, { failed: true }, async () => {
      retainNativeReleaseFailure(async () => { if (++attempts === 1) throw Error("retained close failure"); }); throw Error("admission cancelled");
    }))).rejects.toThrow(/cancelled/);
    await expect(closeNativeEvolutionRelease(r.engine)).rejects.toThrow(/保留/);
    await closeNativeEvolutionRelease(r.engine); expect(attempts).toBe(2); expect(r.handles[0]!.fd).toBe(-1);
  } finally { await r.cleanup(); }
});

test("actual composed v6 admission reads and inspects the capsule once, then only checks its original held files", async () => {
  const { spawnSync } = await import("node:child_process");
  const child = spawnSync(process.execPath, ["--eval", String.raw`
import assert from 'node:assert/strict';import {mock} from 'bun:test';import{readFile,writeFile,mkdir,mkdtemp,realpath,rm}from'node:fs/promises';import{resolve,join,dirname,basename}from'node:path';import{tmpdir}from'node:os';
import{applyNativeEvolutionReleasePatches}from'./scripts/native-evolution-release-patches.ts';
import{withNativeEvolutionRelease,closeNativeEvolutionRelease}from'./backend/native-evolution-release.ts';
const project=process.cwd(),source=resolve('.native-migration/research/native-unified-service/admission-v6.ts'),engine=resolve('.native-migration/research/native-unified-expert/native-engine.ts');
globalThis.__releaseInspections=0;globalThis.__releaseReads=0;globalThis.__releaseHashes=0;let machoChecks=0;const image=p=>({id:p.endsWith('.dylib')?'@rpath/'+basename(p):null,rpaths:[],dependencies:[]});

mock.module(resolve('backend/native-runtime-macho.ts'),()=>({parseImage:p=>image(p),dependencyClosure:async()=>({}),system:()=>true}));
mock.module(resolve('scripts/release-macho.ts'),()=>({checkReleaseMachO:()=>{machoChecks++;}}));
const captured=resolve('.native-migration/research/native-observation-python-parity-v3/sources/admission.source');
const bodies=new Map([[source,await readFile(captured,'utf8')],[engine,await readFile(engine,'utf8')]]);
applyNativeEvolutionReleasePatches({bodies,target:p=>p,root:project,replace:(s,a,b)=>{assert.equal(s.split(a).length,2,a);return s.replace(a,b);}});
const root=await realpath(await mkdtemp(join(tmpdir(),'actual-release-'))),owner={};
try{
 let changed=bodies.get(source).replace(/from\s*(['"])(\.[^'"]+)\1/g,(_match,quote,path)=>'from '+JSON.stringify(resolve(dirname(source),path)));
 changed=changed.replace("=>createHash('sha256').update(raw).digest('hex');","=>{globalThis.__releaseHashes++;return createHash('sha256').update(raw).digest('hex');};").replaceAll('const raw=await readNativeSmallFile','globalThis.__releaseReads++;const raw=await readNativeSmallFile');
 const inspector=join(root,'inspect.ts');await writeFile(inspector,"export function spawnSync(exe,args){globalThis.__releaseInspections++;return{pid:123,status:0,signal:null,stdout:args.at(-1),stderr:''};}");changed=changed.replace("from 'node:child_process'",'from '+JSON.stringify(inspector));
 const modulePath=join(root,'admission.ts');await writeFile(modulePath,changed);const{verifyRelease,sha,LIBRARIES}=await import(modulePath);
 const binding=raw=>({sha256:sha(raw),bytes:Buffer.byteLength(raw)}),proof={kind:'trainer',compile:{exit_code:0,argv:['compiler','source']},inverse_verified:true,original_binary_sha256:'a'.repeat(64),approved_binary_sha256:'b'.repeat(64)};
 const manifest={schema:'luban-native-unified-service-role-capsule-v6',math_changed:false,product_admission:false,quality_admission:false,programs:{trainer:{path:'trainer',proof,closure:[]}},libraries:{},payload:{},licenses:{},assets:{}};
 await mkdir(join(root,'lib'));const paths=['trainer',...LIBRARIES.map(n=>'lib/'+n)];
 for(const path of paths){const raw='fixture '+path,final=binding(raw),original={sha256:proof.approved_binary_sha256,bytes:raw.length};await writeFile(join(root,path),raw);manifest.payload[path]={original,copied:original,stripped:original,relocated:original,final,commands:[],image:image(path)};if(path.startsWith('lib/'))manifest.libraries[path]=final;}
 const raw=JSON.stringify(manifest);await writeFile(join(root,'manifest.json'),raw);
 const authority={manifest_sha256:sha(raw),sources:{trainer:sha(JSON.stringify(proof))},paths:{trainer:'trainer'},manifest_relative:'manifest.json',library_directory:'lib',licenses:[],rpaths:Object.fromEntries(paths.map(p=>[p,[]])),assets:{}};
 const first=await withNativeEvolutionRelease(owner,()=>verifyRelease(root,authority));const initial=globalThis.__releaseInspections,reads=globalThis.__releaseReads,hashes=globalThis.__releaseHashes;assert.equal(reads,6);assert.equal(initial,paths.length*3);assert.equal(machoChecks,paths.length);
 await first.assertUnchanged();await first.close();
 const second=await withNativeEvolutionRelease(owner,()=>verifyRelease(root,authority));await second.assertUnchanged();await second.verifyComplete();await second.assertFilesUnchanged();await second.close();await assert.rejects(second.assertFilesUnchanged());await assert.rejects(second.verifyComplete());
 assert.equal(globalThis.__releaseInspections,initial);assert.equal(machoChecks,paths.length);assert.equal(globalThis.__releaseReads,reads);assert.equal(globalThis.__releaseHashes,hashes);
 await writeFile(join(root,'trainer'),'changed source');await assert.rejects(withNativeEvolutionRelease(owner,()=>verifyRelease(root,authority)));assert.equal(globalThis.__releaseInspections,initial);
 await closeNativeEvolutionRelease(owner);console.log(JSON.stringify({passed:true,inspections:globalThis.__releaseInspections,programs:paths.length}));
}finally{await closeNativeEvolutionRelease(owner);await rm(root,{recursive:true,force:true});}
`], { cwd: process.cwd(), encoding: "utf8", timeout: 30_000 });
  expect(child.status, child.stderr).toBe(0);
  expect(JSON.parse(child.stdout)).toEqual({ passed: true, inspections: 15, programs: 5 });
});

test("cancellation while initial admission is in flight retains its FD until lifecycle cleanup", async () => {
  const r = await fixture(), controller = new AbortController();
  let signalStarted!: () => void, release!: () => void;
  const started = new Promise<void>(resolve => { signalStarted = resolve; }), gate = new Promise<void>(resolve => { release = resolve; });
  try {
    const pending = withNativeEvolutionRelease(r.engine, () => borrowNativeEvolutionRelease(r.root, r.authority, async () => {
      const binding = await r.verify(); signalStarted(); await gate; return binding;
    }, controller.signal));
    await started; controller.abort(); release();
    await expect(pending).rejects.toThrow(); expect(r.handles[0]!.fd).not.toBe(-1);
    await closeNativeEvolutionRelease(r.engine); expect(r.handles[0]!.fd).toBe(-1); expect(r.closes()).toBe(1);
  } finally { release(); await r.cleanup(); }
});
