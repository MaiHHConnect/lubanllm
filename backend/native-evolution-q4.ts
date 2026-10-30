/** Serial ordinary-Q4 observations through the shared native tokenizer/Metal worker. */
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, writeFile, mkdir, readFile, lstat, unlink } from "node:fs/promises";
import { join, isAbsolute } from "node:path";
import { q4ConstraintFrame } from "./native-evolution-constraint-inputs.ts";
import { cleanRuntimeEnvironment } from "./native-process.ts";
import { materialMessages, materialProvenance } from "./native-evolution-material.ts";
import { evaluateGeneration, validateGenerationCheck } from "./generation-checks.ts";

type Row = Record<string, any>;
type Core = { file: string; tensor_key: string; sha256: string };
export type NativeQ4Expert = { file: string; sha256: string; width: number; layer: 23; core_version: string };
type Request = { executable: string; binary: string; model: string; manifestSHA: string; identity: string;
  core?: Core; expert?: NativeQ4Expert; records: Row[]; work: string; signal?: AbortSignal };
export type NativeQ4Observation = Readonly<{ precision: "mixed-q4-g16"; identity: string;
  manifest_sha256: string; core: Readonly<Core> | null; expert: Readonly<NativeQ4Expert & {core_file_sha256:string}> | null; worker_binary_sha256: string; seed: 42; temperature: 0;
  rows: ReadonlyArray<Readonly<{ id: string; answer: string; passed: boolean; complete: boolean;
    branch_enabled: boolean; reference_used: false; tokens: readonly number[]; input_tokens: readonly number[]; finish_reason: string }>> }>;
const observations = new WeakMap<object, string>();
const calibrations = new WeakMap<object, Buffer>();
export function nativeQ4CalibrationInputs(value:Awaited<ReturnType<NativeQ4Observer["captureTraining"]>>,records:Row[]){assertNativeQ4Observation(value.baseline,records);const raw=calibrations.get(value);assert(raw,"Q4 calibration must come from this actual capture");return Buffer.from(raw);}
const tokenIDs = (v: unknown): v is number[] => Array.isArray(v) && v.every(n => Number.isSafeInteger(n) && n >= 0 && n < 248320);
const digest = (v: string) => /^[a-f0-9]{64}$/.test(v);

/** A caller cannot manufacture an observation by setting actual/passed booleans. */
export function assertNativeQ4Observation(value: NativeQ4Observation, records: Row[], expert?: NativeQ4Expert) {
  assert.equal(observations.get(value), JSON.stringify(records), "Q4证据必须来自本进程本次完整原生观察且材料一致");
  assert.deepEqual(value.expert, expert ? {...expert,core_file_sha256:value.core?.sha256} : null, "Q4专家证据范围不一致");
}

export class NativeQ4Observer {
 private child?: ChildProcess;
 private exited?: Promise<void>;
 async close() {
  if (!this.child) return;
  if (this.child.exitCode === null && this.child.signalCode === null) { this.child.stdin!.end(); this.child.kill("SIGTERM"); }
  let timer: ReturnType<typeof setTimeout>;
  try {
    await Promise.race([this.exited!, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("Q4原生进程未完成关闭，必须保留模型互斥")), 10_000); })]);
    this.child = undefined; this.exited = undefined;
  } finally { clearTimeout(timer!); }
 }
 async observe(request: Request): Promise<NativeQ4Observation> { return (await this.run(request, 1))[0]!; }
 async observePair(request: Request): Promise<readonly [NativeQ4Observation, NativeQ4Observation]> {
  const rows = await this.run(request, 2); return Object.freeze([rows[0]!, rows[1]!] as const);
 }
 async captureTraining(request: Request) {
  assert(!request.expert, "Q4 core calibration cannot include an expert");
  assert(request.records.every(row => row.approved === true && row.training_allowed === true && row.purpose === 'training' && materialProvenance(row).evolution_role === 'replay'), 'Q4 constraints require authorized replay TRAIN');
  const capture: {parts: Buffer[]; rows: {id:string;positions:number}[]} = {parts:[],rows:[]};
  const baseline = (await this.run(request, 1, capture))[0]!;
  assert(capture.parts.length > 0, 'No correct Q4 TRAIN predictors');
  const value=Object.freeze({baseline,rows:Object.freeze(capture.rows.map(row=>Object.freeze(row)))});calibrations.set(value,Buffer.concat(capture.parts));return value;
 }
 private async run(request: Request, repetitions: 1 | 2, capture?: {parts:Buffer[];rows:{id:string;positions:number}[]}): Promise<NativeQ4Observation[]> {
  assert(!this.child, "Q4上次观察尚未关闭");
  request = {...request, core: request.core ? Object.freeze({...request.core}) : undefined,
    expert: request.expert ? Object.freeze({...request.expert}) : undefined};
  for (const path of [request.executable, request.binary, request.model, request.work]) assert(isAbsolute(path) && !path.includes("\0"));
  assert(digest(request.identity) && digest(request.manifestSHA));
  assert(Array.isArray(request.records) && request.records.length && new Set(request.records.map(r => r.id)).size === request.records.length);
  if (request.core) assert(isAbsolute(request.core.file) && digest(request.core.sha256) && request.core.tensor_key);
  if (request.expert) {
    const e=request.expert;
    assert(!capture && request.core && isAbsolute(e.file) && !e.file.includes("\0") && digest(e.sha256)
      && Number.isSafeInteger(e.width) && e.width>=32 && e.width<=512 && e.layer===23
      && typeof e.core_version==='string' && e.core_version.length>0 && e.core_version.length<=128, "Invalid Q4 expert binding");
  }
  // Freeze the actual inputs once; caller mutation cannot alter the admitted material later.
  const recordsJSON = JSON.stringify(request.records), records: Row[] = JSON.parse(recordsJSON);
  const samples = records.map(row => ({ id: row.id, messages: materialMessages(row),
    check: validateGenerationCheck(materialProvenance(row).generation_check, row.answer) }));
  request.signal?.throwIfAborted();
  const work = await mkdtemp(join(request.work, "q4-observe-")), config = join(work, "config.json");
  const traceDirectory = capture ? join(work, "trace") : undefined; if(traceDirectory) await mkdir(traceDirectory,{mode:0o700});
  await writeFile(config, JSON.stringify({ ...(traceDirectory ? {trace_directory:traceDirectory}:{}), binary: request.binary, model: request.model, manifest_sha256: request.manifestSHA,
    identity: request.identity, log: join(work, "worker.log"), ...(request.core ? { core: request.core } : {}),
    ...(request.expert ? {expert:{...request.expert,core_file_sha256:request.core!.sha256}} : {}) }), { mode: 0o600, flag: "wx" });
  const child = spawn(request.executable, [config], { stdio: ["pipe", "pipe", "pipe"], env: cleanRuntimeEnvironment() });
  this.child = child;
  let pending = "", stderr = "", failure: Error | undefined, ended = false;
  const queue: Row[] = []; let waiter: { resolve: (value: Row) => void; reject: (error: Error) => void } | undefined;
  const fail = (error: Error) => { failure ??= error; waiter?.reject(failure); waiter = undefined; };
  child.on("error", fail); child.stdin.on("error", fail);
  child.stderr.on("data", (data: Buffer) => { stderr = (stderr + data.toString()).slice(-8192); });
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (data: string) => {
    try {
      pending += data; assert(Buffer.byteLength(pending) <= 4 << 20, "Q4事件超过上限");
      let at: number;
      while ((at = pending.indexOf("\n")) >= 0) {
        const event = JSON.parse(pending.slice(0, at)); pending = pending.slice(at + 1);
        assert(event && typeof event === "object" && typeof event.event === "string", "Q4事件格式无效");
        if (waiter) { const current = waiter; waiter = undefined; current.resolve(event); }
        else { assert(queue.length < (capture ? 260 : 4), "Q4出现未请求事件"); queue.push(event); }
      }
    } catch (error) { fail(error as Error); }
  });
  const exited = new Promise<void>(resolve => child.once("close", (code, signal) => {
    ended = true; if (code !== 0 || signal || pending) fail(new Error(`Q4观察退出失败：${signal ?? code} ${stderr}`));
    else if (waiter) fail(new Error("Q4观察提前退出")); resolve();
  }));
  this.exited = exited;
  async function next(): Promise<Row> {
    if (failure) throw failure;
    if (queue.length) return queue.shift()!;
    assert(!ended && !waiter, "Q4观察已结束或请求并发");
    let timer: ReturnType<typeof setTimeout>;
    try { return await new Promise<Row>((resolve, reject) => {
      waiter = { resolve, reject }; timer = setTimeout(() => fail(new Error("Q4观察超时")), 180_000);
    }); } finally { clearTimeout(timer!); }
  }
  // Before ready, the Swift signal handler may not exist yet. Remember the
  // cancellation and close at the first ready boundary instead of killing startup.
  let readyForCancel = false;
  const abort = () => { if (!ended && readyForCancel) child.kill("SIGUSR1"); };
  request.signal?.addEventListener("abort", abort, { once: true });
  let completed = false, observationFailure: unknown;
  try {
    const ready = await next(); readyForCancel = true; request.signal?.throwIfAborted();
    assert(ready.event === "ready" && ready.pid === child.pid && Number.isSafeInteger(ready.worker_pid) && ready.worker_pid > 0 && ready.eos === 248044);
    const loaded = ready.loaded;
    assert(loaded?.manifest_sha256 === request.manifestSHA && loaded.base_identity === request.identity
      && loaded.inference_precision === "mixed-q4-g16" && loaded.core_loaded === !!request.core
      && (loaded.execution_skip_mask ?? 0) === 0 && (loaded.execution_exit_after ?? 24) === 24
      && !(loaded.layer_paging_enabled ?? false), "Q4观察实际加载范围不一致");
    if (request.core) assert.equal(loaded.core_file_sha256, request.core.sha256);
    assert(typeof loaded.worker_binary_sha256==='string' && digest(loaded.worker_binary_sha256), "Q4 binary receipt missing");
    if (request.expert) assert(loaded.expert_loaded===true && loaded.expert_file_sha256===request.expert.sha256
      && loaded.expert_width===request.expert.width && loaded.expert_layer===23, "Q4 expert actual load differs");
    else assert(loaded.expert_loaded!==true, "Unexpected Q4 expert");
    const batches: any[][] = [];
    for (let repeat = 0; repeat < repetitions; repeat++) {
     const rows: any[] = [];
     for (const sample of samples) {
      request.signal?.throwIfAborted();
      child.stdin.write(JSON.stringify({ id: sample.id, messages: sample.messages, max_tokens: Number(sample.check.max_new_tokens) }) + "\n");
      const event = await next(); request.signal?.throwIfAborted();
      assert(event.event === "observation" && event.id === sample.id && event.worker_pid === ready.worker_pid
        && tokenIDs(event.input_tokens) && event.input_tokens.length && tokenIDs(event.tokens)
        && event.tokens.length <= Number(sample.check.max_new_tokens) && typeof event.answer === "string"
        && event.seed === 42 && event.temperature === 0 && event.thinking === "off"
        && event.branch_enabled === !!request.expert && event.reference_used === false
        && ["stop", "length"].includes(event.finish_reason), "Q4实际逐项输出不完整");
      const complete = event.finish_reason === "stop" && event.tokens.at(-1) === ready.eos;
      assert(event.finish_reason !== "stop" || complete, "Q4停止但缺少EOS");
      rows.push(Object.freeze({ id: sample.id, answer: event.answer, complete,
        passed: evaluateGeneration(event.answer, sample.check, complete).passed,
        branch_enabled: !!request.expert, reference_used: false, tokens: Object.freeze(event.tokens),
        input_tokens: Object.freeze(event.input_tokens), finish_reason: event.finish_reason }));
      if(capture && rows.at(-1)!.passed && complete){
       assert(event.tokens.length<=256 && (capture.parts.length+event.tokens.length)*6144<=10000000, 'Complete Q4 constraint exceeds native tail budget; no truncation');
       child.stdin.write(JSON.stringify({operation:'trace',id:sample.id,input_tokens:event.input_tokens,tokens:event.tokens})+'\n');
       for(let step=0;step<event.tokens.length;step++){
        const frame=await next();request.signal?.throwIfAborted();assert(frame.event==='trace'&&/^\d+$/.test(frame.id));
        for(const kind of ['tail','logits']){const path=join(traceDirectory!,`${frame.id}-step${String(step).padStart(4,'0')}.${kind}.bf16`);assert.equal(frame[kind+'_path'],path);const st=await lstat(path);assert(st.isFile()&&!st.isSymbolicLink());}
        capture.parts.push(q4ConstraintFrame(frame,{id:sample.id,input_tokens:event.input_tokens,tokens:event.tokens},step,await readFile(frame.tail_path),await readFile(frame.logits_path)));
        await unlink(frame.tail_path);await unlink(frame.logits_path);
       }
       const done=await next();assert(done.event==='trace_complete'&&done.id===sample.id&&done.worker_pid===ready.worker_pid&&done.positions===event.tokens.length);
       capture.rows.push({id:sample.id,positions:event.tokens.length});
      }
     }
     batches.push(rows);
    }
    child.stdin.end(); const closed = await next();
    assert(closed.event === "closed" && closed.pid === child.pid); await exited;
    if (failure) throw failure; assert.equal(queue.length, 0); request.signal?.throwIfAborted();
    const result = batches.map(rows => Object.freeze({ precision: "mixed-q4-g16" as const, identity: request.identity,
      manifest_sha256: request.manifestSHA, core: request.core ? Object.freeze({ ...request.core }) : null,
      expert: request.expert ? Object.freeze({...request.expert,core_file_sha256:request.core!.sha256}) : null,
      worker_binary_sha256: loaded.worker_binary_sha256 as string,
      seed: 42 as const, temperature: 0 as const, rows: Object.freeze(rows) }));
    for (const proof of result) observations.set(proof, recordsJSON);
    completed = true; return result;
  } catch (error) {
    observationFailure = error; throw error;
  } finally {
    request.signal?.removeEventListener("abort", abort);
    if (!completed && !ended) { child.stdin.end(); child.kill("SIGTERM"); }
    // Swift closes/reaps the Metal worker before it exits; never release the app lease early.
    try { await this.close(); }
    catch (error) { throw observationFailure === undefined ? error : new AggregateError([observationFailure, error], "Q4观察与进程关闭均失败", { cause: observationFailure }); }
  }
 }
}
