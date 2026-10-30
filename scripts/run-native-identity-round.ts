/** Local research invocation of the installed product service. No alternative trainer or public HTTP entry. */
import assert from "node:assert/strict";
import { appendFileSync, closeSync, fsyncSync, openSync } from "node:fs";
import { chmod, lstat, mkdir, mkdtemp, readFile, realpath, writeFile } from "node:fs/promises";
import { cpus, homedir, totalmem } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { spawnSync } from "node:child_process";
import { defaultHome, requireStorage } from "../backend/settings.ts";
import { NativeEvolutionService, type NativeEvolutionLease, type NativeEvolutionSession } from "../backend/native-evolution-service.ts";
import type { NativeEvolutionMaterial } from "../backend/native-evolution-cohort.ts";
import { consolidateMaterialGroups, validateNativeProductRecords } from "../backend/native-evolution-material.ts";

const AUTHORIZED = "/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/authorized-identity-material.json";
const INSTALLED_WORKER = join(homedir(), "Applications/Luban.app/Contents/Resources/native-evolution/worker.js");

export async function readIdentityMaterial(path: string): Promise<NativeEvolutionMaterial> {
  const stat = await lstat(path); assert(stat.isFile() && !stat.isSymbolicLink() && stat.size > 0 && stat.size <= (16 << 20));
  const material = JSON.parse(await readFile(path, "utf8"));
  assert.deepEqual(Object.keys(material).sort(), ["directedTeaching", "retentionRegressions", "samples"]);
  const groups = consolidateMaterialGroups(material.samples, null);
  validateNativeProductRecords({ groups, protection: [], retention_regressions: material.retentionRegressions, directed_teaching: material.directedTeaching });
  assert.deepEqual(Object.fromEntries(Object.entries(material.directedTeaching.facts).map(([key, fact]: [string, any]) => [key, fact.value])),
    { model_name: "luban0", model_fine_tuner: "麦好火", model_positioning: "无限进化" });
  assert([50,62].includes(material.samples.length) && material.retentionRegressions.length === 3, "身份材料仅限原50条或同三项事实扩写的62条，以及3条只读用户边界");
  assert(groups.new_training.length === material.samples.length-30 && groups.new_validation.length===10 && groups.replay_training.length===16 && groups.replay_validation.length===4, "身份问法扩写不得增加或挪用独立验证、回放与用户边界");
  return material;
}

/** Only the already-running same-user compute child is eligible. Environment text never leaves this function's caller. */
function computePort(launch: { pid: number; compute_pid: number }, processLine: string) {
  assert(Number.isSafeInteger(launch.pid) && launch.pid > 1 && Number.isSafeInteger(launch.compute_pid) && launch.compute_pid > 1);
  const process = /^\s*(\d+)\s+(\d+)\s+([^\n]+)\s*$/.exec(processLine);
  assert(process && Number(process[1]) === launch.pid && Number(process[2]) === globalThis.process.getuid!(), "计算进程不是当前应用的同用户子进程");
  const port = /(?:^|\s)-m\s+luban\.server\s+--port\s+(\d+)(?:\s|$)/.exec(process[3]!);
  assert(port && Number(port[1]) > 0 && Number(port[1]) <= 65535, "当前应用的私有计算端口不可用");
  return port[1]!;
}
export function parseComputeConnection(launch: { pid: number; compute_pid: number }, processLine: string, environment: string) {
  const port = computePort(launch, processLine);
  const token = /(?:^|\s)LUBAN_COMPUTE_TOKEN=([A-Za-z0-9_-]{32,})(?:\s|$)/.exec(environment)?.[1];
  assert(token, "当前应用的私有计算连接不可用");
  return { url: "http://127.0.0.1:" + port, token };
}
export async function applicationConnection(home: string) {
  const path = join(home, "launch.json"), launch = JSON.parse(await readFile(path, "utf8"));
  assert(launch.backend === "bun");
  const inspect = (args: string[]) => {
    const result = spawnSync("/bin/ps", args, { encoding: "utf8", timeout: 5000, maxBuffer: 4 << 20 });
    assert(result.status === 0 && !result.error, "无法核对当前应用计算进程"); return result.stdout;
  };
  const line = inspect(["-ww", "-p", String(launch.compute_pid), "-o", "ppid=,uid=,command="]);
  computePort(launch, line); // Verify same-user parentage before reading any process environment.
  const environment = inspect(["eww", "-p", String(launch.compute_pid), "-o", "command="]);
  const connection = parseComputeConnection(launch, line, environment);
  const current = JSON.parse(await readFile(path, "utf8"));
  assert(current.pid === launch.pid && current.compute_pid === launch.compute_pid && current.port === launch.port, "应用已重启，请重新运行入口");
  return connection;
}

type Worker = { openNativeEvolutionRuntime(input: Record<string, unknown>): Promise<NativeEvolutionSession> };
export async function runIdentityRound(input: { home: string; output: string; material: NativeEvolutionMaterial; worker: Worker;
  compute(path: string, body: Record<string, unknown>): Promise<Response>; resume?: string; pauseAtStep?: number; signal?: AbortSignal; redact?(text: string): string }) {
  assert(input.pauseAtStep === undefined || Number.isSafeInteger(input.pauseAtStep) && input.pauseAtStep > 0 && input.pauseAtStep < 450);
  const began = performance.now(), startedAt = new Date().toISOString();
  await mkdir(input.output, { mode: 0o700 });
  const progressFD = openSync(join(input.output, "progress.jsonl"), "wx", 0o600);
  const redact = input.redact ?? ((value: string) => value);
  let recordingFailure: unknown;
  const event = (value: Record<string, unknown>) => {
    const row = { seconds: (performance.now() - began) / 1000, ...value };
    const text = redact(JSON.stringify(row));
    try { appendFileSync(progressFD, text + "\n"); } catch (error) { recordingFailure ??= error; }
    console.log(text);
  };
  let work: string | undefined, leaseScope: Omit<NativeEvolutionLease, "lease_id"> | undefined;
  const service = new NativeEvolutionService({ compute: input.compute,
    async open(lease, signal, continuation) {
      signal.throwIfAborted(); const { lease_id: _privateLease, ...scope } = lease; leaseScope = scope;
      requireStorage(lease.root);
      const parent = join(input.home, "native-evolution-work"); await mkdir(parent, { recursive: true, mode: 0o700 }); await chmod(parent, 0o700);
      work = await mkdtemp(join(await realpath(parent), "identity-"));
      event({ step: 0, phase: "opening_native_model" });
      const session = await input.worker.openNativeEvolutionRuntime({ home: input.home, model: lease.model, store: lease.store,
        root: lease.root, identity: lease.identity, work, material: continuation ?? input.material });
      return { continuationMaterial: () => session.continuationMaterial?.() ?? null,
        async runRound(options) {
          try { return await session.runRound({ ...options, onProgress(progress) {
            event(progress); options.onProgress(progress);
            if (progress.step === input.pauseAtStep) service.cancel();
          } }); }
          catch (error) { event({ phase: "native_round_error", error: error instanceof Error ? error.stack ?? error.message : String(error) }); throw error; }
        },
        async close() { await session.close(); event({ phase: "native_owners_closed" }); } };
    } });
  const cancel = () => service.cancel(); input.signal?.addEventListener("abort", cancel, { once: true });
  try {
    input.signal?.throwIfAborted(); event({ step: 0, phase: "acquiring_application_lease", fixed_steps: 450 });
    service.start({ continuous: false, ...(input.resume ? { resume: input.resume } : {}) });
    let status: Record<string, any> & { running: boolean; lease_held: boolean } = await service.wait();
    // Do not exit or release the real application lease while a native close/rollback remains incomplete.
    while (status.state === "quarantined") {
      event({ phase: "retrying_same_native_cleanup", error: status.cleanup_error }); await Bun.sleep(1000);
      try { await service.pause(); } catch {} status = service.status();
    }
    const result = { schema: "luban-native-identity-round-v1", started_at: startedAt, total_seconds: (performance.now() - began) / 1000,
      total_scope: "application lease, native model preparation, remaining updates up to original450, BF16/Q4 validation, save/adoption, cleanup and reload request",
      fixed_steps: 450, resume: input.resume ?? null, requested_pause_at_step: input.pauseAtStep ?? null,
      precision: { training: "torch.bfloat16", application: "mixed-q4-g16/full24/resident" },
      hardware: { platform: process.platform, arch: process.arch, cpu: cpus()[0]?.model, system_memory_bytes: totalmem() },
      memory_measurement: "Whole-model peak memory is not sampled here; system_memory_bytes is installed RAM, not model resident memory.",
      material_counts: { samples: input.material.samples.length, readonly_user_regressions: input.material.retentionRegressions?.length ?? 0 },
      recording_error: recordingFailure ? redact(String(recordingFailure)) : null, work, scope: leaseScope, status };
    await writeFile(join(input.output, "result.json"), redact(JSON.stringify(result, null, 2)) + "\n", { flag: "wx", mode: 0o600 });
    event({ phase: "finished", state: status.state, step: status.step, adopted: status.last_result?.adopted ?? false, total_seconds: result.total_seconds });
    return result;
  } finally { input.signal?.removeEventListener("abort", cancel); fsyncSync(progressFD); closeSync(progressFD); }
}

if (import.meta.main) {
  const { values } = parseArgs({ args: Bun.argv.slice(2), strict: true, options: {
    home: { type: "string", default: defaultHome() }, material: { type: "string", default: AUTHORIZED },
    worker: { type: "string", default: INSTALLED_WORKER }, output: { type: "string" }, resume: { type: "string" }, "pause-at-step": { type: "string" },
    "check-only": { type: "boolean", default: false },
  } });
  assert(!values.resume || /^[a-f0-9]{12}$/.test(values.resume), "需要原生journal的12位恢复ID");
  const pauseAtStep = values["pause-at-step"] === undefined ? undefined : Number(values["pause-at-step"]);
  assert(pauseAtStep === undefined || Number.isSafeInteger(pauseAtStep) && pauseAtStep > 0 && pauseAtStep < 450, "暂停点须为1至449的已保存进度");
  const material = await readIdentityMaterial(resolve(values.material!));
  if (values["check-only"]) console.log(JSON.stringify({ valid_material: true, samples: material.samples.length,
    readonly_user_regressions: material.retentionRegressions!.length, fixed_steps: 450, model_loaded: false, trained: false }));
  else {
    assert(values.output, "使用 --output 指定新的研究输出目录");
    const connection = await applicationConnection(resolve(values.home!));
    const worker = await import(pathToFileURL(resolve(values.worker!)).href); assert(typeof worker.openNativeEvolutionRuntime === "function");
    const controller = new AbortController(), stop = () => controller.abort(); process.on("SIGINT", stop); process.on("SIGTERM", stop);
    try {
      const result = await runIdentityRound({ home: resolve(values.home!), output: resolve(values.output), material, worker, resume: values.resume, pauseAtStep,
        signal: controller.signal, redact: value => value.replaceAll(connection.token, "[private-compute-token]"),
        compute: (path, body) => fetch(connection.url + path, { method: "POST", headers: { authorization: "Bearer " + connection.token,
          "content-type": "application/json" }, body: JSON.stringify(body), redirect: "manual", proxy: null } as RequestInit & { proxy: null }) });
      if (result.status.state !== "complete" && result.status.state !== "waiting_material") process.exitCode = 1;
    } finally { process.off("SIGINT", stop); process.off("SIGTERM", stop); }
  }
}
