import { expect, test } from "bun:test";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { Settings } from "../backend/settings";
import { Memory, ledgerStringify } from "../backend/memory";
import { NativeEvolutionService } from "../backend/native-evolution-service";
import { NativeEvolutionAutomatic, nativeAutomaticSignature, nativeEvolutionAttempted, NATIVE_AUTOMATIC_ARGUMENTS } from "../backend/native-evolution-automatic";

function row(id: string, purpose = "training", role = "new") {
  return { id, prompt: id, answer: "ok", purpose, source: "human", approved: true, training_allowed: true,
    provenance: { evolution_role: role, split_family: id, verified_by: "local requester", generation_check: { type: "exact", expected: "ok" } } };
}
const experience = Array.from({ length: 350 }, (_, i) => row("new-" + i, i < 300 ? "training" : "validation"));
const records = { groups: { new_training: experience.slice(0, 57), new_validation: experience.slice(300), replay_training: [row("replay-train", "training", "replay")], replay_validation: [row("replay-valid", "validation", "replay")] }, protection: [], retention_regressions: [], experience };
function fixture() {
  const root = mkdtempSync(join(tmpdir(), "native-evolution-schedule-")), settings = new Settings(join(root, "home"));
  settings.update({ storage_root: root, automatic: { mode: "auto", scheduled_time: "02:00" } });
  const memory = new Memory(root), scope = { root, store: join(root, "modules/luban0-v1"), identity: "a".repeat(64) };
  const starts: any[] = []; let ready = true, resumeSupported = true;
  const service = new NativeEvolutionService({ async compute(path, body) {
    if (path.endsWith("acquire")) return Response.json({ ...scope, lease_id: body.lease_id, model: "/fixture/model", was_ready: false });
    return Response.json({ released: true, was_ready: false });
  }, async open() { return { async runRound(input) { starts.push(input); return { step: 450, phase: "rejected", quality_checked: true, adopted: false, pending: true }; }, async close() {} }; } });
  const create = () => new NativeEvolutionAutomatic({ settings, service, memory: () => memory,
    async inspect() { return { records, core: null, unspent_count: 350, execution_ready: ready, resume_supported: resumeSupported }; } });
  const scheduler = create();
  const addRun = (run: any) => memory.connect(db => {
    db.exec("CREATE TABLE IF NOT EXISTS training_runs (id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)");
    db.query("INSERT INTO training_runs VALUES (?,?,?,?,?,?)").run(run.id, scope.identity, run.kind, run.state, ledgerStringify({ ...run, binding: { identity: scope.identity } }), 1);
  });
  return { root, settings, memory, scope, starts, service, scheduler, create, addRun, setReady(value: boolean) { ready = value; }, setResume(value: boolean) { resumeSupported = value; },
    async close() { await service.close(); memory.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("native signature and rejected-next-day decisions match original Python policy", () => {
  const signature = nativeAutomaticSignature("a".repeat(64), null, records);
  const runs = [{ kind: "evolution", scope: "product", state: "rejected", automatic_signature: signature,
    arguments: NATIVE_AUTOMATIC_ARGUMENTS, started: new Date(2026, 9, 17, 2).getTime() / 1000, scheduled_occurrence: "schedule:2026-10-17" }];
  const python = spawnSync(resolve(".venv/bin/python"), ["-c", `import json,sys
from datetime import datetime
from types import SimpleNamespace
from luban.automatic import AutomaticLearning,digest
v=json.load(sys.stdin)
s=digest(dict(kind='evolution',identity=v['identity'],core=None,expert=None,arguments=v['arguments'],records=v['records']))
print(json.dumps(dict(signature=s,same=AutomaticLearning._attempted(None,s,v['runs'],datetime(2026,10,17,23)),next=AutomaticLearning._attempted(None,s,v['runs'],datetime(2026,10,18,2)))))`], {
    env: { ...process.env, PYTHONPATH: "src" }, input: JSON.stringify({ identity: "a".repeat(64), records, arguments: NATIVE_AUTOMATIC_ARGUMENTS, runs }), encoding: "utf8", timeout: 30000,
  });
  expect(python.status, python.stderr).toBe(0); const expected = JSON.parse(python.stdout);
  expect(signature).toBe(expected.signature);
  expect(nativeEvolutionAttempted(signature, runs, new Date(2026, 9, 17, 23))).toBe(expected.same);
  expect(nativeEvolutionAttempted(signature, runs, new Date(2026, 9, 18, 2))).toBe(expected.next);
});

test("one scheduled day starts one fixed round and a rejected pending pool can return the next day", async () => {
  const f = fixture();
  try {
    await f.scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 1, 59)); expect(f.starts).toHaveLength(0);
    const first = await f.scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 2)); expect(first.job_id).toBeTruthy(); await f.service.wait();
    expect(f.starts).toHaveLength(1); expect(f.starts[0].automatic.scheduled_occurrence).toBe("schedule:2026-10-17");
    f.addRun({ id: "abcdef123456", kind: "evolution", state: "rejected", automatic_signature: f.starts[0].automatic.signature,
      scheduled_occurrence: "schedule:2026-10-17", started: new Date(2026, 9, 17, 2).getTime() / 1000, arguments: NATIVE_AUTOMATIC_ARGUMENTS });
    await f.create().request("tick", {}, f.scope, new Date(2026, 9, 17, 23)); expect(f.starts).toHaveLength(1);
    await f.scheduler.request("tick", {}, f.scope, new Date(2026, 9, 18, 2)); await f.service.wait(); expect(f.starts).toHaveLength(2);
  } finally { await f.close(); }
});

test("manual reminder and deferred choice survive restart without scheduling a model", async () => {
  const f = fixture();
  try {
    f.settings.update({ automatic: { mode: "manual", scheduled_time: "02:00" } });
    const status = await f.scheduler.request("status", {}, f.scope); expect(status.reminder_due).toBe(true);
    await f.scheduler.request("defer", { signature: status.pending_signature }, f.scope);
    expect((await f.create().request("status", {}, f.scope)).reminder_due).toBe(false);
    await f.scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 23)); expect(f.starts).toHaveLength(0);
    await expect(f.scheduler.request("evolve", { signature: "wrong" }, f.scope)).rejects.toThrow("材料已变化");
  } finally { await f.close(); }
});

test("readonly status waits an existing SQL lease and coalesces concurrent polls", async () => {
  const f = fixture(); let inspected = 0;
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service, memory: () => f.memory,
    async inspect() { inspected++; return { records, core: null, execution_ready: true }; } });
  try {
    const { transaction } = await f.memory.dispatch("db_begin", [], {}, f.scope.identity);
    const first = scheduler.request("status", {}, f.scope), second = scheduler.request("status", {}, f.scope);
    expect(first).toBe(second);
    await Promise.resolve(); expect(inspected).toBe(0);
    await f.memory.dispatch("db_execute", [], { sql: "CREATE TABLE owner_data (value TEXT)" }, f.scope.identity, transaction);
    await f.memory.dispatch("db_commit", [], {}, f.scope.identity, transaction);
    await f.memory.dispatch("db_close", [], {}, f.scope.identity, transaction);
    expect((await first).pending_signature).toBeTruthy(); expect(inspected).toBe(1); expect(f.starts).toHaveLength(0);
    expect(f.memory.connect(db => db.query("SELECT name FROM sqlite_master WHERE name='owner_data'").get())).toBeTruthy();
  } finally { await f.close(); }
});

test("manual choices wait another lease and status waits an in-flight tick without starting training", async () => {
  const f = fixture(); let inspected = 0, transaction: string | undefined;
  let releaseInspection!: () => void;
  const inspectionStarted = new Promise<void>(resolve => { releaseInspection = resolve; });
  f.settings.update({ automatic: { mode: "manual", scheduled_time: "02:00" } });
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service, memory: () => f.memory,
    async inspect() {
      inspected++;
      if (inspected === 1) {
        ({ transaction } = await f.memory.dispatch("db_begin", [], {}, f.scope.identity));
        releaseInspection();
      }
      return { records, core: null, execution_ready: true };
    } });
  try {
    const tick = scheduler.request("tick", {}, f.scope);
    await inspectionStarted;
    const status = scheduler.request("status", {}, f.scope);
    await f.memory.dispatch("db_close", [], {}, f.scope.identity, transaction);
    expect((await tick).reminder_due).toBe(true);
    expect((await status).reminder_due).toBe(true);
    expect(inspected).toBe(2); expect(f.starts).toHaveLength(0);
  } finally { await f.close(); }
});

test("unavailable runtime reports a real blocker without consuming the day or falling back", async () => {
  const f = fixture();
  try {
    f.setReady(false); const status = await f.scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 2));
    expect(status.state).toBe("waiting_validation"); expect(status.reason).toContain("尚未完成"); expect(f.starts).toHaveLength(0);
    f.setReady(true); await f.scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 2)); await f.service.wait(); expect(f.starts).toHaveLength(1);
  } finally { await f.close(); }
});

test("native resume keeps the existing run id and refuses unsupported recovery before model handoff", async () => {
  const f = fixture();
  try {
    f.addRun({ id: "abcdef123456", kind: "evolution", state: "paused" }); f.setResume(false);
    await expect(f.scheduler.request("resume", { run_id: "abcdef123456" }, f.scope)).rejects.toThrow("保留原断点"); expect(f.starts).toHaveLength(0);
    f.setResume(true); await f.scheduler.request("resume", { run_id: "abcdef123456" }, f.scope); await f.service.wait();
    expect(f.starts[0].resume).toBe("abcdef123456");
  } finally { await f.close(); }
});

test("manual learn with automatic disabled resumes its existing learning run only after recovery validation", async () => {
  const f = fixture(); const options: any[] = [], start = f.service.start.bind(f.service);
  f.service.start = value => { options.push(value); return start(value); };
  try {
    f.settings.update({ automatic: { learning_enabled: false } });
    f.addRun({ id: "abcdef123456", kind: "learning", state: "paused" });
    f.setResume(false);
    await expect(f.scheduler.request("learn", {}, f.scope)).rejects.toThrow("保留原断点");
    f.setResume(true); f.setReady(false);
    await expect(f.scheduler.request("learn", {}, f.scope)).rejects.toThrow("保留原断点");
    expect(options).toHaveLength(0); expect(f.starts).toHaveLength(0);
    f.setReady(true);
    await f.create().request("learn", {}, f.scope); await f.service.wait();
    expect(f.starts).toHaveLength(1); expect(f.starts[0].resume).toBe("abcdef123456");
    expect(options[0]).toEqual({ continuous: false, resume: "abcdef123456", mode: "learning" });
  } finally { await f.close(); }
});

test("manual learn cannot claim a pending core evolution or other task", async () => {
  for (const kind of ["evolution", "router"]) {
    const f = fixture();
    try {
      f.addRun({ id: "abcdef123456", kind, state: "paused" });
      await expect(f.scheduler.request("learn", {}, f.scope)).rejects.toThrow("断点");
      expect(f.starts).toHaveLength(0);
    } finally { await f.close(); }
  }
});

test("expert scheduling uses authorized small cohorts and binds current core and expert without a 350-item gate", async () => {
  const f = fixture();
  const inspection = { material_kind: "expert" as const, records: { train: Array.from({ length: 16 }, (_, i) => row("expert-train-" + i)),
    valid: [row("expert-valid-0", "validation"), row("expert-valid-1", "validation")], protection: [] },
    core: "core-abcdef123456", expert: null, unspent_count: 16, execution_ready: true,
    steps: 192, material_target: 16, arguments: { steps: 64, protocol: "native-expert-preservation-v1" } };
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service, memory: () => f.memory,
    inspect: async () => inspection });
  try {
    const state = await scheduler.request("status", {}, f.scope);
    expect(state.experience).toMatchObject({ current: 16, target: 16, steps: 192 });
    const signature = state.pending_signature;
    expect(signature).not.toBe(nativeAutomaticSignature(f.scope.identity, inspection.core, inspection.records,
      { ...inspection, expert: "learned-123456abcdef" }));
    await scheduler.request("learn", {}, f.scope); await f.service.wait();
    expect(f.starts[0].automatic.signature).toBe(signature);
    inspection.execution_ready = false;
    await expect(scheduler.request("learn", {}, f.scope)).rejects.toThrow("尚未完成");
    expect(f.starts).toHaveLength(1);
  } finally { await f.close(); }
});

test("explicit learning retries a quality-rejected expert cohort while automatic scheduling stays deduplicated", async () => {
  const f = fixture();
  const options: any[] = [], start = f.service.start.bind(f.service);
  f.service.start = value => { options.push(value); return start(value); };
  const inspection = { material_kind: "expert" as const, records: {
    train: Array.from({ length: 16 }, (_, i) => row("retry-train-" + i)),
    valid: [row("retry-valid-a", "validation"), row("retry-valid-b", "validation")], protection: [] },
    core: "core-abcdef123456", expert: "learned-abcdef123456", unspent_count: 16, execution_ready: true,
    steps: 192, arguments: { steps: 64, protocol: "native-mlx-expert-preservation-v1" } };
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service,
    memory: () => f.memory, inspect: async () => inspection });
  try {
    const signature = nativeAutomaticSignature(f.scope.identity, inspection.core, inspection.records, inspection);
    f.addRun({ id: "abcdef123456", kind: "learning", state: "rejected", automatic_signature: signature,
      result: { promoted: false, quality_checked: true, actual_steps: 192 } });
    await scheduler.request("tick", {}, f.scope);
    expect(f.starts).toHaveLength(0);
    await expect(scheduler.request("start", {}, f.scope)).rejects.toThrow("已经处理");
    f.settings.update({ automatic: { learning_enabled: false } });
    await scheduler.request("learn", {}, f.scope); await f.service.wait();
    expect(f.starts).toHaveLength(1);
    expect(f.starts[0].automatic.signature).toBe(signature);
    expect(options[0]).toMatchObject({ mode: "learning", continuous: false });
    inspection.execution_ready = false;
    await expect(scheduler.request("learn", {}, f.scope)).rejects.toThrow("尚未完成");
    expect(f.starts).toHaveLength(1);
  } finally { await f.close(); }
});

test.each([
  { state: "rejected", result: { promoted: false, quality_checked: true, actual_steps: 0 } },
  { state: "rejected", result: { promoted: false, quality_checked: false, actual_steps: 192 } },
  { state: "rejected", result: { promoted: false, quality_checked: true, actual_steps: -1 } },
  { state: "committed", result: { promoted: true, quality_checked: true, actual_steps: 24 } },
  { state: "failed", result: { promoted: false, quality_checked: true, actual_steps: 192 } },
])("manual retry cannot reopen an inadmissible terminal cohort: %j", async prior => {
  const f = fixture();
  const inspection = { material_kind: "expert" as const, records: {
    train: Array.from({ length: 16 }, (_, i) => row("guard-train-" + i)),
    valid: [row("guard-valid-a", "validation"), row("guard-valid-b", "validation")], protection: [] },
    core: "core-abcdef123456", expert: "learned-abcdef123456", unspent_count: 16, execution_ready: true,
    steps: 192, arguments: { steps: 64, protocol: "native-mlx-expert-preservation-v1" } };
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service,
    memory: () => f.memory, inspect: async () => inspection });
  try {
    f.addRun({ id: "abcdef123456", kind: "learning", automatic_signature:
      nativeAutomaticSignature(f.scope.identity, inspection.core, inspection.records, inspection), ...prior });
    await expect(scheduler.request("learn", {}, f.scope)).rejects.toThrow("已经处理");
    expect(f.starts).toHaveLength(0);
  } finally { await f.close(); }
});

test("learning contents reuse inspected expert rows and invalidate changed sources without another inspection", async () => {
  const f = fixture(); let inspections = 0;
  const inspection: any = { material_kind: "expert", inference_engine: "mlx-q4", records: {
    train: [row("actual-training")], valid: [row("actual-validation", "validation")], protection: [row("protection", "validation")] },
    core: null, expert: null, execution_ready: false, readiness_error: "真实MLX验证尚未通过" };
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service, memory: () => f.memory,
    inspect: async () => { inspections++; return inspection; } });
  try {
    const empty = await scheduler.request("contents", {}, f.scope);
    expect(empty.records).toBeNull(); expect(empty.state).toBe("waiting_validation"); expect(inspections).toBe(0);
    await scheduler.request("status", {}, f.scope);
    const contents = await scheduler.request("contents", {}, f.scope);
    expect(contents.records).toEqual({ train: inspection.records.train, valid: inspection.records.valid });
    expect(contents.error).toBe(inspection.readiness_error); expect(contents.binding.engine).toBe("mlx-q4");
    contents.records.train[0].prompt = "mutated client copy";
    expect((await scheduler.request("contents", {}, f.scope)).records.train[0].prompt).toBe("actual-training");
    expect(inspections).toBe(1);
    mkdirSync(f.scope.store, { recursive: true }); writeFileSync(join(f.scope.store, "active.json"), JSON.stringify({ id: "changed-expert" }));
    expect((await scheduler.request("contents", {}, f.scope)).records).toBeNull(); expect(inspections).toBe(1);
    await scheduler.request("status", {}, f.scope);
    f.settings.update({ native_inference_precision: "mlx-q4", native_mlx_model_path: "/fixture/mlx" });
    expect((await scheduler.request("contents", {}, f.scope)).records).toBeNull(); expect(inspections).toBe(2);
  } finally { await f.close(); }
});

test("ready expert runtime waits for material without consuming its scheduled day", async () => {
  const f = fixture();
  const inspection: any = { material_kind: "expert", records: null, material_ready: false, core: "core-abcdef123456",
    execution_ready: true, readiness_error: "等待足够的已核验专家资料", steps: 192, material_target: 16, arguments: { steps: 64 } };
  const selected: any[] = [], start = f.service.start.bind(f.service);
  f.service.start = options => { selected.push(options); return start(options); };
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service, memory: () => f.memory, inspect: async () => inspection });
  try {
    const waiting = await scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 2));
    expect(waiting.state).toBe("waiting_materials"); expect(waiting.experience.target).toBe(16); expect(selected).toHaveLength(0);
    await expect(scheduler.request("learn", {}, f.scope)).rejects.toThrow("等待足够");
    inspection.records = { train: Array.from({ length: 16 }, (_, i) => row("fresh-" + i)), valid: [row("valid-a"), row("valid-b")], protection: [] };
    inspection.material_ready = true; inspection.readiness_error = null;
    await scheduler.request("tick", {}, f.scope, new Date(2026, 9, 17, 2)); await f.service.wait();
    expect(selected).toHaveLength(1); expect(selected[0].continuous).toBe(true); expect(selected[0].mode).toBe("learning");
  } finally { await f.close(); }
});

test.each([192, 256, 450])("busy automatic status uses the actual session budget %s without inspecting model materials", async budget => {
  const f = fixture(); let finish!: () => void, entered!: () => void;
  const running = new Promise<void>(resolve => { entered = resolve; }), done = new Promise<void>(resolve => { finish = resolve; });
  const service = new NativeEvolutionService({ async compute(path, body) {
    return Response.json(path.endsWith("acquire") ? { ...f.scope, lease_id: body.lease_id, model: "/fixture/model", was_ready: false }
      : { released: true, was_ready: false });
  }, async open() { return { ...(budget === 450 ? {} : { stepBudget: budget, earlyStop: true, kind: "evolution" as const }),
    async runRound() { entered(); await done; return { step: 0, phase: "paused", quality_checked: false, adopted: false, pending: true }; }, async close() {} }; } });
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service, memory: () => { throw Error("Busy status must not reopen storage"); },
    async inspect() { throw Error("Busy status must not inspect material"); } });
  try {
    service.start();
    expect((await scheduler.request("status", {}, f.scope)).experience).toMatchObject({ steps: 0, target: 0 });
    await running;
    expect(service.status()).toMatchObject({ steps: budget, earlyStop: budget !== 450 });
    expect((await scheduler.request("status", {}, f.scope)).experience).toMatchObject({ steps: budget,
      target: budget === 450 ? 350 : 16, batch_limit: budget === 450 ? 57 : 16 });
    finish(); await service.wait();
  } finally { finish(); await service.close(); await f.close(); }
});

test.each([true, false])("pending expert status retains inspected material and budget (resumable=%s)", async resumable => {
  const f = fixture(); f.addRun({ id: "abcdef123456", kind: "learning", state: "paused" });
  const scheduler = new NativeEvolutionAutomatic({ settings: f.settings, service: f.service, memory: () => f.memory,
    async inspect() { return { material_kind: "expert", records: null, core: "core-abcdef123456", unspent_count: 23,
      execution_ready: true, resume_supported: resumable, steps: 256, material_target: 16 }; } });
  try {
    const result = await scheduler.request("status", {}, f.scope);
    expect(result).toMatchObject({ state: resumable ? "paused" : "waiting_validation", resumable,
      experience: { current: 23, target: 16, steps: 256, batch_limit: 16, continuing: true } });
  } finally { await f.close(); }
});

test("idle expert learning is independent of manual evolution and starts each cohort only once", async () => {
 const f=fixture();
 f.settings.update({automatic:{mode:"manual",scheduled_time:"23:59"}});
 const inspected={material_kind:"expert" as const,records:{train:Array.from({length:16},(_,i)=>row("daily-"+i)),valid:[row("daily-v0","validation"),row("daily-v1","validation")],protection:[]},core:"core-abcdef123456",execution_ready:true,material_ready:true,unspent_count:16,steps:192,material_target:16,arguments:{steps:64}};
 const options: any[] = [], start=f.service.start.bind(f.service); f.service.start=value=>{options.push(value);return start(value);};
 const make=()=>new NativeEvolutionAutomatic({settings:f.settings,service:f.service,memory:()=>f.memory,inspect:async()=>inspected});
 try {
  const scheduler=make();const before=await scheduler.request("status",{},f.scope);expect(f.starts).toHaveLength(0);
  const started=await scheduler.request("tick",{},f.scope,new Date(2026,9,20,10));await f.service.wait();
  expect(started.state).toBe("running");expect(f.starts).toHaveLength(1);expect(options[0].mode).toBe("learning");
  expect(f.starts[0].automatic).toEqual({signature:before.pending_signature,scheduled_occurrence:null});
  f.addRun({id:"actual-daily",kind:"learning",state:"rejected",automatic_signature:before.pending_signature,result:{reason:"当前MLX来源未保持硬保护基线"}});
  const rejected=await make().request("status",{},f.scope);
  expect(rejected.state).toBe("waiting_validation");expect(rejected.reason).toBe("旧能力检查未通过，尚未开始训练。");expect(rejected.pending_signature).toBeNull();
  await make().request("tick",{},f.scope,new Date(2026,9,20,11));expect(f.starts).toHaveLength(1);
 } finally {await f.close();}
});

test("idle learning retries a busy handoff before a training journal exists", async () => {
 const f=fixture();f.settings.update({automatic:{mode:"manual",scheduled_time:"23:59"}});let acquisitions=0,rounds=0;
 const service=new NativeEvolutionService({async compute(path,body){if(path.endsWith("acquire")){if(++acquisitions===1)return Response.json({error:"busy"},{status:409});return Response.json({...f.scope,lease_id:body.lease_id,model:"/fixture/model",was_ready:false});}return Response.json({released:true,was_ready:false});},async open(){return{async runRound(){rounds++;return{step:0,phase:"rejected",quality_checked:true,adopted:false,pending:false};},async close(){}};}});
 const inspect=async()=>({material_kind:"expert" as const,records:{train:Array.from({length:16},(_,i)=>row("retry-"+i)),valid:[row("rv0","validation"),row("rv1","validation")],protection:[]},core:null,execution_ready:true,steps:192,arguments:{steps:64}});
 const make=()=>new NativeEvolutionAutomatic({settings:f.settings,service,memory:()=>f.memory,inspect});
 try{await make().request("tick",{},f.scope);await service.wait();expect(rounds).toBe(0);await make().request("tick",{},f.scope);await service.wait();expect(rounds).toBe(1);expect(acquisitions).toBe(2);}finally{await service.close();await f.close();}
});

test.each(["learn", "start"])("disabled automatic learning survives reopen and permits explicit %s", async method => {
 const f=fixture();
 const inspected={material_kind:"expert" as const,records:{train:Array.from({length:16},(_,i)=>row("switch-"+i)),valid:[row("sv0","validation"),row("sv1","validation")],protection:[]},core:null,execution_ready:true,steps:192,unspent_count:16,arguments:{steps:64}};
 try {
  expect(f.settings.snapshot().automatic.learning_enabled).toBe(true);
  f.settings.update({automatic:{learning_enabled:false}});
  const settings=new Settings(join(f.root,"home"));
  expect(settings.snapshot().automatic.learning_enabled).toBe(false);
  settings.update({automatic:{mode:"auto",scheduled_time:"02:00"}});
  expect(settings.snapshot().automatic.learning_enabled).toBe(false);
  const scheduler=new NativeEvolutionAutomatic({settings,service:f.service,memory:()=>f.memory,inspect:async()=>inspected});
  const status=await scheduler.request("status",{},f.scope);
  expect(status.experience.current).toBe(16);expect(status.pending_signature).toBeTruthy();
  await scheduler.request("tick",{},f.scope,new Date(2026,9,20,10));expect(f.starts).toHaveLength(0);
  await scheduler.request(method,{},f.scope);await f.service.wait();expect(f.starts).toHaveLength(1);
 } finally {await f.close();}
});

test("automatic learning switch does not disable scheduled core evolution", async()=>{
 const f=fixture();try{f.settings.update({automatic:{learning_enabled:false}});
  await f.scheduler.request("tick",{},f.scope,new Date(2026,9,20,10));await f.service.wait();expect(f.starts).toHaveLength(1);
 }finally{await f.close();}
});
