/** Native learning/evolution scheduling. Other idle work keeps its existing priority. */
import { Memory, ledgerParse } from "./memory";
import { Settings } from "./settings";
import { materialContentSHA, validateNativeProductRecords } from "./native-evolution-material";
import { NativeEvolutionService } from "./native-evolution-service";
import { readFileSync } from "node:fs";
import { join } from "node:path";

export type NativeEvolutionScope = { root: string; store: string; identity: string };
export type NativeEvolutionInspection = { records: any; core: string | null; expert?: string | null;
  inference_engine?: string;
  unspent_count?: number; execution_ready: boolean; resume_supported?: boolean; readiness_error?: string | null;
  material_kind?: "expert"; material_ready?: boolean; steps?: number; material_target?: number; arguments?: Record<string, unknown> };
export const NATIVE_AUTOMATIC_ARGUMENTS = { steps: 450, learning_rate: 1e-5, distillation_weight: 5,
  preserve_replay_subspace: true, partial_policy: "closed-book-growth-v1" };
const product = (run: any) => run.scope !== "research" || run.product_accepted === true;
export function nativeScheduleDay(now: Date) {
  return "schedule:" + [now.getFullYear(), String(now.getMonth() + 1).padStart(2, "0"), String(now.getDate()).padStart(2, "0")].join("-");
}
export function nativeEvolutionAttempted(signature: string, runs: any[], now = new Date()) {
  return runs.some(run => {
    if (run.automatic_signature !== signature) return false;
    if (run.kind === "evolution" && product(run) && run.state === "rejected"
      && ["closed-book-growth-v1", "closed-book-growth-450-v1"].includes(run.arguments?.partial_policy)) {
      const day = run.scheduled_occurrence || nativeScheduleDay(new Date(Number(run.started) * 1000));
      if (day < nativeScheduleDay(now)) return false;
    }
    return true;
  });
}
export function nativeAutomaticSignature(identity: string, core: string | null, records: any, expert?: NativeEvolutionInspection) {
  return materialContentSHA({ kind: expert?.material_kind === "expert" ? "learning" : "evolution", identity, core,
    expert: expert?.expert ?? null, arguments: expert?.material_kind === "expert" ? expert.arguments : NATIVE_AUTOMATIC_ARGUMENTS, records });
}
type Options = {
  settings: Settings; service: NativeEvolutionService; memory(root: string): Memory;
  inspect(scope: NativeEvolutionScope, memory: Memory): Promise<NativeEvolutionInspection>;
};
function need(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
export class NativeEvolutionAutomatic {
  private active?: Promise<any>;
  private activeMethod?: string;
  private activeScope?: string;
  private contentCache?: { source: string; value: Record<string, any> };
  constructor(private readonly options: Options) {}
  private contentSource(scope: NativeEvolutionScope) {
    const settings = this.options.settings.snapshot();
    const pointers = ["active-core.json", "active.json", "active-router.json"].map(name => {
      try { return readFileSync(join(scope.store, name), "utf8"); }
      catch (error: any) { if (error.code === "ENOENT") return null; throw error; }
    });
    return JSON.stringify([scope, settings.native_inference_precision, settings.native_mlx_model_path, pointers]);
  }
  private async inspect(scope: NativeEvolutionScope, memory: Memory) {
    this.contentCache = undefined;
    const source = this.contentSource(scope), result = await this.options.inspect(scope, memory);
    if (source !== this.contentSource(scope)) { this.contentCache = undefined; return result; }
    this.contentCache = { source, value: {
      records: result.material_kind === "expert" && result.records ? {
        train: structuredClone(result.records.train), valid: structuredClone(result.records.valid),
      } : null,
      state: result.execution_ready ? "ready" : "waiting_validation",
      error: result.execution_ready ? null : result.readiness_error || "专家学习来源尚未通过验证",
      binding: { engine: result.inference_engine ?? this.options.settings.snapshot().native_inference_precision,
        core: result.core, expert: result.expert ?? null },
    } };
    return result;
  }
  request(method: string, payload: Record<string, any>, scope: NativeEvolutionScope, now = new Date()): Promise<any> {
    need(["status", "contents", "tick", "start", "learn", "evolve", "defer", "resume"].includes(method), "未知原生进化调度操作");
    if (method === "contents") {
      try {
        need(scope && /^[a-f0-9]{64}$/.test(scope.identity) && typeof scope.store === "string" && scope.store.startsWith("/") && !scope.store.includes("\0"), "原生进化数据范围尚未就绪");
        const cached = this.contentCache;
        return Promise.resolve(cached && cached.source === this.contentSource(scope) ? structuredClone(cached.value)
          : { records: null, state: "waiting_validation", error: "正在核对专家学习资料，请稍后刷新。", binding: null });
      } catch (error) { return Promise.reject(error); }
    }
    if (this.active) {
      if (method === "status") {
        if (this.activeMethod === "status" && this.activeScope === JSON.stringify(scope)) return this.active;
        return this.active.then(() => this.request(method, payload, scope, now), () => this.request(method, payload, scope, now));
      }
      return Promise.reject(new Error("原生进化调度正在读取资料，请稍后重试"));
    }
    const task = this.perform(method, payload, scope, now); this.active = task;
    this.activeMethod = method; this.activeScope = JSON.stringify(scope);
    const clear = () => { this.active = undefined; this.activeMethod = undefined; this.activeScope = undefined; };
    void task.then(clear, clear); return task;
  }
  private async choices(memory: Memory, identity: string, signature: string, action?: string) {
    const { transaction } = await memory.dispatch("db_begin", [], {}, identity);
    try {
      await memory.dispatch("db_execute", [], { sql: "CREATE TABLE IF NOT EXISTS automatic_choices (identity TEXT,signature TEXT,action TEXT,created REAL,PRIMARY KEY(identity,signature))" }, identity, transaction);
      if (action) await memory.dispatch("db_execute", [], { sql: "INSERT INTO automatic_choices VALUES (?,?,?,?) ON CONFLICT(identity,signature) DO UPDATE SET action=excluded.action,created=excluded.created", parameters: [identity, signature, action, Date.now() / 1000] }, identity, transaction);
      const result = await memory.dispatch("db_execute", [], { sql: "SELECT action FROM automatic_choices WHERE identity=? AND signature=?", parameters: [identity, signature] }, identity, transaction);
      await memory.dispatch("db_commit", [], {}, identity, transaction);
      return result.rows[0]?.[0] ?? null;
    } finally { await memory.dispatch("db_close", [], {}, identity, transaction); }
  }
  private async runs(memory: Memory, identity: string) {
    const { transaction } = await memory.dispatch("db_begin", [], {}, identity);
    try {
      const exists = await memory.dispatch("db_execute", [], { sql: "SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'" }, identity, transaction);
      if (!exists.rows.length) return [];
      const result = await memory.dispatch("db_execute", [], { sql: "SELECT id,kind,state,body FROM training_runs WHERE identity=? ORDER BY updated DESC", parameters: [identity] }, identity, transaction);
      return result.rows.map(([id, kind, state, raw]: any[]) => {
        const body = ledgerParse(raw); need(body.id === id && body.kind === kind && body.binding?.identity === identity, "进化账本身份不一致，等待恢复");
        return { ...body, state };
      });
    } finally { await memory.dispatch("db_close", [], {}, identity, transaction); }
  }
  private async perform(method: string, payload: Record<string, any>, scope: NativeEvolutionScope, now: Date) {
    need(scope && /^[a-f0-9]{64}$/.test(scope.identity) && [scope.root, scope.store].every(path => typeof path === "string" && path.startsWith("/") && !path.includes("\0")), "原生进化数据范围尚未就绪");
    const { settings, service } = this.options, current = service.status(), config = settings.snapshot().automatic;
    const result: Record<string, any> = { ...config, state: "waiting_materials", reason: "等待350条有效去重新内容；已有经验不会因重试增加",
      pending_signature: null, reminder_due: false, experience: { current: 0, target: 350, continuing: false, steps: 450, batch_limit: 57 } };
    const inspectedExperience = (inspected: NativeEvolutionInspection) => {
      if (inspected.material_kind === "expert") result.experience = { current: inspected.unspent_count ?? 0,
        target: inspected.material_target ?? 16, continuing: true, steps: inspected.steps, batch_limit: inspected.material_target ?? 16 };
      else result.experience.current = inspected.unspent_count ?? 0;
    };
    if (current.running || current.lease_held || current.state === "quarantined") {
      if (!["status", "tick"].includes(method)) throw new Error("原生进化正在运行或等待完成清理");
      result.experience = { ...result.experience, steps: current.steps,
        ...(current.earlyStop === true ? { target: 16, batch_limit: 16, continuing: true }
          : current.steps === 0 ? { target: 0, batch_limit: 0 } : {}) };
      return { ...result, state: "running", reason: current.cleanup_error ?? "正在原生进化并检查新旧能力", job_id: current.id };
    }
    const memory = this.options.memory(scope.root), runs = await this.runs(memory, scope.identity);
    const pending = runs.find(run => ["queued", "running", "paused", "committing"].includes(run.state));
    if (pending) {
      need(product(pending) && ["evolution", "learning"].includes(pending.kind), "已有其它学习或研究断点，原生进化等待其完成");
      if (method === "status") {
        try {
          const prepared = await this.inspect(scope, memory);
          inspectedExperience(prepared);
          const resumable = prepared.execution_ready === true && prepared.resume_supported === true;
          return { ...result, state: resumable ? "paused" : "waiting_validation", resumable,
            reason: resumable ? "原生进化有可恢复断点" : "原生进化断点恢复尚未完成接线，已保留原断点" };
        } catch (error) {
          return { ...result, state: "waiting_validation", resumable: false, reason: String(error instanceof Error ? error.message : error) };
        }
      }
      need(method === "resume" && payload.run_id === pending.id || method === "learn" && pending.kind === "learning",
        "原生进化存在待恢复断点，不能另开新轮");
      const prepared = await this.inspect(scope, memory);
      need(prepared.execution_ready === true && prepared.resume_supported === true, "原生进化断点恢复尚未完成接线，已保留原断点");
      const job = service.start({ continuous: false, resume: pending.id,
        mode: pending.kind === "learning" ? "learning" : "consolidation" });
      return { ok: true, job_id: job.id };
    }
    need(method !== "resume", "原生进化断点已改变，请刷新后重试");
    let inspected: Awaited<ReturnType<Options["inspect"]>>;
    try {
      inspected = await this.inspect(scope, memory);
      inspectedExperience(inspected);
      if (inspected.material_kind === "expert") {
        need(inspected.execution_ready === true, inspected.readiness_error || "原生专家运行程序尚未完成接线验收");
        if (inspected.material_ready === false || inspected.records === null) {
          const reason = inspected.readiness_error || "等待足够的已核验专家学习资料和独立验证资料";
          if (["status", "tick"].includes(method)) return { ...result, state: "waiting_materials", reason };
          throw new Error(reason);
        }
        need(inspected.records && Array.isArray(inspected.records.train) && inspected.records.train.length >= 16
          && Array.isArray(inspected.records.valid) && inspected.records.valid.length >= 2
          && Array.isArray(inspected.records.protection), "专家学习需要合法训练资料和独立验证资料");
        need(Number.isSafeInteger(inspected.steps) && inspected.steps! > 0 && inspected.steps! <= 256
          && inspected.arguments?.steps === 64, "专家实际训练预算无效");
        result.reason = "已核验资料用于专家学习；核心保持冻结，通过完整验证即可结束本轮";
      } else {
        validateNativeProductRecords(inspected.records);
        need("experience" in inspected.records, "日常自动进化只使用已核验350经验池，身份定向教学使用独立授权入口");
        need(inspected.execution_ready === true, "原生进化运行程序尚未完成接线验收");
      }
    } catch (error) {
      if (!["status", "tick"].includes(method)) throw error;
      return { ...result, state: "waiting_validation", reason: String(error instanceof Error ? error.message : error), evolution_blocked_reason: String(error instanceof Error ? error.message : error) };
    }
    const signature = nativeAutomaticSignature(scope.identity, inspected.core, inspected.records, inspected);
    result.experience = { ...result.experience, current: inspected.unspent_count ?? 0, continuing: true };
    if (!nativeEvolutionAttempted(signature, runs, now)) {
      result.pending_signature = signature;
      result.reminder_due = config.mode === "manual" && await this.choices(memory, scope.identity, signature) === null;
    } else if (inspected.material_kind === "expert") {
      const previous = runs.find(run => run.automatic_signature === signature);
      if (previous?.state === "rejected") {
        result.state = "waiting_validation";
        result.reason = previous.result?.reason === "当前MLX来源未保持硬保护基线"
          ? "旧能力检查未通过，尚未开始训练。"
          : previous.result?.reason === "新知识尚未被当前专家路由覆盖，等待路由校准；未执行无效训练"
            ? "待学内容尚未接入专家学习分支，本轮未开始训练。" : "本轮学习未通过检查，结果未采纳。";
        result.evolution_blocked_reason = result.reason;
      }
    }
    if (method === "status") return result;
    if (method === "defer") {
      need(payload.signature === signature, "进化材料已变化，请刷新后重试");
      await this.choices(memory, scope.identity, signature, "deferred"); return { ok: true };
    }
    if (method === "evolve") need(payload.signature === signature, "进化材料已变化或本轮已处理，请刷新后重试");
    const day = nativeScheduleDay(now);
    const idleLearning = method === "tick" && inspected.material_kind === "expert";
    if (idleLearning) {
      if (settings.snapshot().automatic.learning_enabled === false) return { ...result, learning_enabled: false, reason: "", reminder_due: false };
      if (nativeEvolutionAttempted(signature, runs, now)) return result;
    } else if (method === "tick") {
      const time = String(now.getHours()).padStart(2, "0") + ":" + String(now.getMinutes()).padStart(2, "0");
      if (config.mode !== "auto" || time < config.scheduled_time || await this.choices(memory, scope.identity, day)
        || runs.some(run => run.scheduled_occurrence === day) || nativeEvolutionAttempted(signature, runs, now)) return result;
    } else {
      const previous = runs.filter(run => run.automatic_signature === signature);
      const retry = method === "learn" && inspected.material_kind === "expert" && previous.length > 0
        && previous.every(run => product(run) && run.kind === "learning" && run.state === "rejected"
          && run.result?.promoted === false && run.result?.quality_checked === true
          && Number.isSafeInteger(run.result?.actual_steps) && run.result.actual_steps > 0);
      need(retry || !nativeEvolutionAttempted(signature, runs, now), "这批材料与训练配置已经处理，请等待新的核验资料");
    }
    // Expert evolution can continue with newly bound material; an explicit learning request runs one round.
    const job = service.start({ continuous: inspected.material_kind === "expert" && ["start", "evolve", "tick"].includes(method),
      mode: method === "learn" || idleLearning ? "learning" : "consolidation",
      automatic: { signature, scheduled_occurrence: method === "tick" && !idleLearning ? day : null } });
    if (!idleLearning) await this.choices(memory, scope.identity, method === "tick" ? day : signature, "started");
    return { ok: true, job_id: job.id, state: "running", reason: "正在原生进化并检查新旧能力" };
  }
}
