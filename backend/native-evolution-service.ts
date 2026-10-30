/** Own the application hand-off until native work and all native owners have closed. */
import type { NativeEvolutionMaterial } from "./native-evolution-cohort.ts";
import { randomBytes, randomUUID } from "node:crypto";

export type NativeEvolutionLease = {
  lease_id: string; was_ready: boolean; identity: string; root: string; store: string; model: string;
  manifest_sha256?: string; activeCore?: string | null;
};
export type NativeEvolutionRoundResult = {
  step: number; phase: string; quality_checked: boolean; adopted: boolean; pending: boolean;
  [key: string]: unknown;
};
export type NativeEvolutionSession = {
  /** Total actual Adam updates across all candidates; old core rounds remain fixed at 450. */
  readonly stepBudget?: number;
  readonly kind?: "learning" | "evolution";
  readonly earlyStop?: boolean;
  runRound(input: { signal: AbortSignal; round: number; resume?: string;
    automatic?: { signature: string; scheduled_occurrence: string | null };
    onProgress: (progress: { step: number; phase: string }) => void }): Promise<NativeEvolutionRoundResult>;
  continuationMaterial?(): NativeEvolutionMaterial | null;
  close(): Promise<void>;
};
type ServiceOptions = {
  compute(path: string, body: Record<string, unknown>): Promise<Response>;
  open(lease: NativeEvolutionLease, signal: AbortSignal, material: NativeEvolutionMaterial | null,
    mode: "learning" | "consolidation"): Promise<NativeEvolutionSession>;
};
function need(condition: unknown, message: string): asserts condition { if (!condition) throw new Error(message); }

export class NativeEvolutionService {
  private job: Record<string, any> = { state: "idle", kind: "evolution", round: 0, step: 0, steps: 450 };
  private controller?: AbortController;
  private task?: Promise<void>;
  private lease?: NativeEvolutionLease;
  private leaseID?: string;
  private session?: NativeEvolutionSession;
  private cleanupTask?: Promise<void>;
  private closing = false;
  private terminal = "failed";
  constructor(private readonly options: ServiceOptions) {}

  status() { return structuredClone({ ...this.job, running: !!this.task, lease_held: !!this.leaseID }); }

  start(options: { mode?: "learning" | "consolidation"; continuous?: boolean; resume?: string; automatic?: { signature: string; scheduled_occurrence: string | null } } = {}) {
    need(!this.closing, "应用正在退出");
    need(!this.task && !this.leaseID && !this.session && this.job.state !== "quarantined", "原生进化正在运行或等待完成清理");
    need(options.continuous === undefined || typeof options.continuous === "boolean", "持续进化选项无效");
    need(options.mode === undefined || ["learning", "consolidation"].includes(options.mode), "原生专家学习模式无效");
    need(options.resume === undefined || /^[a-f0-9]{12}$/.test(options.resume), "恢复任务标识无效");
    need(!options.automatic || /^[a-f0-9]{64}$/.test(options.automatic.signature) && (options.automatic.scheduled_occurrence === null
      || /^schedule:\d{4}-\d{2}-\d{2}$/.test(options.automatic.scheduled_occurrence)), "自动进化调度绑定无效");
    options = structuredClone(options);
    const controller = this.controller = new AbortController();
    this.job = { id: randomBytes(6).toString("hex"), kind: options.mode === "learning" ? "learning" : "evolution", state: "preparing", round: 0,
      step: 0, steps: 0, phase: "acquiring_model", started_at: new Date().toISOString(),
      continuous: options.continuous ?? true, resumed_from: options.resume ?? null, error: null, last_result: null };
    // Reserve synchronously: two HTTP requests cannot both enter the compute hand-off.
    const task = Promise.resolve().then(() => this.run(controller.signal, options));
    this.task = task;
    void task.finally(() => { if (this.task === task) this.task = undefined; });
    return this.status();
  }

  cancel() {
    if (this.task) { this.job.state = "cancelling"; this.controller!.abort(new Error("用户停止原生进化")); }
    return this.status();
  }

  async wait() { await this.task; return this.status(); }

  private async command(path: string, body: Record<string, unknown> = {}) {
    const response = await this.options.compute(path, body);
    let value: any;
    try { value = await response.json(); } catch { throw new Error("模型交接响应无效：" + path); }
    need(response.ok, typeof value?.detail === "string" ? value.detail : "模型交接失败：" + path);
    return value;
  }

  private async run(signal: AbortSignal, options: { mode?: "learning" | "consolidation"; continuous?: boolean; resume?: string; automatic?: { signature: string; scheduled_occurrence: string | null } }) {
    this.terminal = "failed";
    let continuationMaterial: NativeEvolutionMaterial | null = null;
    try {
      this.leaseID = randomUUID();
      const lease = await this.command("/internal/native-evolution/acquire", { lease_id: this.leaseID });
      // Retain a usable lease even if the remainder of the response is malformed, so release can run.
      need(lease && lease.lease_id === this.leaseID, "模型交接租约标识不匹配");
      this.lease = lease;
      need(typeof lease.was_ready === "boolean" && /^[a-f0-9]{64}$/.test(lease.identity)
        && [lease.root, lease.store, lease.model].every(p => typeof p === "string" && p.startsWith("/") && !p.includes("\0")), "模型交接范围无效");
      for (;;) {
        signal.throwIfAborted();
        this.job.round++; this.job.step = 0; this.job.steps = 0; delete this.job.earlyStop; this.job.phase = "preparing_round";
        this.session = await this.options.open(lease, signal, continuationMaterial, options.mode ?? "consolidation");
        const budget = this.session.stepBudget ?? 450, earlyStop = this.session.earlyStop === true;
        need(Number.isSafeInteger(budget) && budget > 0 && budget <= 450
          && (this.session.earlyStop === undefined || typeof this.session.earlyStop === "boolean")
          && (this.session.kind === undefined || ["learning", "evolution"].includes(this.session.kind)), "原生学习轮次预算无效");
        this.job.steps = budget; this.job.kind = this.session.kind ?? "evolution"; this.job.earlyStop = earlyStop;
        signal.throwIfAborted(); this.job.state = "running";
        const round = this.job.round;
        const result = await this.session.runRound({ signal, round, ...(round === 1 && options.resume ? { resume: options.resume } : {}),
          ...(round === 1 && options.automatic ? { automatic: options.automatic } : {}),
          onProgress: progress => {
            need(Number.isSafeInteger(progress.step) && progress.step >= this.job.step && progress.step <= budget
              && typeof progress.phase === "string", "原生进化进度无效");
            this.job.step = progress.step; this.job.phase = progress.phase;
          } });
        need(result && Number.isSafeInteger(result.step) && result.step >= this.job.step && result.step <= budget
          && typeof result.phase === "string" && typeof result.quality_checked === "boolean"
          && typeof result.adopted === "boolean" && typeof result.pending === "boolean", "原生进化结果无效");
        const waitingMaterial = result.phase === "committed_waiting_material";
        need(!result.adopted || (result.step === budget || earlyStop) && result.quality_checked && (result.phase === "committed" || waitingMaterial), "未完成质量验收的进化不能标为已采纳");
        need(!waitingMaterial || result.adopted && result.pending && typeof result.continuation_reason === "string"
          && result.continuation_reason.trim().length > 0, "等待新训练材料的进化结果无效");
        this.job.step = result.step; this.job.phase = result.phase; this.job.last_result = structuredClone(result);
        const finished = (result.step === budget || earlyStop) && result.quality_checked;
        const continueRound = !waitingMaterial && finished && result.adopted && result.pending && options.continuous !== false;
        continuationMaterial = continueRound ? structuredClone(this.session.continuationMaterial?.() ?? null) : null;
        // A successor must bind its own accepted state and materials. The model stays resident within a round.
        await this.session.close(); this.session = undefined;
        signal.throwIfAborted();
        if (!continueRound) {
          this.terminal = waitingMaterial ? "waiting_material" : finished && result.adopted ? "complete" : result.phase.startsWith("paused") ? "paused" : "unaccepted";
          break;
        }
      }
    } catch (error) {
      this.terminal = signal.aborted ? "paused" : "failed";
      this.job.error = String(error instanceof Error ? error.message : error);
    }
    try { await this.cleanup(); }
    catch (error) { this.job.state = "quarantined"; this.job.cleanup_error = String(error instanceof Error ? error.message : error); }
  }

  private cleanup() {
    if (this.cleanupTask) return this.cleanupTask;
    const task = this.cleanupActual(); this.cleanupTask = task;
    void task.then(() => { this.cleanupTask = undefined; }, () => { this.cleanupTask = undefined; });
    return task;
  }

  private async cleanupActual() {
    if (this.session) { await this.session.close(); this.session = undefined; }
    if (this.leaseID) {
      const released = await this.command("/internal/native-evolution/release", { lease_id: this.leaseID });
      need(released && typeof released.released === "boolean" && typeof released.was_ready === "boolean", "模型交接释放响应无效");
      need(released.released || !this.lease && !released.was_ready, "原生进化租约尚未确认释放");
      this.lease = undefined; this.leaseID = undefined;
      if (released.was_ready && !this.closing) {
        try { await this.command("/api/model/reload"); }
        catch (error) { this.job.restore_error = String(error instanceof Error ? error.message : error); this.terminal = "failed"; }
      }
    }
    this.job.state = this.terminal; this.job.finished_at = new Date().toISOString(); delete this.job.cleanup_error;
  }

  /** Wait for the current checkpoint/rollback and retry retained cleanup without starting a new round. */
  async pause() {
    this.cancel(); await this.task;
    try { await this.cleanup(); }
    catch (error) { this.job.state = "quarantined"; this.job.cleanup_error = String(error instanceof Error ? error.message : error); throw error; }
    return this.status();
  }

  /** An unsuccessful close remains retryable on the same native owners and compute lease. */
  async close() {
    this.closing = true; await this.pause();
  }
}
