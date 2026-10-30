import { t, locale } from "./i18n";

export type EvolutionSummary = {
  model_name: string | null;
  round_number: number | null;
  current: { state: string | null; phase: string | null; step: number | null; steps: number | null; materials: number | null; gaps: number | null };
  previous: { mastered: number | null; pending: number | null; retention: "stable" | "regressed" | "unverified" | null; outcome?: "published" | "safe_unadopted_continue" | null } | null;
  snapshot: { available: boolean | null; retained: boolean | null; id: string | null; version: string | null; error?: string | null };
  error: string | null;
};

type Runtime = { state: string; job?: { kind?: string; state: string; phase?: string; step?: number; steps?: number; round_step?: number; round_steps?: number; absolute_step?: number; result?: { continuation_reason?: string } } };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const text = (id: string, value: string) => { $(id).textContent = value; };
const knownCount = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const count = (value: unknown) => knownCount(value) ? value.toLocaleString(locale()) : "—";

export function initializeEvolutionSummary(api: <T>(path: string, body?: unknown) => Promise<T>) {
  let summary: EvolutionSummary | undefined;
  let runtime: Runtime | undefined;
  let request = 0;

  function render() {
    const job = ["evolution", "learning"].includes(runtime?.job?.kind || "") ? runtime!.job : undefined;
    // Summary validates frozen lineage. Until it arrives, local fields only apply
    // when the live runtime explicitly supplies the fixed round pair.
    const local = job?.round_steps === 450 && knownCount(job.round_step) && job.round_step <= 450;
    const hasRoundFields = job && [job.round_steps, job.round_step, job.absolute_step].some(value => value !== undefined);
    const current = job ? { ...job, ...(hasRoundFields ? { step: local ? job.round_step : null, steps: local ? 450 : null } : {}),
      materials: summary?.current.materials ?? null, gaps: summary?.current.gaps ?? null } : summary?.current;
    const waitingMaterial = job?.state === "waiting_material";
    const state = waitingMaterial ? job.state : current?.state;
    const phase = current?.phase;
    const active = state === "running" || state === "queued";
    const states: Record<string, string> = { waiting_material: t("等待新训练材料"), idle: t("等待进化"), queued: t("等待开始"), paused: t("已暂停"), complete: t("已完成"), rejected: t("未采纳"), error: t("未完成") };
    const verifying = ["training_check", "validation", "retention", "native_check"].includes(phase || "");
    const stateLabel = state === "running" ? verifying ? t("验证中") : phase === "training" ? t("学习中") : phase === "commit" ? t("保存中") : phase === "preparing" ? t("准备中") : t("进化中")
      : states[state || ""] || ({ ready: t("等待进化"), stopped: t("未启动"), loading: t("加载中"), error: t("模型异常"), evaluating: t("评测中") } as Record<string, string>)[runtime?.state || ""] || t("待核验");
    const round = knownCount(summary?.round_number) && summary!.round_number! > 0;
    text("evolution-cycle-round", round ? t`第 ${count(summary!.round_number)} 轮进化` : t("轮次待确认"));
    const step = current?.step, steps = current?.steps;
    const hasProgress = knownCount(step) && knownCount(steps) && steps > 0 && step <= steps;
    text("evolution-cycle-progress", hasProgress ? t`${active ? t("正在成长") : stateLabel}：${count(step)} / ${count(steps)} 步`
      : active ? t("正在成长 · 步数待确认") : state === "idle" || !state ? t("等待开始成长") : t`${stateLabel} · 步数待确认`);
    text("evolution-cycle-next", waitingMaterial ? t(job.result?.continuation_reason || "等待新训练材料") : active && verifying ? t("正在验证新知识与原有能力") : active && phase === "commit" ? t("验证已完成，正在保存结果")
      : active ? t("训练期间检查能力，验证通过后提前结束") : t("学习专家模块，验证通过后结束本轮"));
    const progress = $("evolution-progress") as HTMLProgressElement;
    progress.hidden = !hasProgress && !active;
    progress.max = hasProgress ? steps : 1;
    if (hasProgress) progress.value = step;
    else progress.removeAttribute("value");
    $("evolution-panel").style.setProperty("--evolution-progress", String(hasProgress ? step / steps : 0));
  }

  async function refresh() {
    const revision = ++request;
    try {
      const next = await api<EvolutionSummary>("/api/evolution/summary");
      if (revision !== request) return;
      summary = next;
    } catch {
      if (revision !== request) return;
      summary = undefined;
    }
    render();
  }
  function renderRuntime(value: Runtime) { runtime = value; render(); }
  render();
  return { refresh, renderRuntime };
}
