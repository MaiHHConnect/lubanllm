import { initializeLanguage, localizeDocument, locale, t, LEARNING_SOURCE_RECHECK_NOTICE } from "./i18n";
import { COUNTRY_CODES, countryLabel } from "./countries";
import { initializeUpdates } from "./updates";
import { initializeModelTools } from "./model-tools";
import { initializeHome, modelStartPresentation } from "./home";
import { initializeEvolutionSummary } from "./evolution-summary";
import { initializeChat } from "./chat";
import { initializeAppAd } from "./app-ad";
import { reportAppStart } from "./app-starts";
import { initializeModelQuiz } from "./model-quiz";
import { initializeEvolutionShare } from "./evolution-share";
export {};
localizeDocument(document);
initializeAppAd();
document.addEventListener("contextmenu", event => event.preventDefault());
document.addEventListener("keydown", event => {
  if (event.key === "F5" || (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "r") event.preventDefault();
});
type Settings = { automatic?: { learning_enabled?: boolean }; model_path: string; storage_root: string; local_thinking: string; native_inference_precision: "bf16" | "mixed-q4-g32" | "mixed-q4-g16" | "mixed-q8"; max_new_tokens: number; context_tokens: number; context_mode?: "auto" | "manual"; temperature: number; teacher: { teaching_modalities?: string[]; answer_mode?: "local_first" | "teacher_first" | "auto"; name?: string; base_url: string; model: string; thinking: string; send_reasoning_effort: boolean; session_call_limit: number; key_configured: boolean } };
type Agreement = { version: string; accepted: boolean; text: string };
type TrainingBackup = { id: string; created_at?: string; kind?: "learn" | "calibrate"; active_expert?: string | null; valid: boolean };
type EvolutionCapacity = { current_parameters: number | null; after_parameters: number | null; additional_weight_bytes: number | null; will_grow: boolean; eligible: boolean };
type Evolution = { media_materials?: { kind: string; lessons: number; samples: number; candidates: number; ready: boolean }[]; capacity?: EvolutionCapacity; prepared_count: number; verified_count: number; ready: boolean; blocked_reason: string | null; core?: { version?: string; counts?: Record<string, number> } };
type Automatic = { decision_learned: number; decision_pending: number; knowledge_learned: number; knowledge_pending: number; fastbrain_updates: number; learning_count: number; evolution_count: number; state: string; reason: string; experience: { current: number; target: number; continuing?: boolean }; evolution_blocked_reason?: string; mode: "manual" | "auto"; scheduled_time: string; pending_signature: string | null; reminder_due: boolean; last_evolution_timing?: Record<string, unknown> | null };
type RankingScores = { benchmark?: number | null; generalization?: number | null; benchmark_delta?: number | null; generalization_delta?: number | null; suite_id?: string | null; elapsed_seconds?: number | null };
type BenchmarkTask = { state: "idle" | "queued" | "running" | "complete" | "error"; completed?: number; total?: number; phase?: string; error?: string; result?: RankingScores | null };
type GrowthSummary = { receipt_id?: string; domain: string; baseline_score?: number | null; current_score?: number | null; net_delta?: number | null; gap_recovery?: number | null; retention?: number | null; question_count?: number | null; eligible?: boolean; retention_check?: { retention?: number | null } | null };
type GrowthTask = { state: string; phase?: string; completed?: number; total?: number; elapsed_seconds?: number; budget_seconds?: number; error?: string; message?: string; result?: GrowthSummary | null; eligible_skills?: string[] };
type RankingProfile = RankingScores & { growth?: GrowthSummary | null; growth_profile?: { domain: string; growth: GrowthSummary | null }[]; token_total?: number | null; username: string; country_code: string | null; evolution_count: number | null; count_error: string | null; published_profile: { username: string; evolution_count: number; country_code: string | null } | null };
type Ranking = { profile: RankingProfile; entries: (RankingScores & { growth?: GrowthSummary | null; token_total?: number | null; username: string; evolution_count: number; rank: number; country_code?: string | null })[]; page: number; page_size: number; total: number; updated_at: string | null; error: string | null };
type Usage = { prompt_tokens: number | null; completion_tokens: number | null; total_tokens: number | null; calls: number; estimated_calls: number; unknown_calls?: number };
type Status = {
  runtime: { native_process?: { physical_footprint_bytes?: number } | null; context?: { mode: string; current_limit: number | null; configured_limit: number; model_limit?: number | null; estimated: boolean }; model?: string; state: string; error?: string; device: string; base_parameters: number; language_parameters?: number; language_layers?: number; routing_layers?: number; core_training_layer?: number | null; expert_parameters: number; active_expert?: string; rss_bytes: number; metal_bytes: number; router: { calibrated?: boolean; validation_accuracy?: number; supported_actions?: string[]; gates_expert?: boolean }; job?: { kind?: string; state: string; phase?: string; step?: number; steps?: number; error?: string; result?: { promoted?: boolean; continuation_reason?: string; rejection_reason?: string; router_result?: { calibrated?: boolean; validation_accuracy?: number; supported_actions?: string[]; gates_expert?: boolean } } } };
  api_base: string; samples: { training: number; validation: number };
  usage?: { local: Usage; teacher: Usage }; teacher_materials?: { total: number; pending: number; eligible: number };
  automatic?: Automatic; benchmark?: BenchmarkTask; growth?: GrowthTask;
};
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => $<HTMLInputElement>(id);
const select = (id: string) => $<HTMLSelectElement>(id);
const button = (id: string) => $<HTMLButtonElement>(id);
const text = (id: string, value: string) => { $(id).textContent = value; };
const count = (value?: number | null) => typeof value === "number" ? value.toLocaleString(locale()) : "—";
let status: Status | undefined;
let teacherFirst = false;
let teacherConfigured = false;
let navigationVersion = 0;
let selectedPage = "home";
let modelStarting: boolean | "stopping" = false;
let startupReadyRequested = false;
let bundledStartupBusy = false, bundledStartupAttempted = false;
function chatReady() {
  const autoTraining = status?.automatic?.state === "running" && ["router", "learning", "evolution"].includes(status?.runtime.job?.kind || "");
  return !backupRestoreBusy && !!(teacherFirst || autoTraining || status?.runtime.state === "evaluating" || status?.runtime.state === "ready" && !["queued", "running"].includes(status?.runtime.job?.state || ""));
}
const chat = initializeChat({ api, failure, refresh: pollStatus, ready: chatReady, teacherFirst: () => teacherFirst });
let evolution: Evolution | undefined;
const evolutionSummary = initializeEvolutionSummary(api);
const evolutionShare = initializeEvolutionShare({ toast, failure });
let evolutionGraphKey = "";
let lastJobState = "";
let toastTimer: ReturnType<typeof setTimeout>;
let modelTools: ReturnType<typeof initializeModelTools> | undefined;
let home: ReturnType<typeof initializeHome> | undefined;
let modelQuiz: ReturnType<typeof initializeModelQuiz> | undefined;
let agreement: Agreement | undefined;
let backupRestoreBusy = false;
let rankingBusy = false;
let rankingLoaded = false;
let rankingCountryEdited = false;
let rankingPage = 1;
let rankingTotal = 0;
let rankingSort = "growth";
let rankingDomain = "mathematics";
let growthState: GrowthTask | undefined;
let benchmarkState: BenchmarkTask | undefined;
let benchmarkRefreshPending = false;
let rankingCountError: string | null = t("正在统计模型升级次数");
let settingsEdited = false;
for (const id of ["settings-form", "teacher-form"]) $(id).addEventListener("input", () => { settingsEdited = true; });

async function api<T = Record<string, unknown>>(path: string, data?: unknown, method = data === undefined ? "GET" : "POST"): Promise<T> {
  try {
    const response = await fetch(path, { method, credentials: "same-origin", headers: data === undefined ? undefined : { "Content-Type": "application/json" }, body: data === undefined ? undefined : JSON.stringify(data) });
    if (!response.ok) {
      let message: string | undefined;
      try { const body = await response.json(); message = typeof body.error === "object" ? body.error?.message : body.error || body.detail || body.message; } catch { /* Show the status if the server returned no JSON. */ }
      throw new Error(typeof message === "string" ? message : response.status === 401 ? t("请重新打开鲁班零号应用") : t`请求失败（${response.status}）`);
    }
    const body = await response.text();
    return (body ? JSON.parse(body) : {}) as T;
  } catch (error) {
    if (error instanceof TypeError) throw new Error(t("无法连接本地应用服务，请确认鲁班零号已启动；若正在更新或训练，请完成后重试。"));
    throw error;
  }
}
function failure(error: unknown) { text("error-text", error instanceof Error ? t(error.message) : t("操作失败，请重试")); $("error-banner").hidden = false; }
function isExpectedStartupUnavailable(error: unknown) {
  if (!(error instanceof Error)) return false;
  return [
    t("模型计算服务尚未就绪，请稍后重试"),
    t("模型状态尚未就绪，请稍后重试"),
    t("模型数据范围尚未就绪，请稍后重试"),
  ].includes(error.message);
}
function toast(message: string) { clearTimeout(toastTimer); text("toast", message); $("toast").hidden = false; toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500); }
function handle(action: () => Promise<unknown>) { void action().catch(failure); }
async function operation(target: HTMLButtonElement, action: () => Promise<unknown>, success?: string) {
  target.disabled = true;
  try { await action(); if (success) toast(success); }
  catch (error) { failure(error); }
  finally { target.disabled = false; if (status) renderStatus(status); }
}
function concealKey() { input("access-key").value = ""; input("access-key").type = "password"; text("reveal-key", t("显示")); button("copy-key").disabled = true; }
async function copy(value: string) { if (!navigator.clipboard) throw new Error(t("当前窗口不支持自动复制，请选中文字复制")); await navigator.clipboard.writeText(value); toast(t("已复制")); }
function showStartupReady(apiBase: string) {
  const dialog = $<HTMLDialogElement>("startup-ready-dialog");
  if (!dialog || dialog.open) return;
  input("startup-api").value = apiBase || "http://127.0.0.1:8765/v1";
  input("startup-key").value = "";
  input("startup-model").value = "luban0";
  text("startup-ready-feedback", "");
  dialog.showModal();
}
function bytes(value: number) { return value ? `${(value / 1024 ** 3).toFixed(2)} GB` : "—"; }
function renderSelectedPage() {
  const needsModel = ["learning", "evolution", "chat"].includes(selectedPage);
  const modelRequired = needsModel && (!status || status.runtime.state === "stopped");
  const teacherRequired = ["learning", "evolution"].includes(selectedPage) && !teacherConfigured;
  const visiblePage = modelRequired ? "model-required" : teacherRequired ? "teacher-required" : selectedPage;
  const entering = $(`view-${visiblePage}`).hidden;
  text("model-required-message", status ? t("模型还没启动") : t("正在读取模型状态…"));
  for (const page of document.querySelectorAll<HTMLElement>(".page")) page.hidden = page.id !== `view-${visiblePage}`;
  if (entering && visiblePage === "learning") { handle(loadVersions); handle(loadTrainingBackups); }
  if (entering && visiblePage === "evolution") handle(loadEvolution);
}
function renderStatus(next: Status) {
  const previousError = status?.runtime.error;
  status = next;
  const runtime = next.runtime;
  evolutionShare.observeRuntime(runtime);
  if ($("error-text").textContent === t("模型计算服务尚未就绪，请稍后重试")) $("error-banner").hidden = true;
  const job = runtime.job;
  const working = job?.state === "queued" || job?.state === "running";
  const start = modelStartPresentation(working ? job?.kind === "evolution" ? "evolving" : "learning" : runtime.state, backupRestoreBusy || working, modelStarting);
  text("runtime-state", start.detail);
  button("reload-model").dataset.state = start.state;
  const ready = runtime.state === "ready" && !working && !backupRestoreBusy;
  if (ready && startupReadyRequested) {
    startupReadyRequested = false;
    showStartupReady(next.api_base);
  }
  if (runtime.state === "stopped" && !modelStarting) startupReadyRequested = false;
  chat.renderControls();
  modelQuiz?.renderControls();
  button("reload-model").disabled = start.disabled;
  button("reload-model").setAttribute("aria-busy", String(start.state === "loading" || start.state === "stopping"));
  text("home-start-action", start.label);
  const modelAction = [start.label, start.detail].filter(Boolean).join("，");
  button("reload-model").setAttribute("aria-label", modelAction);
  button("reload-model").title = modelAction;
  for (const restore of document.querySelectorAll<HTMLButtonElement>(".backup-restore")) restore.disabled = !ready || restore.dataset.valid !== "true";
  for (const restore of document.querySelectorAll<HTMLButtonElement>(".version-restore")) restore.disabled = !ready || restore.dataset.version === runtime.active_expert;
  input("api-base").value = next.api_base || "http://127.0.0.1:8765/v1";
  const stopped = runtime.state === "stopped";
  text("device-summary", stopped ? t("模型未启动") : `${runtime.device === "mps" ? "Apple Metal" : runtime.device.toUpperCase()} · RSS ${bytes(runtime.rss_bytes)}`);
  text("memory-summary", stopped ? t("模型未启动；启动后显示实时内存") : t`主进程 RSS：${bytes(runtime.rss_bytes)} · 主进程 Metal：${bytes(runtime.metal_bytes)}${runtime.native_process?.physical_footprint_bytes ? t(" · 推理进程：") + bytes(runtime.native_process.physical_footprint_bytes) : ""} · 常驻新增参数：${count(runtime.expert_parameters)}。各内存口径不能直接相加。`);
  for (const source of ["local", "teacher"] as const) {
    const usage = next.usage?.[source];
    text(`${source}-prompt`, count(usage?.prompt_tokens)); text(`${source}-completion`, count(usage?.completion_tokens)); text(`${source}-total`, count(usage?.total_tokens));
    text(`${source}-calls`, t`${count(usage?.calls)} 次调用`);
    text(`${source}-estimated`, usage?.unknown_calls ? t`${count(usage.unknown_calls)} 次调用未提供词元统计` : usage?.estimated_calls ? t`其中 ${count(usage.estimated_calls)} 次调用使用估算词元数` : "");
    $(`${source}-estimated`).hidden = !usage?.estimated_calls && !usage?.unknown_calls;
  }
  const localTokens = next.usage?.local?.total_tokens, teacherTokens = next.usage?.teacher?.total_tokens;
  const totalTokens = typeof localTokens === "number" && typeof teacherTokens === "number" ? localTokens + teacherTokens : NaN;
  text("evolution-token-total", Number.isSafeInteger(totalTokens) && totalTokens >= 0 ? count(totalTokens) : "—");
  text("evolution-usage-label", next.usage?.local && next.usage?.teacher && (next.usage.local.estimated_calls || next.usage.teacher.estimated_calls) ? t("累计使用词元（含估算）") : t("累计使用词元"));
  const context = runtime.context;
  text("context-summary", context?.current_limit === 0 ? t("当前内存不足，暂无法分配上下文；容量设置已保留。") : context?.current_limit != null ? t`${context.mode === "auto" ? t("自动档") : t("手动档")} · 当前${(context.current_limit / 1024).toFixed(1)}K · 模型上限${((context.model_limit ?? context.configured_limit) / 1024).toFixed(0)}K${context.mode === "auto" ? t("（按可用内存保守估算）") : ""}。输出按实际可用容量的 1/4 自动预留（受执行器限制）；上下文达到 75% 时自动压缩历史。` : t("模型加载后按实际可用上下文的 1/4 自动预留输出（受执行器限制）；75% 时压缩历史。"));
  home?.renderRuntime(next.runtime, next.api_base);
  renderAutomatic();
  renderLearningControls();
  $("job").hidden = !job;
  if (job) {
    const router = job.kind === "router" || !!job.result?.router_result;
    text("job-title", job.state === "waiting_material" ? t("等待新训练材料") : job.state === "paused" ? t("学习已暂停") : working ? router ? t("后台更新能力判断") : job.kind === "evolution" ? t("正在进化并检查效果") : t("后台学习并检查效果") : job.state === "error" ? t("任务未完成") : router ? runtime.router.calibrated ? t("能力判断通过检查") : t("能力判断未通过检查") : job.kind === "media_learning" && job.phase === "awaiting_quality" ? t("本轮训练完成，候选已保存；质量与旧能力检查尚未完成，未采纳。") : job.result?.promoted ? t("检查通过，已保留新学习结果") : t("未保留候选学习结果"));
    const maximum = Number.isFinite(job.steps) && job.steps! > 0 ? job.steps! : 0;
    const step = Number.isFinite(job.step) ? Math.max(0, job.step!) : 0;
    text("job-progress", working && maximum ? t`${step} / ${maximum} 步` : "");
    const bar = $<HTMLProgressElement>("job-bar");
    bar.max = maximum || 1; bar.value = Math.min(step, maximum || 1); bar.hidden = !working;
    if (working && !maximum) bar.removeAttribute("value");
    text("job-detail", job.error || (job.state === "waiting_material" && job.result?.continuation_reason ? t(job.result.continuation_reason) : "") || (job.state === "paused" ? t("已暂停，等待继续。") : working ? t("只有通过检查的结果才会保留。") : router ? runtime.router.calibrated ? t("留出的路由测试达到门槛，已启用新的能力判断；效果限于本次测试材料。") : t("未通过检查，继续使用本地核心。") : job.result?.promoted ? t("独立检查改善且原有能力保持在要求内，已保留新学习结果。") : t("检查未达到保留条件，已恢复原有学习结果。")));
    if (job.state !== lastJobState) handle(loadVersions);
    if (["queued", "running"].includes(lastJobState) && !working) handle(loadTrainingBackups);
    lastJobState = job.state;
  }
  if (runtime.error) failure(new Error(runtime.error));
  else if (runtime.state === "stopped" && $("error-text").textContent === t("模型状态尚未就绪，请稍后重试")) $("error-banner").hidden = true;
  else if (previousError && $("error-text").textContent === previousError) $("error-banner").hidden = true;
  renderEvolution();
  renderBenchmark(next.benchmark);
  renderGrowth(next.growth);
  renderSelectedPage();
}
async function pollStatus() {
  try {
    renderStatus(await api<Status>("/api/status"));
  } catch (error) {
    // The compute service can need a moment to bind while the app window is opening.
    // A stopped model is an intentional state, so do not present that startup race as an error.
    if ((!status || ["loading", "stopped"].includes(status.runtime.state)) && isExpectedStartupUnavailable(error)) {
      $("error-banner").hidden = true;
      return;
    }
    throw error;
  }
  if (!$("view-evolution").hidden && !["queued", "running"].includes(benchmarkState?.state || "") && status?.runtime.state !== "evaluating") await loadEvolution();
  if (benchmarkRefreshPending && !$("view-leaderboard").hidden && !rankingBusy) await rankingOperation(() => loadRanking(1));
  await modelTools?.refresh();
  await prepareBundledStartup();
  if (["ready", "generating", "learning", "evolving", "evaluating"].includes(status?.runtime.state || "")) {
    await home?.refresh().catch(failure);
  }
}
async function prepareBundledStartup() {
  if (bundledStartupBusy || bundledStartupAttempted || !agreement?.accepted || !status || modelStarting || backupRestoreBusy) return;
  if (["ready", "generating", "learning", "evolving", "evaluating"].includes(status.runtime.state)) { bundledStartupAttempted = true; return; }
  if (status.runtime.state !== "stopped" || ["queued", "running"].includes(status.runtime.job?.state || "")) return;
  bundledStartupBusy = true;
  try {
    if (await modelTools?.prepareBundledModels() && status.runtime.state === "stopped" && !modelStarting) {
      bundledStartupAttempted = true;
      await changeModelState(false, false);
    }
  } finally { bundledStartupBusy = false; }
}
function applySettings(value: Settings, preserveDraft = false) {
  if (!automaticLearningSaveBusy) {
    input("learning-automatic").checked = value.automatic?.learning_enabled !== false;
    input("learning-automatic").disabled = false;
  }
  const teacher = value.teacher;
  teacherConfigured = !!(teacher.key_configured && teacher.base_url.trim() && teacher.model.trim() && (teacher.teaching_modalities ?? ["text"]).length);
  teacherFirst = (teacher.teaching_modalities ?? ["text"]).includes("text") && teacher.answer_mode === "teacher_first" && teacher.key_configured && !!teacher.base_url && !!teacher.model;
  chat.renderControls();
  if (preserveDraft && settingsEdited) return;
  input("model-path").value = value.model_path; input("storage-root").value = value.storage_root; select("local-thinking").value = value.local_thinking;
  select("chat-thinking").value = value.local_thinking;
  select("context-tokens").value = value.context_mode === "manual" ? String(value.context_tokens) : "auto"; input("temperature").value = String(value.temperature);
  settingsEdited = false;
  const answerMode = ["local_first", "teacher_first", "auto"].includes(String(teacher.answer_mode)) ? String(teacher.answer_mode) : "local_first";
  for (const mode of ["local_first", "teacher_first", "auto"]) input(`teacher-answer-${mode}`).checked = mode === answerMode;
  for (const kind of ["text", "image", "video", "audio"]) input(`teacher-teach-${kind}`).checked = (teacher.teaching_modalities ?? ["text"]).includes(kind);
  input("teacher-name").value = teacher.name || "";
  text("home-teacher-name", teacher.name ? t`API 教师（${teacher.name}）` : t("API 教师"));
  input("teacher-base").value = teacher.base_url; input("teacher-model").value = teacher.model; select("teacher-thinking").value = teacher.thinking;
  input("teacher-send-reasoning").checked = teacher.send_reasoning_effort; select("teacher-limit").value = ["0", "50", "100", "200", "300", "500"].includes(String(teacher.session_call_limit)) ? String(teacher.session_call_limit) : "0";
  input("teacher-key").value = ""; input("teacher-key").type = "password"; text("toggle-teacher-key", t("显示"));
  input("teacher-key").placeholder = teacher.key_configured ? t("密钥已保存；留空保留") : t("输入 API 密钥"); text("teacher-key-state", teacher.key_configured ? t("密钥已保存") : t("未配置密钥"));
}
async function refreshSavedSettings() {
  applySettings(await api<Settings>("/api/settings"), true);
}
function teacherPayload() {
  const answer_mode = ["local_first", "teacher_first", "auto"].find(mode => input(`teacher-answer-${mode}`).checked) || "local_first";
  const teacher: Record<string, unknown> = { teaching_modalities: ["text", "image", "video", "audio"].filter(kind => input(`teacher-teach-${kind}`).checked), answer_mode, name: input("teacher-name").value.trim(), base_url: input("teacher-base").value.trim(), model: input("teacher-model").value.trim(), thinking: select("teacher-thinking").value, send_reasoning_effort: input("teacher-send-reasoning").checked, session_call_limit: Number(select("teacher-limit").value) };
  if (input("teacher-key").value) teacher.api_key = input("teacher-key").value;
  return teacher;
}
async function saveTeacher() {
  applySettings(await api<Settings>("/api/settings", { teacher: teacherPayload() }, "PATCH"));
}
async function testTeacher() {
  const target = button("test-teacher"), result = $("teacher-test-result");
  target.disabled = true; result.hidden = false; result.setAttribute("role", "status"); result.textContent = t("正在测试当前表单中的教师配置…");
  try {
    const response = await api<{ model: string; reply: string; elapsed_seconds: number }>("/api/teacher/test", { teacher: teacherPayload() });
    result.textContent = t`测试成功 · ${response.model} · ${response.elapsed_seconds.toFixed(2)} 秒 · 回复：${response.reply || t("（空回复）")}。配置尚未保存。`;
  } catch (error) { result.setAttribute("role", "alert"); result.textContent = error instanceof Error ? error.message : t("教师测试失败，请重试"); }
  finally { target.disabled = false; }
}
async function loadVersions() {
  const { versions, last_attempt } = await api<{ versions: { id: string; parameters: number; created_at?: string }[];
    last_attempt?: { started_at: string; state: string } | null }>("/api/experts");
  const latest = Date.parse(last_attempt?.started_at || "");
  text("last-learning-time", Number.isFinite(latest) ? new Date(latest).toLocaleString(locale()) : "—");
  const states: Record<string, string> = { queued: t("已排队"), running: t("进行中"), paused: t("已暂停"),
    committing: t("正在保存结果"), committed: t("已保留学习结果"), rejected: t("未通过检查，未更新模型"),
    error: t("学习未完成，未更新模型") };
  text("last-learning-state", last_attempt ? states[last_attempt.state] || t("状态待确认") : "");
  $("last-learning-state").hidden = !last_attempt;
  const list = $("versions"); list.replaceChildren();
  if (!versions.length) { const note = document.createElement("p"); note.className = "muted"; note.textContent = t("还没有保存的学习记录。"); list.append(note); }
  for (const version of versions) {
    const row = document.createElement("div"); row.className = "version";
    const date = new Date(version.created_at || "");
    const time = Number.isNaN(date.getTime()) ? t("时间未知") : date.toLocaleString(locale());
    const name = document.createElement("span"); name.textContent = t`${version.id} · ${count(version.parameters)} 参数 · ${time}`;
    const restore = document.createElement("button"); restore.className = "button version-restore"; restore.dataset.version = version.id; restore.textContent = status?.runtime.active_expert === version.id ? t("当前版本") : t("恢复此版本");
    restore.disabled = backupRestoreBusy || status?.runtime.active_expert === version.id || status?.runtime.state !== "ready" || ["queued", "running"].includes(status?.runtime.job?.state || "");
    restore.addEventListener("click", () => operation(restore, async () => { await api("/api/expert/rollback", { id: version.id }); await pollStatus(); await loadVersions(); }, t("已恢复这次学习记录")));
    row.append(name, restore); list.append(row);
  }
  if (status) renderStatus(status);
}
async function loadTrainingBackups() {
  const { backups } = await api<{ backups: TrainingBackup[] }>("/api/training/backups");
  const list = $("training-backups"); list.replaceChildren();
  if (!backups.length) { const note = document.createElement("p"); note.className = "muted"; note.textContent = t("尚无训练前备份；下一次学习前会自动保存。"); list.append(note); }
  for (const backup of backups.slice(0, 3)) {
    const row = document.createElement("div"); row.className = "version";
    const date = new Date(backup.created_at || "");
    const time = Number.isNaN(date.getTime()) ? t("时间未知") : date.toLocaleString(locale());
    const label = backup.kind === "calibrate" ? t("能力判断更新前") : backup.kind === "learn" ? t("模型训练前") : t("训练前备份");
    const name = document.createElement("span"); name.textContent = `${time} · ${label}${"active_expert" in backup ? backup.active_expert ? t(" · 含已学分支") : t(" · 未加载已学分支") : ""} · ${backup.valid ? t("可恢复") : t("备份不完整，无法恢复")}`;
    const restore = document.createElement("button"); restore.className = "button backup-restore"; restore.dataset.valid = String(backup.valid); restore.disabled = true; restore.textContent = t("恢复此状态"); restore.setAttribute("aria-label", t`恢复 ${time} ${label}的状态`);
    restore.addEventListener("click", () => operation(restore, async () => {
      backupRestoreBusy = true; if (status) renderStatus(status);
      try { await api("/api/training/backups/restore", { id: backup.id }); await pollStatus(); await loadVersions(); await loadTrainingBackups(); }
      finally { backupRestoreBusy = false; }
    }, t("已恢复训练前状态")));
    row.append(name, restore); list.append(row);
  }
  if (status) renderStatus(status);
}
async function loadAgreement(showAtStartup = false) {
  agreement = await api<Agreement>("/api/agreement");
  text("agreement-version", t`使用协议 · ${agreement.version}`);
  text("agreement-text", agreement.text);
  text("agreement-state", agreement.accepted ? t`已同意 ${agreement.version} 版使用协议` : t("尚未同意使用协议"));
  button("agreement-accept").hidden = agreement.accepted;
  button("agreement-close").hidden = !agreement.accepted;
  if (showAtStartup && !agreement.accepted) $<HTMLDialogElement>("agreement-dialog").showModal();
}
async function acceptAgreement() {
  if (!agreement || agreement.accepted) return;
  const target = button("agreement-accept"); target.disabled = true; $("agreement-error").hidden = true;
  try {
    await api("/api/agreement/accept", {});
    agreement.accepted = true; text("agreement-state", t`已同意 ${agreement.version} 版使用协议`);
    target.hidden = true; button("agreement-close").hidden = false;
    $<HTMLDialogElement>("agreement-dialog").close();
  } catch (error) { text("agreement-error", error instanceof Error ? error.message : t("保存协议同意记录失败，请重试")); $("agreement-error").hidden = false; }
  finally { target.disabled = false; }
}
async function loadEvolution() {
  if (["queued", "running"].includes(benchmarkState?.state || "") || status?.runtime.state === "evaluating") return;
  evolution = await api<Evolution>("/api/evolution/status");
  renderEvolution();
  await evolutionSummary.refresh();
}
function automaticCount(value: number | undefined) {
  return Number.isSafeInteger(value) && value! >= 0 ? count(value) : "—";
}
function renderAutomatic() {
  const automatic = status?.automatic;
  for (const [id, key] of [["fastbrain-updates", "fastbrain_updates"], ["learning-count", "learning_count"], ["evolution-total", "evolution_count"]] as const)
    text(id, automaticCount(automatic?.[key]));
  const states: Record<string, string> = { queued: t("学习已排队"), running: t("后台正在学习"), paused: t("学习已暂停"), error: t("学习未完成"), waiting_validation: t("学习正在等待验证"), waiting_materials: t("等待已确认的训练和独立验证资料。") };
  const learningState = $("learning-state");
  const materials = /^等待足够的已核验学习资料：训练 (\d+) \/ (\d+) 条，独立验证 (\d+) \/ (\d+) 条。$/.exec(automatic?.reason || "");
  const reason = automatic?.reason === LEARNING_SOURCE_RECHECK_NOTICE ? t("学习暂不可用：引擎更新后需要重新验证。")
    : materials ? t`等待足够的已核验学习资料：训练 ${materials[1]} / ${materials[2]} 条，独立验证 ${materials[3]} / ${materials[4]} 条。`
    : automatic?.reason ? t(automatic.reason) : "";
  learningState.hidden = !automatic || !states[automatic.state];
  text("learning-state", automatic && states[automatic.state] ? reason || states[automatic.state] : "");
  const experience = automatic?.experience;
  const known = !!experience && Number.isSafeInteger(experience.current) && experience.current >= 0
    && Number.isSafeInteger(experience.target) && experience.target > 0;
  text("evolution-experience-label", known ? t`合格新知识 ${count(experience!.current)} / ${count(experience!.target)} 条` : t("正在统计合格新知识…"));
  const bar = $<HTMLProgressElement>("evolution-experience");
  bar.hidden = !known; bar.max = known ? experience!.target : 1; bar.value = known ? Math.min(experience!.current, experience!.target) : 0;
}
let learningStartBusy = false;
let automaticLearningSaveBusy = false;
function renderLearningControls() {
  button("learning-start").disabled = learningStartBusy || ["queued", "running"].includes(status?.runtime.job?.state || "")
    || ["generating", "learning", "evolving", "training", "evaluating"].includes(status?.runtime.state || "")
    || status?.automatic?.state === "running";
}
async function saveAutomaticLearning() {
  if (automaticLearningSaveBusy) return;
  const target = input("learning-automatic"), enabled = target.checked;
  automaticLearningSaveBusy = true; target.disabled = true;
  try {
    const saved = await api<Settings>("/api/settings", { automatic: { learning_enabled: enabled } }, "PATCH");
    target.checked = saved.automatic?.learning_enabled !== false;
  } catch (error) { target.checked = !enabled; throw error; }
  finally { automaticLearningSaveBusy = false; target.disabled = false; }
}
async function startLearning() {
  renderLearningControls();
  if (button("learning-start").disabled) return;
  const automatic = status?.automatic, experience = automatic?.experience;
  if (status?.runtime.state !== "ready") { toast(t("请先启动模型，待就绪后开始学习。")); return; }
  if (!experience || !Number.isSafeInteger(experience.current) || experience.current < 0
    || !Number.isSafeInteger(experience.target) || experience.target <= 0) {
    toast(t("正在统计合格新知识，请稍后重试。")); return;
  }
  if (experience.current < experience.target || automatic.state === "waiting_materials" && /(?:资料|内容).*(?:不足|等待)|等待.*(?:资料|内容)/.test(automatic.reason || "")) {
    toast(t("资料不足")); return;
  }
  if (automatic.state === "waiting_validation") {
    toast(automatic.reason === LEARNING_SOURCE_RECHECK_NOTICE ? t("学习暂不可用：引擎更新后需要重新验证。")
      : automatic.reason ? t(automatic.reason) : t("学习正在等待验证")); return;
  }
  learningStartBusy = true; renderLearningControls();
  try { await api("/api/learn", {}); await pollStatus(); await loadVersions(); }
  catch (error) { failure(error); }
  finally { learningStartBusy = false; renderLearningControls(); }
}
let evolutionStartBusy = false;
function evolutionStartState() {
  const runtime = status?.runtime, automatic = status?.automatic;
  const busy = evolutionStartBusy || ["queued", "running"].includes(runtime?.job?.state || "")
    || ["training", "evaluating"].includes(runtime?.state || "");
  if (busy) return { disabled: true, reason: evolutionStartBusy ? t("正在开始进化…") : t("请等待当前任务完成后再进化。") };
  if (!runtime || runtime.state !== "ready") return { disabled: false, reason: runtime?.state === "stopped"
    ? t("请先启动模型，再开始进化。") : t("模型尚未就绪，请稍后再开始进化。") };
  const experience = automatic?.experience;
  if (!experience || !Number.isSafeInteger(experience.current) || experience.current < 0
    || !Number.isSafeInteger(experience.target) || experience.target <= 0)
    return { disabled: false, reason: t("正在统计合格新知识，请稍后重试。") };
  if (experience.current < experience.target)
    return { disabled: false, reason: t`还差 ${count(experience.target - experience.current)} 条合格新知识才能进化（已有 ${count(experience.current)} / ${count(experience.target)} 条）。` };
  const ready = (typeof automatic?.pending_signature === "string" && automatic.pending_signature.length > 0)
    || evolution?.ready === true && evolution.capacity?.eligible === true;
  const reason = automatic?.evolution_blocked_reason || evolution?.blocked_reason || automatic?.reason;
  return { disabled: false, reason: ready ? "" : reason === LEARNING_SOURCE_RECHECK_NOTICE
    ? t("学习暂不可用：引擎更新后需要重新验证。") : reason ? t(reason) : t("等待足够的已核验内容。") };
}
async function startEvolution() {
  const state = evolutionStartState();
  if (state.disabled) return;
  if (state.reason) { toast(state.reason); return; }
  const target = button("evolution-start");
  evolutionStartBusy = true; target.disabled = true; renderEvolution();
  try { await api("/api/evolution/start", {}); await pollStatus(); }
  catch (error) { failure(error); }
  finally { evolutionStartBusy = false; renderEvolution(); }
}
function renderEvolutionCapacity() {
  const capacity = evolution?.capacity;
  const parameters = (value?: number | null) => Number.isSafeInteger(value) && value! > 0 ? t`${count(value!)} 参数` : "—";
  text("evolution-current-parameters", parameters(capacity?.current_parameters));
  text("evolution-after-parameters", parameters(capacity?.after_parameters));
  const weightBytes = capacity?.additional_weight_bytes;
  text("evolution-additional-weights", Number.isSafeInteger(weightBytes) && weightBytes! >= 0 ? weightBytes === 0 ? "0 B" : `${(weightBytes! / 1024 ** 2).toFixed(1)} MB` : t("待测"));
  text("evolution-additional-memory", t("待测"));
  const state = evolutionStartState();
  button("evolution-start").disabled = state.disabled;
  text("evolution-start-hint", state.reason || t("通过独立验证后才会保留新结果。"));
  text("evolution-growth-state", capacity?.will_grow === false ? t("本次巩固更新既有参数 · 🔒 结构扩容暂未开放") : t("🔒 结构扩容暂未开放"));
}
function renderEvolution() {
  renderEvolutionCapacity();
  const runtime = status?.runtime;
  const layers = Number.isInteger(runtime?.language_layers) && runtime!.language_layers! > 0 ? runtime!.language_layers! : 0;
  if (evolutionGraphKey !== String(layers)) {
    evolutionGraphKey = String(layers);
    const stars = $("evolution-connections"), strands = $("evolution-layers");
    stars.replaceChildren(); strands.replaceChildren();
    const element = (tag: string, attributes: Record<string, string>) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
      return node;
    };
    for (let index = 0; index < 68; index++) {
      const angle = index * 2.39996, radius = 190 + (index * 31 % 154);
      stars.append(element("circle", { cx: String(400 + Math.cos(angle) * radius),
        cy: String(250 + Math.sin(angle) * radius * .65), r: index % 7 ? ".8" : "1.8",
        class: "energy-star", opacity: String(.15 + index % 5 * .12) }));
    }
    for (let index = 0; index < 24; index++) {
      const angle = index * Math.PI / 12, x = Math.cos(angle) * 136, y = Math.sin(angle) * 136;
      const group = element("g", { class: "energy-strand" });
      group.append(element("path", { d: `M${400 + x},${250 + y} C${400 - y * .95},${250 + x * .7} ${400 + y * .95},${250 - x * .7} ${400 - x},${250 - y}`,
        fill: "none", stroke: "url(#energy-thread)", "stroke-width": ".8", opacity: String(.22 + index % 4 * .12) }));
      strands.append(group);
    }
    text("evolution-graph-title", t("鲁班零号 · 进化能量核心"));
    text("evolution-graph-description", t`抽象能量、轨道和粒子表现进化状态；${layers ? t`当前模型共${layers}层` : t("模型结构尚未连接")}，光效不表示神经元活动或所有参数同时训练。`);
  }
  $("evolution-graph").removeAttribute("hidden");
  $("evolution-graph-wait").hidden = !!layers;
  text("evolution-graph-wait", runtime?.state === "stopped" ? t("模型未启动") : runtime?.state === "error" ? t("模型连接异常") : t("正在连接核心…"));
  text("evolution-layer-count", layers ? t("核心已连接") : t("核心未连接"));
  const job = runtime?.job, coreJob = job?.kind === "evolution";
  const working = !!job && ["queued", "running"].includes(job.state);
  const evolving = coreJob && working;
  const step = Number.isFinite(job?.step) ? Math.max(0, job!.step!) : 0;
  const maximum = Number.isFinite(job?.steps) && job!.steps! > 0 ? job!.steps! : 0;
  const progress = maximum ? Math.min(step / maximum, 1) : 0;
  const state = evolving ? job!.state : coreJob ? job!.state === "paused" ? "paused" : job!.state === "error" ? "error" : job!.result?.promoted ? "success" : "rejected" : "idle";
  text("evolution-energy-state", evolving ? t("正在进化") : state === "paused" ? t("进化已暂停") : state === "success" ? t("新的成长") : t("积蓄新的力量"));
  $("evolution-panel").dataset.state = state;
  $("evolution-panel").dataset.phase = coreJob ? job!.phase || "" : "";
  $("evolution-panel").setAttribute("aria-busy", String(!!evolving));
  $("evolution-panel").style.setProperty("--evolution-progress", String(progress));
  const bar = $<HTMLProgressElement>("evolution-progress");
  bar.hidden = !evolving; bar.max = maximum || 1; bar.value = Math.min(step, maximum || 1);
  if (evolving && !maximum) bar.removeAttribute("value");
  if (runtime) evolutionSummary.renderRuntime(runtime);
}
window.matchMedia("(max-width: 680px)").addEventListener("change", renderEvolution);
function scoreText(value?: number | null, delta?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const change = typeof delta === "number" && Number.isFinite(delta) ? ` · ${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp` : "";
  return value.toFixed(1) + change;
}
function percentText(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}%` : "—";
}
function deltaText(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(1)} pp` : "—";
}
const growthDomains: Record<string, string> = { mathematics: t("数学"), programming: t("编程"), data: t("数据处理") };
const capabilityDomains: Record<string, string> = {
  mathematics: t("数学"), programming: t("编程与计算机"), data: t("数据与分析"), language: t("语言与翻译"),
  writing: t("写作与表达"), science: t("自然科学"), medicine: t("医疗与健康"), law: t("法律"),
  finance: t("财经与金融"), education: t("教育与学习"), engineering: t("工程与技术"), business: t("商业与管理"),
  humanities: t("历史与人文"), arts: t("艺术与设计"), daily_life: t("生活与实用知识"), pets: t("宠物与动物"),
};
const evaluationBusy = () => [benchmarkState?.state, growthState?.state].some(state => state === "queued" || state === "running");
function renderRankingScores(scores: RankingScores) {
  text("leaderboard-benchmark", scoreText(scores.benchmark, scores.benchmark_delta));
  text("leaderboard-elapsed", typeof scores.elapsed_seconds === "number" && Number.isFinite(scores.elapsed_seconds)
    && scores.elapsed_seconds >= 0 ? t`${scores.elapsed_seconds.toFixed(1)} 秒` : "—");
}
function renderCapabilities(profile: RankingProfile) {
  const list = $("capability-profile"); list.replaceChildren();
  for (const [domain, label] of Object.entries(capabilityDomains)) {
    const item = profile.growth_profile?.find(item => item.domain === domain);
    const summary = item ? item.growth : (profile.growth?.domain === domain ? profile.growth : undefined);
    const card = document.createElement("section"); card.className = "capability-domain";
    const heading = document.createElement("h3"); heading.textContent = label; card.append(heading);
    const scores = document.createElement("p"); scores.className = "capability-scores";
    scores.textContent = summary ? `${percentText(summary.baseline_score)} → ${percentText(summary.current_score)}` : t("待验证");
    card.append(scores);
    const metrics = document.createElement("p"); metrics.className = "field-hint";
    metrics.textContent = summary ? t`缺口解决 ${percentText(summary.gap_recovery)} · 旧能力保持 ${percentText(summary.retention)}` : t("暂缺可独立验证的新题，尚不计分");
    card.append(metrics);
    const evidence = document.createElement("p"); evidence.className = "field-hint";
    evidence.textContent = summary ? t`本次成长 ${deltaText(summary.net_delta)} · 有效题数 ${Number.isSafeInteger(summary.question_count) && summary.question_count! > 0 ? count(summary.question_count!) : "—"}${summary.eligible === false ? t(" · 尚不计入排行") : ""}` : t("有效题数 —");
    card.append(evidence);
    if (summary?.retention_check) { const delayed = document.createElement("p"); delayed.className = "field-hint"; delayed.textContent = t`延迟保持 ${percentText(summary.retention_check.retention)}`; card.append(delayed); }
    list.append(card);
  }
}
function renderEvaluationControls() {
  const busy = evaluationBusy();
  button("leaderboard-publish").disabled = rankingBusy || busy || Boolean(rankingCountError);
  button("benchmark-start").disabled = rankingBusy || busy || Boolean(rankingCountError);
  text("leaderboard-publish", ["queued", "running"].includes(growthState?.state || "") ? t("正在评测…") : t("评测并上报"));
  input("leaderboard-username").disabled = rankingBusy || busy;
  select("leaderboard-country").disabled = rankingBusy || busy;
  $("leaderboard-form").setAttribute("aria-busy", String(rankingBusy || busy));
  button("benchmark-stop").disabled = rankingBusy;
  button("growth-stop").disabled = rankingBusy;
}
function renderBenchmark(task?: BenchmarkTask) {
  const previous = benchmarkState?.state;
  benchmarkState = task;
  const busy = task?.state === "queued" || task?.state === "running";
  const completedNow = task?.state === "complete" && (previous === "queued" || previous === "running");
  if (completedNow && task?.result) renderRankingScores(task.result);
  const completed = Number.isSafeInteger(task?.completed) && task!.completed! >= 0 ? task!.completed! : 0;
  const total = Number.isSafeInteger(task?.total) && task!.total! > 0 ? task!.total! : 0;
  const phases: Record<string, string> = { benchmark: t("正在测通用能力"), generalization: t("正在测通用能力变式"), publishing: t("正在上报通用成绩"), publish: t("正在上报通用成绩") };
  text("benchmark-state", task?.state === "queued" ? t("通用能力评测已排队") : task?.state === "running" ? `${phases[task.phase || ""] || t("正在测通用能力")}${total ? ` · ${completed} / ${total}` : ""}`
    : task?.state === "error" ? task.error || t("通用能力评测未完成，请重试") : task?.state === "complete" ? t("通用能力评测完成，已上报") : "");
  $("benchmark-state").setAttribute("role", task?.state === "error" ? "alert" : "status");
  const progress = $<HTMLProgressElement>("benchmark-progress");
  progress.hidden = !busy; progress.max = total || 1; progress.value = Math.min(completed, total || 1);
  if (busy && !total) progress.removeAttribute("value");
  $("benchmark-stop").hidden = !busy;
  renderEvaluationControls();
  if (completedNow) benchmarkRefreshPending = true;
}
function renderGrowth(task?: GrowthTask) {
  const previous = growthState?.state;
  const previousReceipt = growthState?.result?.receipt_id;
  growthState = task;
  const busy = task?.state === "queued" || task?.state === "running";
  const completed = Number.isSafeInteger(task?.completed) && task!.completed! >= 0 ? task!.completed! : 0;
  const total = Number.isSafeInteger(task?.total) && task!.total! > 0 ? task!.total! : 0;
  const phases: Record<string, string> = { baseline: t("正在建立成长基线"), before: t("正在建立成长基线"), after: t("正在评测学习后的成长"), retention: t("正在核验延迟保持"), original_retest: t("正在核验旧能力保持"), publishing: t("正在上报成长"), publish: t("正在上报成长") };
  const phase = phases[task?.phase || ""] || task?.phase || t("正在评测个人成长");
  const terminalPhase = task?.phase && !Object.hasOwn(phases, task.phase) ? task.phase : "";
  const message = task?.state === "baseline" ? terminalPhase || t("已建立成长基线，学习后自动评测")
    : task?.state === "queued" ? t("成长评测已排队")
    : task?.state === "running" ? `${phase}${total ? ` · ${completed} / ${total}` : ""}`
    : task?.state === "error" ? task.error || terminalPhase || t("成长评测未完成，请重试")
    : task?.state === "complete" ? terminalPhase || (task.result?.eligible === false ? t("成长评测完成，证据不足，尚不计分") : t("成长评测完成，已上报"))
    : task?.state === "cancelled" ? terminalPhase || t("成长评测已取消，未上报本次结果")
    : task?.message || (task?.eligible_skills?.length === 0 ? t("暂缺可独立验证的新题，尚不计分") : t("学习前建立基线，学习后自动核验个人成长。"));
  text("growth-state", message);
  $("growth-state").setAttribute("role", task?.state === "error" ? "alert" : "status");
  const exam = $("growth-exam"); exam.hidden = !busy;
  exam.setAttribute("data-state", task?.state || "idle");
  text("growth-exam-phase", task?.state === "queued" ? t("等待考试开始") : phase);
  const clockText = (seconds?: number) => {
    if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "—";
    const whole = Math.floor(seconds); return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  };
  text("growth-time", t`已用 ${clockText(task?.elapsed_seconds)} · 上限 ${clockText(task?.budget_seconds)}`);
  const progress = $<HTMLProgressElement>("growth-progress");
  progress.hidden = !busy; progress.max = total || 1; progress.value = Math.min(completed, total || 1);
  if (busy && !total) progress.removeAttribute("value");
  $("growth-stop").hidden = !busy;
  renderEvaluationControls();
  if (["complete", "baseline"].includes(task?.state || "") && (previous !== task?.state || previousReceipt !== task?.result?.receipt_id)) benchmarkRefreshPending = true;
}
function renderRankingProfile(profile: RankingProfile, fillForm = false) {
  if (fillForm) {
    input("leaderboard-username").value = profile.username;
  }
  if (fillForm || !rankingCountryEdited && !select("leaderboard-country").value && profile.country_code) select("leaderboard-country").value = profile.country_code || "";
  rankingCountError = profile.count_error;
  text("leaderboard-count", profile.evolution_count === null ? t("暂不可用") : t`${count(profile.evolution_count)} 次`);
  text("leaderboard-tokens", Number.isSafeInteger(profile.token_total) && profile.token_total! >= 0 ? count(profile.token_total!) : "—");
  renderRankingScores(profile);
  renderCapabilities(profile);
  const published = profile.published_profile;
  text("leaderboard-profile-state", profile.count_error || (published ? t`已上报：${published.username} · ${count(published.evolution_count)} 次` : t("尚未上报")));
  $("leaderboard-profile-state").setAttribute("role", profile.count_error ? "alert" : "status");
  renderBenchmark(benchmarkState);
  renderGrowth(growthState);
}
function renderRankingPagination() {
  const pages = Math.max(1, Math.ceil(rankingTotal / 100));
  $("leaderboard-pagination").hidden = rankingTotal <= 100;
  text("leaderboard-page", `${rankingPage} / ${pages}`);
  button("leaderboard-previous").disabled = rankingBusy || rankingPage <= 1;
  button("leaderboard-next").disabled = rankingBusy || rankingPage >= pages;
}
select("leaderboard-country").addEventListener("change", () => { rankingCountryEdited = true; });
const regionNames = new Intl.DisplayNames(["zh-CN"], { type: "region" });
for (const code of [...COUNTRY_CODES].sort((a, b) => (regionNames.of(a) || a).localeCompare(regionNames.of(b) || b, "zh-CN"))) {
  const option = document.createElement("option"); option.value = code; option.textContent = countryLabel(code);
  select("leaderboard-country").append(option);
}
async function loadRanking(page = rankingPage, fillForm = !rankingLoaded) {
  if (benchmarkRefreshPending) { page = 1; benchmarkRefreshPending = false; }
  const ranking = await api<Ranking>(`/api/leaderboard?page=${page}&sort=${rankingSort}&domain=${rankingDomain}`);
  if (!ranking.error && page > Math.max(1, Math.ceil(ranking.total / 100))) return loadRanking(1, fillForm);
  renderRankingProfile(ranking.profile, fillForm); rankingLoaded = true;
  rankingPage = ranking.page; rankingTotal = ranking.total; renderRankingPagination();
  const rows = $("leaderboard-entries"); rows.replaceChildren();
  const entries = rankingSort === "growth" ? ranking.entries.filter(entry => entry.growth?.domain === rankingDomain && entry.growth.eligible) : ranking.entries;
  for (const entry of entries) {
    const row = document.createElement("tr");
    const growth = entry.growth?.domain === rankingDomain ? entry.growth : null;
    const growthValue = growth?.eligible && typeof growth.net_delta === "number" && Number.isFinite(growth.net_delta) ? `${deltaText(growth.net_delta)}${Number.isSafeInteger(growth.question_count) && growth.question_count! > 0 ? t` · ${count(growth.question_count!)} 题` : ""}` : "—";
    for (const value of [count(entry.rank), entry.username, countryLabel(entry.country_code), Number.isSafeInteger(entry.token_total) && entry.token_total! >= 0 ? count(entry.token_total!) : "—", count(entry.evolution_count), growthValue, growth?.eligible ? percentText(growth.retention) : "—", scoreText(entry.benchmark)]) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    rows.append(row);
  }
  if (!entries.length) { const row = document.createElement("tr"), cell = document.createElement("td"); cell.colSpan = 8; cell.textContent = ranking.error ? t("全球排行暂时无法读取") : t("还没有上榜用户"); row.append(cell); rows.append(row); }
  text("leaderboard-summary", ranking.error || t`${rankingSort === "growth" ? growthDomains[rankingDomain] + t("成长 · ") : ""}全球 ${count(ranking.total)} 位用户${ranking.total > ranking.entries.length ? t` · 当前页 ${count(ranking.entries.length)} 位` : ""}`);
  text("leaderboard-caption", t`${rankingSort === "growth" ? growthDomains[rankingDomain] + t("领域 · ") : ""}成长为百分点（pp） · 未验证显示 —`);
  $("leaderboard-summary").setAttribute("role", ranking.error ? "alert" : "status");
}
async function rankingOperation(action: () => Promise<void>, success?: string) {
  if (rankingBusy) return;
  rankingBusy = true;
  for (const id of ["leaderboard-publish", "leaderboard-refresh", "leaderboard-previous", "leaderboard-next", "benchmark-start", "benchmark-stop", "growth-stop"]) button(id).disabled = true;
  input("leaderboard-username").disabled = true;
  select("leaderboard-country").disabled = true;
  select("leaderboard-sort").disabled = true;
  select("leaderboard-domain").disabled = true;
  $("leaderboard-form").setAttribute("aria-busy", "true");
  const result = $("leaderboard-result"); result.hidden = true;
  try { await action(); if (success) { result.textContent = success; result.setAttribute("role", "status"); result.hidden = false; } }
  catch (error) { result.textContent = error instanceof Error ? error.message : t("排行操作失败，请重试"); result.setAttribute("role", "alert"); result.hidden = false; }
  finally {
    rankingBusy = false;
    for (const id of ["leaderboard-publish", "leaderboard-refresh", "leaderboard-previous", "leaderboard-next", "benchmark-start", "benchmark-stop", "growth-stop"]) button(id).disabled = false;
    input("leaderboard-username").disabled = false;
    select("leaderboard-country").disabled = false;
    select("leaderboard-sort").disabled = false;
    select("leaderboard-domain").disabled = false;
    renderRankingPagination();
    renderBenchmark(benchmarkState);
  }
}
async function saveRankingProfile() {
  const response = await api<{ profile: RankingProfile }>("/api/leaderboard/profile", { username: input("leaderboard-username").value, ...(rankingCountryEdited || select("leaderboard-country").value ? { country_code: select("leaderboard-country").value || null } : {}) }, "PUT");
  renderRankingProfile(response.profile, true); rankingLoaded = true;
}
$("leaderboard-form").addEventListener("submit", event => {
  event.preventDefault();
  if (rankingCountError || evaluationBusy() || !$<HTMLFormElement>("leaderboard-form").reportValidity()) return;
  void rankingOperation(async () => { await saveRankingProfile(); renderGrowth(await api<GrowthTask>("/api/growth/start", {})); });
});
$("benchmark-start").addEventListener("click", () => {
  if (rankingCountError || evaluationBusy() || !$<HTMLFormElement>("leaderboard-form").reportValidity()) return;
  void rankingOperation(async () => { await saveRankingProfile(); renderBenchmark(await api<BenchmarkTask>("/api/benchmark/start", {})); });
});
$("growth-stop").addEventListener("click", () => {
  if (!["queued", "running"].includes(growthState?.state || "")) return;
  void rankingOperation(async () => { await api("/api/growth/cancel", {}); await pollStatus(); });
});
$("benchmark-stop").addEventListener("click", () => {
  if (!["queued", "running"].includes(benchmarkState?.state || "")) return;
  void rankingOperation(async () => { await api("/api/benchmark/cancel", {}); await pollStatus(); });
});
$("leaderboard-sort").addEventListener("change", () => {
  const sort = select("leaderboard-sort").value;
  if (!["growth", "evolution_count", "benchmark"].includes(sort)) return;
  rankingSort = sort;
  $("leaderboard-domain-filter").hidden = sort !== "growth";
  void rankingOperation(() => loadRanking(1));
});
$("leaderboard-domain").addEventListener("change", () => {
  const domain = select("leaderboard-domain").value;
  if (!Object.hasOwn(growthDomains, domain)) return;
  rankingDomain = domain;
  void rankingOperation(() => loadRanking(1));
});
$("leaderboard-refresh").addEventListener("click", () => { void rankingOperation(() => loadRanking()); });
$("leaderboard-previous").addEventListener("click", () => { if (rankingPage > 1) void rankingOperation(() => loadRanking(rankingPage - 1)); });
$("leaderboard-next").addEventListener("click", () => { if (rankingPage < Math.ceil(rankingTotal / 100)) void rankingOperation(() => loadRanking(rankingPage + 1)); });
button("evolution-start").addEventListener("click", () => handle(startEvolution));
input("learning-automatic").addEventListener("change", () => handle(saveAutomaticLearning));
button("learning-start").addEventListener("click", () => handle(startLearning));
$("view-agreement").addEventListener("click", () => handle(async () => { await loadAgreement(); $("agreement-error").hidden = true; $<HTMLDialogElement>("agreement-dialog").showModal(); }));
$("agreement-accept").addEventListener("click", () => { void acceptAgreement(); });
$("agreement-close").addEventListener("click", () => { $<HTMLDialogElement>("agreement-dialog").close(); });
$("agreement-dialog").addEventListener("cancel", event => { if (!agreement?.accepted) event.preventDefault(); });
$("refresh-backups").addEventListener("click", () => handle(loadTrainingBackups));
const fragment = location.hash.slice(1);
history.replaceState(null, "", location.pathname + location.search);
const params = new URLSearchParams(fragment);
const launchToken = params.get("token") || params.get("launch_token") || (fragment && !fragment.includes("=") ? fragment : "");
$("dismiss-error").addEventListener("click", () => { $("error-banner").hidden = true; });
$("copy-base").addEventListener("click", () => handle(() => copy(input("api-base").value)));
$("reveal-key").addEventListener("click", () => operation(button("reveal-key"), async () => {
  if (input("access-key").value) { concealKey(); return; }
  input("access-key").value = (await api<{ key: string }>("/api/access-key")).key; input("access-key").type = "text"; text("reveal-key", t("隐藏")); button("copy-key").disabled = false;
}));
$("copy-key").addEventListener("click", () => handle(async () => { if (input("access-key").value) await copy(input("access-key").value); }));
$("rotate-key").addEventListener("click", () => operation(button("rotate-key"), async () => { await api("/api/access-key/rotate", {}); concealKey(); }, t("密钥已轮换，请更新调用方配置")));
$("toggle-teacher-key").addEventListener("click", () => { const key = input("teacher-key"); key.type = key.type === "password" ? "text" : "password"; text("toggle-teacher-key", key.type === "password" ? t("显示") : t("隐藏")); });
$("teacher-form").addEventListener("submit", event => { event.preventDefault(); void operation($<HTMLFormElement>("teacher-form").querySelector<HTMLButtonElement>('button[type="submit"]')!, saveTeacher, t("教师设置已保存")); });
$("test-teacher").addEventListener("click", () => { void testTeacher(); });
$("fetch-models").addEventListener("click", () => operation(button("fetch-models"), async () => {
  const { models } = await api<{ models: string[] }>("/api/teacher/models", { teacher: teacherPayload() });
  const list = $("teacher-models"); list.replaceChildren();
  for (const model of models) { const option = document.createElement("option"); option.value = model; list.append(option); }
  if (!models.length) throw new Error(t("教师未返回模型列表，可手动填写模型名称"));
  if (!input("teacher-model").value) input("teacher-model").value = models[0];
  toast(t`已获取 ${models.length} 个模型，选择后请保存`); input("teacher-model").focus();
}));
$("settings-form").addEventListener("submit", event => {
  event.preventDefault();
  void operation($<HTMLFormElement>("settings-form").querySelector<HTMLButtonElement>('button[type="submit"]')!, async () => {
    applySettings(await api<Settings>("/api/settings", { model_path: input("model-path").value.trim(), storage_root: input("storage-root").value.trim(), local_thinking: select("local-thinking").value, max_new_tokens: 0, context_mode: select("context-tokens").value === "auto" ? "auto" : "manual", context_tokens: select("context-tokens").value === "auto" ? 262144 : Number(select("context-tokens").value), temperature: Number(input("temperature").value) }, "PATCH"));
  }, t("运行设置已保存"));
});
for (const [field, feedback] of [["api", "API 地址已复制"], ["model", "模型名称已复制"], ["key", "访问密钥已复制"]]) {
  $(`copy-startup-${field}`).addEventListener("click", async () => {
    const target = button(`copy-startup-${field}`); target.disabled = true;
    text("startup-ready-feedback", "");
    try {
      const value = field === "key" ? (await api<{ key: string }>("/api/access-key")).key : input(`startup-${field}`).value;
      await copy(value);
      text("startup-ready-feedback", t(feedback));
    } catch (error) {
      text("startup-ready-feedback", error instanceof Error ? t(error.message) : t("操作失败，请重试"));
    } finally { target.disabled = false; }
  });
}
async function changeModelState(stopping: boolean, showReady = true) {
  if (modelStarting || button("reload-model").disabled) return;
  bundledStartupAttempted = true;
  if (!stopping) startupReadyRequested = showReady;
  modelStarting = stopping ? "stopping" : true;
  if (status) renderStatus(status);
  await operation(button("reload-model"), async () => {
    try {
      await api(stopping ? "/api/model/stop" : "/api/model/reload", {});
    } finally {
      modelStarting = false;
      await pollStatus();
    }
    if (!$("view-learning").hidden) await loadVersions();
  });
}
$("reload-model").addEventListener("click", () => {
  void changeModelState(status?.runtime.state === "ready");
});
document.addEventListener("toggle", event => {
  const tip = event.target;
  if (!(tip instanceof HTMLDetailsElement) || !tip.matches(".help-tip") || !tip.open) return;
  document.querySelectorAll<HTMLDetailsElement>(".help-tip[open]").forEach(other => { if (other !== tip) other.open = false; });
  const content = tip.querySelector<HTMLElement>(".help-content")!;
  content.style.left = "0px"; content.style.top = "calc(100% + 8px)"; content.style.bottom = "auto";
  const bounds = content.getBoundingClientRect();
  if (bounds.bottom > innerHeight - 16 && tip.getBoundingClientRect().top > bounds.height + 16) { content.style.top = "auto"; content.style.bottom = "calc(100% + 8px)"; }
  content.style.left = `${Math.max(16, Math.min(bounds.left, innerWidth - bounds.width - 16)) - tip.getBoundingClientRect().left}px`;
}, true);
document.addEventListener("click", event => {
  document.querySelectorAll<HTMLDetailsElement>(".help-tip[open]").forEach(tip => { if (!tip.contains(event.target as Node)) tip.open = false; });
});
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  document.querySelectorAll<HTMLDetailsElement>(".help-tip[open]").forEach(tip => { tip.open = false; tip.querySelector("summary")?.focus(); });
});
$("refresh-versions").addEventListener("click", () => handle(loadVersions));
document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach(tab => tab.addEventListener("click", () => {
  const version = ++navigationVersion;
  handle(async () => {
  const selected = tab.dataset.tab!;
  if (selected === "settings" || status && status.runtime.state !== "stopped" && ["learning", "evolution"].includes(selected)) await refreshSavedSettings();
  if (version !== navigationVersion) return;
  selectedPage = selected;
  renderSelectedPage();
  for (const item of document.querySelectorAll<HTMLButtonElement>("[data-tab]")) { const active = item.dataset.tab === selected && item.closest(".tabs") !== null; item.classList.toggle("active", active); if (active) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current"); }
  if (["home-teacher", "configure-teacher"].includes(tab.id)) document.querySelector<HTMLButtonElement>('[data-setting="teacher"]')!.click();
  if (selected === "interface" || selected === "home") handle(() => home!.refresh(true));
  if (selected === "leaderboard") void rankingOperation(() => loadRanking());
  if (selected !== "interface") concealKey();
  });
}));
const rankingTabs = [...document.querySelectorAll<HTMLButtonElement>("[data-ranking-panel]")];
for (const tab of rankingTabs) {
  tab.addEventListener("click", () => {
    for (const item of rankingTabs) {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      $(`ranking-${item.dataset.rankingPanel}`).hidden = !selected;
    }
  });
  tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? rankingTabs[0] : event.key === "End" ? rankingTabs.at(-1) : rankingTabs.find(item => item !== tab);
    next?.focus(); next?.click();
  });
}
document.addEventListener("visibilitychange", () => { if (document.hidden) concealKey(); });
window.addEventListener("pagehide", concealKey);
async function initialize() {
  initializeLanguage(async () => {
    if (settingsEdited) throw new Error(t("请先保存运行或教师设置，再切换语言"));
    await modelTools?.prepareLanguageChange();
    await chat.prepareLanguageChange();
  }, failure);
  if (launchToken) await api("/api/session", { token: launchToken });
  void reportAppStart();
  await chat.load();
  await loadAgreement(true);
  applySettings(await api<Settings>("/api/settings"));
  home = initializeHome({ api, failure, toast, copy });
  await initializeUpdates();
  modelTools = initializeModelTools({ api, failure, toast, refresh: pollStatus });
  modelQuiz = initializeModelQuiz({ api, ready: () => status?.runtime.state === "ready"
    && !["queued", "running"].includes(status?.runtime.job?.state || "") && !backupRestoreBusy, refresh: pollStatus });
  let polling = true;
  setInterval(() => { if (polling || document.hidden) return; polling = true; void pollStatus().catch(failure).finally(() => { polling = false; }); }, 2500);
  try { await pollStatus(); } catch (error) { failure(error); }
  finally { polling = false; }
}
handle(initialize);
