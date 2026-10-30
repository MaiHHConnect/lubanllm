import { t, locale } from "./i18n";
import { INTEGRATIONS } from "./integrations";
import { initializeBackgrounds } from "./backgrounds";

type Growth = { eligible?: boolean; current_score?: number | null; net_delta?: number | null; question_count?: number | null };
type Activity = { local_calls: number; feedback_received: number; teacher_flagged_gaps: number; corrections: number | null; mastered: number | null };
type Home = {
  model: { id: string; name: string; can_rename: boolean; evolution_count: number | null; core_parameters: number | null };
  activity: { today: Activity; week: Activity; application_attribution: boolean };
  growth_profile: { domain: string; growth: Growth | null }[];
};
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const text = (id: string, value: string) => { $(id).textContent = value; };
const count = (value: number | null | undefined) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(locale()) : "—";
const domainNames: Record<string, string> = { mathematics: t("基础算术"), coding: t("ASCII 大写转换"), data_analysis: t("JSON 值提取"), programming: t("ASCII 大写转换"), data_processing: t("JSON 值提取"), data: t("JSON 值提取") };

export function modelStartPresentation(state: string, blocked: boolean, requesting: boolean | "stopping") {
  const current = requesting === "stopping" ? "stopping" : requesting ? "loading" : state;
  const busy = blocked || ["loading", "stopping", "generating", "learning", "training", "evolving", "evaluating"].includes(current);
  const label = current === "loading" ? t("启动中") : current === "ready" ? t("停止模型")
    : current === "stopping" ? t("停止中") : current === "error" ? t("重新启动") : current === "stopped" ? t("启动模型")
    : ({ generating: t("回答中"), learning: t("学习中"), training: t("学习中"), evolving: t("进化中"), evaluating: t("评测中") } as Record<string, string>)[current] || t("连接中");
  const detail = current === "loading" ? t("正在加载模型…") : current === "stopping" ? t("正在停止模型…") : current === "ready" ? t("模型已启动")
    : current === "error" ? t("启动失败，点击重试") : current === "stopped" ? "" : t("请稍候");
  return { state: current, disabled: busy, label, detail };
}

export function initializeHome(actions: {
  api: <T = Record<string, unknown>>(path: string, data?: unknown, method?: string) => Promise<T>;
  failure: (error: unknown) => void;
  toast: (message: string) => void;
  copy: (value: string) => Promise<void>;
}) {
  const backgrounds = initializeBackgrounds();
  let current: Home | undefined;
  let lastRefresh = 0;
  let pending = false;
  const dialog = $<HTMLDialogElement>("connection-dialog");
  const names = $<HTMLDialogElement>("model-name-dialog");
  const run = (action: () => Promise<unknown>) => { void action().catch(actions.failure); };
  const protocol = $<HTMLSelectElement>("connection-protocol");
  let apiBase = "http://127.0.0.1:8765/v1";
  try { protocol.value = localStorage.getItem("luban.connection-protocol") === "claude" ? "claude" : "openai"; }
  catch { protocol.value = "openai"; }
  const connectionBase = () => protocol.value === "claude" ? apiBase.replace(/\/v1\/?$/, "") : apiBase;
  const renderConnection = () => { text("home-api-base", connectionBase()); };
  protocol.addEventListener("change", () => {
    try { localStorage.setItem("luban.connection-protocol", protocol.value); } catch { /* Keep this session's selection. */ }
    text("home-connection-feedback", ""); renderConnection();
  });
  renderConnection();
  for (const app of INTEGRATIONS) {
    const card = document.createElement("button"); card.className = "integration-app";
    const icon = document.createElement("img"); icon.src = app.icon; icon.alt = ""; icon.width = 34; icon.height = 34;
    const label = document.createElement("span"); label.textContent = app.name;
    const vendor = document.createElement("small"); vendor.textContent = app.label;
    card.append(icon, label, vendor);
    card.addEventListener("click", () => {
      text("connection-heading", t`连接 ${app.name}`);
      text("connection-guide", app.guide);
      text("connection-boundary", app.boundary);
      const docs = $<HTMLAnchorElement>("connection-docs"); docs.href = app.docs; docs.hidden = !app.docs;
      dialog.showModal();
    });
    $("integration-apps").append(card);
  }
  const other = document.createElement("button"); other.className = "integration-app other-app";
  other.textContent = t("＋ 其他应用");
  other.addEventListener("click", () => {
    text("connection-heading", t("连接其他应用"));
    text("connection-guide", t("打开应用设置 → 添加自定义模型 → 选择 OpenAI 兼容，填写 API 地址、访问密钥和模型 ID：luban0。"));
    text("connection-boundary", t("应用需允许自定义本机 API 地址。不同应用的字段名称可能不同。"));
    $("connection-docs").hidden = true; dialog.showModal();
  });
  $("integration-apps").append(other);
  $("close-connection").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    const key = $<HTMLInputElement>("access-key"); key.value = ""; key.type = "password";
    text("reveal-key", t("显示")); $<HTMLButtonElement>("copy-key").disabled = true;
  });
  $("home-copy-connection").addEventListener("click", () => run(async () => {
    const target = $<HTMLButtonElement>("home-copy-connection"); target.disabled = true;
    try {
      const { key } = await actions.api<{ key: string }>("/api/access-key");
      const name = protocol.value === "claude" ? "Claude Messages" : "OpenAI Chat Completions";
      await actions.copy(t`协议：${name}\nAPI 地址：${connectionBase()}\nAPI 密钥：${key}\n模型 ID：luban0`);
      text("home-connection-feedback", t("地址、密钥和模型 ID 已复制，请填入应用的自定义模型设置。"));
    } finally { target.disabled = false; }
  }));
  $("rename-model").addEventListener("click", () => run(async () => {
    await refresh(true);
    if (!current?.model.can_rename) { actions.toast(t("完成一次进化后可修改名字")); return; }
    $<HTMLInputElement>("model-display-name").value = current.model.name;
    names.showModal();
  }));
  $("close-model-name").addEventListener("click", () => names.close());
  $("model-name-form").addEventListener("submit", event => {
    event.preventDefault(); run(async () => {
      const target = names.querySelector<HTMLButtonElement>('button[type="submit"]')!; target.disabled = true;
      try {
        await actions.api("/api/home/model-name", { name: $<HTMLInputElement>("model-display-name").value.trim() }, "PATCH");
        await refresh(true); names.close(); actions.toast(t("模型名字已保存"));
      } finally { target.disabled = false; }
    });
  });
  function render(value: Home) {
    backgrounds.update(value.model.evolution_count);
    current = value;
    text("home-model-name", value.model.name);
    text("home-model-meta", t`已进化 ${count(value.model.evolution_count)} 次`);
    $("rename-model").title = value.model.can_rename ? t("修改展示名字") : t("完成一次进化后可修改名字");
    text("home-core-parameters", t`${count(value.model.core_parameters)} 参数`);
    const scores = value.growth_profile.filter(item => item.growth?.eligible && Number.isFinite(item.growth.current_score) && (item.growth.question_count ?? 0) > 0);
    const strongest = [...scores].sort((a, b) => b.growth!.current_score! - a.growth!.current_score!)[0];
    text("home-capability", strongest ? t`${domainNames[strongest.domain] || strongest.domain} · ${strongest.growth!.current_score!.toFixed(1)}%（${count(strongest.growth!.question_count)} 题）` : t("尚无个人评测 · 评测后显示"));
    const { today, week } = value.activity;
    text("home-week-growth", week.mastered === null ? t`新增掌握待复测 · 收到 ${count(week.feedback_received)} 条反馈` : t`独立复测通过 ${count(week.mastered)} 条`);
    text("home-today-gaps", count(today.teacher_flagged_gaps));
    text("home-teacher-feedback", t`今天收到 ${count(today.feedback_received)} 条教师反馈`);
  }
  async function refresh(force = false) {
    if (pending || !force && (Date.now() - lastRefresh < 10000 || ($("view-interface").hidden && $("view-home").hidden))) return;
    pending = true;
    try { render(await actions.api<Home>("/api/home")); lastRefresh = Date.now(); }
    finally { pending = false; }
  }
  function renderRuntime(runtime: {
    state: string;
    job?: { state: string; kind?: string };
    last_generation_usage?: { decode_tokens_per_second?: number | null; tokens_per_second?: number | null; end_to_end_tokens_per_second?: number | null } | null;
    rss_bytes?: number;
    native_process?: { physical_footprint_bytes?: number } | null;
  }, base: string) {
    apiBase = base || "http://127.0.0.1:8765/v1"; renderConnection();
    const usage = runtime.last_generation_usage;
    const speed = usage?.decode_tokens_per_second ?? usage?.tokens_per_second ?? usage?.end_to_end_tokens_per_second;
    text("home-token-speed", typeof speed === "number" && Number.isFinite(speed) && speed > 0 ? t`${speed.toFixed(1)} token/秒` : "—");
    const memory = (value: number | undefined) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? `${(value / 1024 ** 3).toFixed(2)} GiB` : "—";
    text("home-memory-main", runtime.state === "stopped" ? t("模型未启动") : t`主进程 RSS ${memory(runtime.rss_bytes)}`);
    text("home-memory-inference", runtime.state === "stopped" ? t("启动模型后显示推理进程内存") : t`推理进程 ${memory(runtime.native_process?.physical_footprint_bytes)}`);
  }
  return { refresh, renderRuntime };
}
