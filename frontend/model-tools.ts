import { t, locale, LEARNING_SOURCE_RECHECK_NOTICE } from "./i18n";
import { initializeMediaLearning } from "./media-learning";
type FeedingFile = { name: string; supported: boolean; reason?: string };
type FeedingItem = { id: string; state: string; question: string; answer: string; quote: string; file_name: string };
type FeedingBatch = { id: string; mode?: "self" | "teacher"; state: string; files: FeedingFile[]; provider?: string; model?: string; fee_notice?: string;
  max_teacher_calls: number; teacher_calls_used?: number; unknown_call_count?: number; reason?: string; preview: FeedingItem[];
  calls?: { usage_status?: string; usage?: { total_tokens?: number } }[]; counts?: { pending: number; confirmed: number };
  added?: number; already_confirmed?: number; pending?: number };
type PackageTask = { id?: string; state: string; kind: string; phase?: string; bytes?: number; total?: number;
  error?: string; result?: { download?: string; activated?: boolean; installed?: boolean; cleanup?: unknown } };
type PackageStatus = { task?: PackageTask | null; foundation_available: boolean; foundation_url?: string; export_contains?: string[]; installation?: { required: boolean; download_bytes: number; bundled?: boolean } };
const allowedExtensions = /\.(txt|md|csv|json|jsonl|tsv|log|rst|pdf|doc|docx|rtf|odt|html|htm)$/i;
const documentExtensions = /\.(pdf|doc|docx|rtf|odt|html|htm)$/i;
const teacherAccept = ".txt,.md,.csv,.json,.jsonl,.tsv,.log,.rst,.pdf,.doc,.docx,.rtf,.odt,.html,.htm";
const privateNames = new Set(["internal", "node_modules", "credentials", "credentials.json", "settings", "settings.json", "memory.sqlite3", "模型流程脑图.html"]);

// Inspect names and bounds before reading bytes; unsupported/private files never enter scan payloads.
export async function inspectFeedingFiles(files: Iterable<File>, mode: "self" | "teacher" = "teacher") {
  const payload: { name: string; content: string }[] = [], rejected: FeedingFile[] = [];
  let total = 0;
  const names = new Set<string>();
  for (const file of files) {
    const name = file.webkitRelativePath || file.name;
    const documentFile = documentExtensions.test(name);
    const parts = name.normalize("NFKC").toLowerCase().split("/");
    let reason = "";
    if (!name || name.length > 300 || /[\\:\u0000-\u001f]/.test(name) || parts.some(part => !part || part.startsWith(".") || privateNames.has(part) || /^(settings|credentials)(?:[._-]|$)/.test(part))) reason = t("不能使用私有或内部文件");
    else if (!(mode === "self" ? /\.(csv|json|jsonl)$/i : allowedExtensions).test(name)) reason = t("不支持此格式");
    else if (names.has(name.toLowerCase())) reason = t("文件名重复");
    else if (file.size > (documentFile ? 8 * 1024 * 1024 : 65536)) reason = documentFile ? t("文档为空或超过 8 MiB") : t("超过单文件 64KiB");
    else if (payload.length >= 20) reason = t("超过每批 20 个文件");
    else if (!documentFile && total + file.size > 524288) reason = t("超过每批 512KiB");
    if (reason) { rejected.push({ name, supported: false, reason }); continue; }
    try {
      let content: string;
      if (documentFile) {
        const response = await fetch("/api/feeding/extract?name=" + encodeURIComponent(file.name.split("/").pop()!), {
          method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/octet-stream" }, body: file,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(t(String(result.error?.message || result.detail || "文档无法读取")));
        if (typeof result.text !== "string") throw new Error(t("文档未提取到文本。"));
        content = result.text;
      } else {
        content = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
      }
      const bytes = documentFile ? new TextEncoder().encode(content).length : file.size;
      if (bytes > 65536) reason = t("提取文字超过 64 KiB，请拆分文件后重试");
      else if (total + bytes > 524288) reason = t("超过每批 512KiB");
      else if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(content)) reason = t("不是可读文本");
      else if (content.includes(["LUBAN_INTERNAL", "ONLY_MINDMAP"].join("_"))) reason = t("不能使用内部资料");
      else if (!content.trim()) reason = t("空文件");
      if (reason) rejected.push({ name, supported: false, reason });
      else { payload.push({ name, content }); total += bytes; names.add(name.toLowerCase()); }
    } catch (error) { rejected.push({ name, supported: false, reason: documentFile && error instanceof Error ? error.message : t("无法读取有效 UTF-8 文本") }); }
  }
  return { payload, rejected };
}

export function initializeModelTools({ api, failure, toast, refresh: refreshApp }: {
  api: <T = Record<string, unknown>>(path: string, data?: unknown, method?: string) => Promise<T>;
  failure: (error: unknown) => void; toast: (message: string) => void; refresh: () => Promise<void>;
}) {
  const get = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
  const input = (id: string) => get<HTMLInputElement>(id);
  const button = (id: string) => get<HTMLButtonElement>(id);
  const text = (id: string, value: string) => { get(id).textContent = value; };
  let batch: FeedingBatch | undefined, localRejected: FeedingFile[] = [], feedingBusy = false, polling = false;
  let packages: PackageStatus | undefined, packageBusy = false, pendingImport: File | undefined;
  let bundledInstallAttempted = false;
  let previewKey = "", latestRead = false, feedingMode: "self" | "teacher" = "teacher";
  let feedingKind: "text" | "image" | "sound" | "operation" = "text";
  const feedingKinds = ["text", "image", "sound", "operation"] as const;
  const operationPicker = typeof window === "undefined" ? undefined : (window as any).webkit?.messageHandlers?.lubanFeeding;
  const media = initializeMediaLearning({ api, failure, refreshApp });
  const attempted = new Set<string>(), selected = new Set<string>();
  const visible = (id: string) => !get(id).hidden;
  const size = (n: number) => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GiB` : n >= 1024 ** 2 ? `${(n / 1024 ** 2).toFixed(1)} MiB` : t`${Math.max(0, n).toLocaleString(locale())} 字节`;
  const activePackage = () => packageBusy || ["queued", "running", "uploading"].includes(packages?.task?.state || "");
  function feedingControls() {
    const cleanable = batch?.state === "scanned" && !attempted.has(batch.id) && batch.max_teacher_calls > 0 && batch.max_teacher_calls <= 10;
    button("feeding-clean").disabled = feedingKind !== "text" || feedingBusy || !cleanable || !input("feeding-api-consent").checked;
    button("feeding-confirm").disabled = feedingKind !== "text" || feedingBusy || !selected.size || !input("feeding-answer-consent").checked;
    input("feeding-files").disabled = feedingKind !== "text" || feedingBusy || batch?.state === "cleaning";
    input("feeding-folder").disabled = input("feeding-files").disabled;
    button("feeding-select").disabled = input("feeding-files").disabled;
    button("feeding-select-folder").disabled = input("feeding-files").disabled;
    button("feeding-record-operation").disabled = feedingKind !== "operation" || !operationPicker;
    for (const check of get("feeding-preview").querySelectorAll<HTMLInputElement>("input")) check.disabled = feedingBusy || check.dataset.confirmed === "true";
    const pending = batch?.preview?.filter(item => item.state !== "confirmed") || [];
    button("feeding-select-all").hidden = feedingMode !== "teacher" || !pending.length;
    button("feeding-select-all").disabled = feedingKind !== "text" || feedingBusy || batch?.state === "cleaning";
    text("feeding-select-all", pending.length && pending.every(item => selected.has(item.id)) ? t("取消全选") : t("全选"));
    get("feeding-panel").setAttribute("aria-busy", String(feedingBusy || batch?.state === "cleaning"));
  }
  function selectFeedingKind(kind: typeof feedingKind) {
    feedingKind = kind;
    get("feeding-text-panel").hidden = kind !== "text";
    get("feeding-operation-panel").hidden = kind !== "operation";
    media.select(kind, feedingMode);
    for (const value of feedingKinds) get(`feeding-kind-${value}`).setAttribute("aria-pressed", String(value === kind));
    feedingControls();
  }
  function renderFiles(files: FeedingFile[]) {
    const list = get("feeding-file-list"); list.replaceChildren();
    for (const file of [...files, ...localRejected]) {
      const row = document.createElement("div"); row.className = "feeding-file";
      const name = document.createElement("span"), state = document.createElement("span");
      name.textContent = file.name; state.textContent = file.supported ? t("可用于学习") : t`不支持 · ${file.reason ? t(file.reason) : t("无法识别")}`;
      row.append(name, state); list.append(row);
    }
  }
  function renderFeeding(next: FeedingBatch) {
    const changed = batch?.id !== next.id;
    batch = next;
    if (changed) { input("feeding-api-consent").checked = false; localRejected = []; }
    renderFiles(next.files || []);
    const labels: Record<string, string> = { scanned: t("识别完成，尚未调用教师。"), cleaning: t("教师正在整理资料，请勿重复提交。"), pending_confirmation: t("整理完成，请逐条核对内容。"), pending: t("尚无可确认的问答，保留资料等待处理。"), confirmed: t("已确认内容，合格资料将自动进入学习。"), partially_confirmed: t("已确认部分内容，其余问答保持待核验。") };
    text("feeding-status", (next.reason ? t(next.reason) : "") || (feedingMode === "self" && next.state === "pending_confirmation" ? t("教材已读取，请核对问答后加入学习。") : labels[next.state]) || t("正在读取学习资料状态…"));
    get("feeding-clean-panel").hidden = feedingMode === "self" || next.state !== "scanned";
    const limit = Number.isSafeInteger(next.max_teacher_calls) ? next.max_teacher_calls : 0;
    text("feeding-fee", t`点击“API 教师自动整理”会把支持的资料发送给已配置的教师，可能产生费用。 教师：${next.provider || t("未配置")} · ${next.model || t("未配置")}。最多 ${limit} 次调用，本次上限 10 次。${limit > 10 ? t("请减少资料后重新识别，本批不会调用教师。") : ""}${attempted.has(next.id) ? t("本批已尝试整理；为避免重复收费，不再发起请求。") : ""}`);
    const calls = next.calls || [];
    get("feeding-usage").hidden = !calls.length;
    if (calls.length) {
      const unknown = (next.unknown_call_count || 0) > 0 || calls.some(call => !["reported", "estimated", "not_called"].includes(call.usage_status || "") || call.usage_status === "unknown");
      const tokens = calls.reduce((sum, call) => sum + (call.usage?.total_tokens || 0), 0);
      text("feeding-usage", t`教师已记录调用 ${next.teacher_calls_used ?? t("未知")} 次；${unknown ? t("部分调用或 token 用量未知，实际费用未知。") : t`${calls.some(call => call.usage_status === "estimated") ? t("含估算的") : t("已报告的")}用量 ${tokens.toLocaleString(locale())} tokens，费用由教师服务结算。`}`);
    }
    const key = JSON.stringify([next.id, next.preview]);
    if (key !== previewKey) {
      previewKey = key; selected.clear(); input("feeding-answer-consent").checked = false;
      const list = get("feeding-preview"); list.replaceChildren();
      for (const item of next.preview || []) {
        const row = document.createElement("article"); row.className = "feeding-qa";
        const label = document.createElement("label"); label.className = "checkbox-label";
        const check = document.createElement("input"); check.type = "checkbox"; check.dataset.itemId = item.id; check.dataset.confirmed = String(item.state === "confirmed"); check.disabled = item.state === "confirmed";
        check.addEventListener("change", () => { if (check.checked) selected.add(item.id); else selected.delete(item.id); feedingControls(); });
        const title = document.createElement("span"); title.textContent = `${item.state === "confirmed" ? t("已确认") : t("选择此问答")} · ${item.file_name}`;
        label.append(check, title);
        const details = document.createElement("details"), question = document.createElement("summary");
        const answer = document.createElement("p"), quote = document.createElement("blockquote");
        question.textContent = t`问题：${item.question}`; answer.textContent = t`答案：${item.answer}`; quote.textContent = t`原文引用：${item.quote}`;
        details.append(question, answer, quote); row.append(label, details); list.append(row);
      }
    }
    get("feeding-confirm-panel").hidden = !next.preview?.some(item => item.state !== "confirmed");
    feedingControls();
  }
  async function readFeeding() {
    const requestedMode = feedingMode;
    const result = await api<FeedingBatch | { batches: FeedingBatch[] }>(batch ? `/api/feeding?id=${encodeURIComponent(batch.id)}` : "/api/feeding");
    if (requestedMode !== feedingMode || feedingBusy && batch?.state !== "cleaning") return;
    const next = "batches" in result ? result.batches.find(item => (item.mode || "teacher") === feedingMode) : result;
    latestRead = true;
    if (next) renderFeeding(next);
  }
  async function chooseFiles(files: Iterable<File> | null, rejected: FeedingFile[] = []) {
    const chosen = Array.from(files || []);
    if (feedingKind !== "text" || (!chosen.length && !rejected.length) || feedingBusy || batch?.state === "cleaning") return;
    feedingBusy = true; batch = undefined; selected.clear(); previewKey = ""; get("feeding-preview").replaceChildren();
    get("feeding-confirm-panel").hidden = true; get("feeding-clean-panel").hidden = true; get("feeding-usage").hidden = true;
    feedingControls(); text("feeding-status", t("正在识别文件，不调用教师…"));
    try {
      const inspected = await inspectFeedingFiles(chosen, feedingMode);
      if (feedingKind !== "text") return;
      inspected.rejected.push(...rejected);
      localRejected = inspected.rejected; renderFiles(inspected.payload.map(file => ({ name: file.name, supported: true })));
      if (!inspected.payload.length) { text("feeding-status", t("没有可用于学习的文件，请查看清单。 ")); return; }
      const next = await api<FeedingBatch>("/api/feeding/scan", { files: inspected.payload, mode: feedingMode });
      renderFeeding(next); latestRead = true; localRejected = inspected.rejected; renderFiles(next.files);
    } catch (error) { text("feeding-status", error instanceof Error ? t(error.message) : t("文件识别失败")); failure(error); }
    finally { feedingBusy = false; feedingControls(); input("feeding-files").value = ""; input("feeding-folder").value = ""; }
  }
  async function clean() {
    if (button("feeding-clean").disabled || !batch) return;
    const id = batch.id; attempted.add(id); batch.state = "cleaning"; feedingBusy = true; feedingControls(); text("feeding-status", t("教师正在整理资料，请勿重复提交…"));
    try { renderFeeding(await api<FeedingBatch>("/api/feeding/clean", { id, max_calls: 10, confirmed_api_use: true })); }
    catch (error) {
      failure(error); text("feeding-status", t("整理请求未正常结束，正在核对已保存状态；不会自动重复调用教师。"));
      try { await readFeeding(); } catch { text("feeding-status", t("暂时无法核对状态；本批不会重复调用教师，请稍后查看。实际费用可能未知。")); }
    } finally { feedingBusy = false; feedingControls(); }
  }
  async function confirmAnswers() {
    if (button("feeding-confirm").disabled || !batch) return;
    feedingBusy = true; feedingControls();
    try {
      const result = await api<FeedingBatch>("/api/feeding/confirm", { id: batch.id, ids: [...selected], confirmed_answers: true });
      renderFeeding(result); text("feeding-status", t`新增 ${result.added ?? 0} 条 · 已确认 ${result.already_confirmed ?? 0} 条 · 待处理 ${result.pending ?? result.counts?.pending ?? 0} 条。合格资料将自动学习。`);
      await refreshApp(); toast(t("内容核验已保存"));
    } catch (error) { failure(error); }
    finally { feedingBusy = false; feedingControls(); }
  }
  function packageControls() {
    const busy = activePackage(); button("package-export").disabled = busy; input("package-import").disabled = busy;
    button("install-models").disabled = busy;
    button("package-reset").disabled = busy || !packages?.foundation_available || input("reset-phrase").value !== "我同意重修";
  }
  function renderPackages(next: PackageStatus) {
    packages = next;
    const installRequired = next.installation?.required === true;
    const bundled = next.installation?.bundled === true;
    text("install-models", bundled ? t("安装内置模型") : t("下载基础模型"));
    text("install-description", bundled ? t("文字、图片、语音模型和初始专家已包含在完整安装包中，无需联网下载。") : t("首次使用需下载文字、图片和语音模型。"));
    get("first-install").hidden = !installRequired;
    get("reload-model").hidden = installRequired;
    const installTask = next.task?.kind === "install" ? next.task : undefined;
    const installing = installTask && ["queued", "running"].includes(installTask.state);
    const installProgress = get<HTMLProgressElement>("install-progress");
    installProgress.hidden = !installing;
    installProgress.max = installTask?.total || next.installation?.download_bytes || 1;
    installProgress.value = installTask?.bytes || 0;
    text("install-status", installing ? `${t(installTask.phase === "extract" ? "正在校验与解压" : bundled || installTask.phase === "copy" ? "正在安装内置模型" : "正在下载基础模型")} · ${size(installTask.bytes || 0)} / ${size(installProgress.max)}`
      : installTask?.state === "error" ? t(installTask.error || "基础模型安装失败，请重试")
      : bundled ? t("模型已内置，同意使用协议后自动安装并启动。") : next.installation ? t`下载大小 ${size(next.installation.download_bytes)}，安装后点击启动模型。` : "");
    text("foundation-state", next.foundation_available ? t("基础包已发布；下载、校验和加载成功后才会重置。") : t("基础重修包尚未发布，暂不能重修。当前模型保留。"));
    const task = next.task, progress = get<HTMLProgressElement>("package-progress"), download = get<HTMLAnchorElement>("package-download");
    download.hidden = true; download.removeAttribute("href"); progress.hidden = !task || !["queued", "running", "uploading"].includes(task.state);
    if (!task || task.state === "idle") text("package-status", packageBusy ? t("正在提交模型包任务…") : t("导出后可下载模型 ZIP。 "));
    else {
      const titles: Record<string, string> = { upload: t("正在上传"), export: t("正在打包"), import: t("正在校验与导入"), download: t("正在下载基础包") };
      const running = ["queued", "running", "uploading"].includes(task.state);
      if (running) {
        const validBytes = Number.isFinite(task.bytes) && task.bytes! >= 0;
        const validTotal = Number.isFinite(task.total) && task.total! > 0;
        text("package-status", `${task.state === "queued" ? t("已排队") : titles[task.phase || task.kind] || t("正在处理")}${validBytes ? ` · ${size(task.bytes!)}${validTotal ? ` / ${size(task.total!)}` : ""}` : "…"}`);
        progress.max = validTotal ? task.total! : 1;
        if (validTotal && validBytes) progress.value = Math.min(task.bytes!, task.total!); else progress.removeAttribute("value");
      } else if (task.state === "error") text("package-status", task.error ? t(task.error) : t("任务未完成，请查看错误信息。 "));
      else if (task.state === "complete" && task.kind === "install" && task.result?.installed === true) {
        text("package-status", bundled ? t("内置模型已安装。") : t("基础模型已安装，请点击启动模型。"));
      } else if (task.state === "complete" && task.kind === "export") {
        const path = task.result?.download;
        if (path && /^\/api\/model-packages\/download\/[a-zA-Z0-9_-]+$/.test(path)) { download.href = path; download.hidden = false; text("package-status", t("你的鲁班已打包完成，可以下载保存。 ")); }
        else text("package-status", t("打包完成，尚未取得有效下载地址。 "));
      } else if (task.state === "complete" && task.result?.activated === true) text("package-status", task.kind === "reset" ? t("已切换到基础版本。旧文件清理状态以服务端记录为准。") : t("已切换到包中的学习版本，旧资料保留。"));
      else text("package-status", t("任务处理结束，尚未确认模型版本已切换。 "));
    }
    packageControls();
  }
  async function readPackages() { renderPackages(await api<PackageStatus>("/api/model-packages")); }
  async function packageAction(action: () => Promise<unknown>) {
    if (activePackage()) return;
    packageBusy = true; packageControls(); get("package-download").hidden = true;
    try { await action(); await readPackages(); await refreshApp(); }
    catch (error) { failure(error); try { await readPackages(); } catch { text("package-status", t("暂时无法核对模型包任务状态，请稍后查看。 ")); } }
    finally { packageBusy = false; packageControls(); }
  }
  async function prepareBundledModels() {
    if (!packages?.installation?.bundled || activePackage()) return false;
    if (!packages.installation.required) return true;
    if (bundledInstallAttempted || packages.task?.kind === "install" && packages.task.state === "error") return false;
    // A lost response may still have started the local task. Poll it; never repeat automatically.
    bundledInstallAttempted = true;
    await packageAction(() => api("/api/model-packages/install", {}));
    return false;
  }
  async function upload(file: File) {
    // Send the File as a streamable body, without allocating a multi-GB arrayBuffer/blob copy.
    const response = await fetch("/api/model-packages/import", { method: "POST", headers: { "Content-Type": "application/zip" }, credentials: "same-origin", body: file });
    const result = await response.json();
    if (!response.ok) throw new Error(t(String(result.error || t("导入请求失败"))));
  }
  const deletingLearning = new Set<string>();
  async function removeLearningItem(item: { id: string }) {
    if (deletingLearning.has(item.id)) return;
    deletingLearning.add(item.id);
    try {
      await api(`/api/learning/contents/${encodeURIComponent(item.id)}`, undefined, "DELETE");
      toast(t("已从学习列表删除"));
      await readLearningContents();
      await refreshApp();
    } catch (error) {
      failure(error);
    } finally {
      deletingLearning.delete(item.id);
    }
  }
  async function readLearningContents() {
    try {
      const value = await api<{ learned: { count: number; items: { id: string; prompt: string }[] };
        pending: { count: number; items: { id: string; prompt: string; answer?: string }[] }; limit: number; state: string; error: string | null }>("/api/learning/contents");
      for (const group of [value.learned, value.pending]) {
        if (!group || !Number.isSafeInteger(group.count) || group.count < 0 || !Array.isArray(group.items)
          || group.items.length > 50 || group.items.length > group.count
          || group.items.some(item => typeof item.id !== "string" || typeof item.prompt !== "string")) throw Error(t("学习内容暂不可用"));
      }
      for (const [kind, group] of [["mastered", value.learned], ["pending", value.pending]] as const) {
        text(`learning-${kind}-count`, value.state === "waiting_model" ? "—" : group.count.toLocaleString(locale()));
        const list = get(`learning-${kind}-list`);
        const expanded = new Set([...list.querySelectorAll<HTMLDetailsElement>("details")].filter(row => row.open).map(row => row.dataset.id));
        list.replaceChildren();
        for (const item of group.items) {
          const prompt = [...item.prompt].slice(0, 180).join("") + ([...item.prompt].length > 180 ? "…" : "");
          const row = document.createElement("li");
          if (kind === "pending") {
            const label = document.createElement("details"); label.dataset.id = item.id; label.open = expanded.has(item.id);
            const summary = document.createElement("summary"); summary.textContent = prompt; label.append(summary);
            if (item.prompt !== prompt) { const question = document.createElement("p"); question.textContent = item.prompt; label.append(question); }
            const heading = document.createElement("strong"); heading.textContent = t("参考答案");
            const answer = document.createElement("p"); answer.textContent = "answer" in item && typeof item.answer === "string" && item.answer ? item.answer : t("暂无答案");
            label.append(heading, answer);
            const remove = document.createElement("button"); remove.type = "button"; remove.className = "learning-delete";
            remove.textContent = t("删除"); remove.setAttribute("aria-label", `${t("删除学习条目")}: ${prompt}`);
            remove.disabled = deletingLearning.has(item.id); remove.addEventListener("click", () => { void removeLearningItem(item); });
            row.append(label, remove);
          } else row.textContent = prompt;
          list.append(row);
        }
        if (!group.items.length) {
          const row = document.createElement("li"); row.textContent = kind === "mastered" ? t("暂无独立测试通过的内容") : t("暂无已进入学习流程的资料"); list.append(row);
        }
      }
      // The automatic-learning status owns this blocker; do not repeat it as a content notice.
      const visibleError = value.error === LEARNING_SOURCE_RECHECK_NOTICE ? "" : value.error;
      get("learning-content-state").hidden = value.error === LEARNING_SOURCE_RECHECK_NOTICE || !visibleError && !get("learning-state").hidden;
      text("learning-content-state", (visibleError ? t(visibleError) : "") || (value.state === "waiting_model" ? t("等待模型就绪后核对学习内容。")
        : value.learned.count + value.pending.count === 0 ? t("这里只统计已进入学习流程的资料；没有待学资料不代表模型全部学会。")
        : value.learned.count > 50 || value.pending.count > 50 ? t("已学会以独立测试通过为准；待验证不代表答错。每组展示前 50 项。") : t("已学会以独立测试通过为准；待验证不代表答错。")));
    } catch (error) {
      for (const kind of ["mastered", "pending"]) { text(`learning-${kind}-count`, "—"); get(`learning-${kind}-list`).replaceChildren(); }
      get("learning-content-state").hidden = false;
      text("learning-content-state", error instanceof Error ? t(error.message) : t("学习内容暂不可用"));
    }
  }
  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      if (visible("view-learning")) {
        await readLearningContents();
        if (visible("feeding-panel") && feedingKind !== "text") await media.refresh();
        if (feedingKind === "text" && visible("feeding-panel") && visible("feeding-operation") && (!feedingBusy || batch?.state === "cleaning") && (!latestRead || batch?.state === "cleaning")) await readFeeding();
      }
      if (!packages || packages.installation?.required || (visible("view-settings") && (visible("settings-packages") || visible("settings-reset"))) || activePackage()) await readPackages();
    } finally { polling = false; }
  }
  const run = (action: () => Promise<void>) => { void action().catch(failure); };
  get("install-models").addEventListener("click", () => run(() => packageAction(() => api("/api/model-packages/install", {}))));
  const nativePicker = typeof window !== "undefined" ? (window as Window & { webkit?: { messageHandlers?: { lubanFeeding?: { postMessage: (body: { action: string }) => void } } } }).webkit?.messageHandlers?.lubanFeeding : undefined;
  get("feeding-select-folder").hidden = !!nativePicker;
  if (!nativePicker) text("feeding-select", t("选择文件"));
  get("open-feeding").addEventListener("click", () => {
    get("learning-panel").hidden = true; get("feeding-panel").hidden = false;
    get("open-feeding").setAttribute("aria-expanded", "true");
    get("feeding-choice").hidden = false; get("feeding-operation").hidden = true;
    text("feeding-heading", t("教AI新知识")); text("feeding-back", t("← 返回学习"));
    get("feeding-back").focus();
  });
  for (const mode of ["self", "teacher"] as const) get(`feeding-${mode}`).addEventListener("click", () => {
    if (feedingBusy) return;
    feedingMode = mode; batch = undefined; localRejected = []; selected.clear(); previewKey = ""; latestRead = false;
    get("feeding-choice").hidden = true; get("feeding-operation").hidden = false;
    text("feeding-heading", mode === "self" ? t("自己训练") : t("教师训练")); text("feeding-back", t("← 返回选择"));
    text("feeding-mode-hint", mode === "self" ? t("选择问答教材，核对后加入学习；不调用 API 教师。") : t("选择原文，交给 API 教师整理后再核对并加入学习。"));
    text("feeding-format-hint", mode === "self" ? t("使用 UTF-8 CSV、JSON 或 JSONL，严格包含 question、answer 两个字段；每批最多 16 条，每条问答合计不超过 2000 字符，可下载模板填写。") : t("支持 PDF、Word（DOC/DOCX）、RTF、ODT、HTML，以及 TXT、MD、CSV、TSV、JSON、JSONL、LOG、RST。表格和 JSON 无需预先整理成问答；扫描件请先转为文字。"));
    input("feeding-files").accept = mode === "self" ? ".csv,.json,.jsonl" : teacherAccept;
    get("feeding-guide").hidden = mode === "teacher";
    get("feeding-source-template").hidden = mode === "self";
    get("feeding-file-list").replaceChildren(); get("feeding-preview").replaceChildren();
    for (const id of ["feeding-clean-panel", "feeding-confirm-panel", "feeding-usage"]) get(id).hidden = true;
    text("feeding-status", t("请选择教材。")); selectFeedingKind("text"); run(readFeeding);
  });
  for (const kind of feedingKinds) get(`feeding-kind-${kind}`).addEventListener("click", () => selectFeedingKind(kind));
  get("feeding-record-operation").addEventListener("click", () => {
    if (feedingKind !== "operation" || button("feeding-record-operation").disabled || !operationPicker) return;
    operationPicker.postMessage({ action: "record-operation" });
  });
  if (typeof window !== "undefined") window.addEventListener("luban-operation-recording", event => {
    const detail = (event as CustomEvent<{ message?: unknown; state?: unknown; id?: unknown; actions?: unknown }>).detail;
    if (!detail || typeof detail.message !== "string") return;
    text("feeding-operation-status", t(detail.message.slice(0, 500)));
  });
  get("feeding-back").addEventListener("click", () => {
    if (!get("feeding-operation").hidden) {
      get("feeding-choice").hidden = false; get("feeding-operation").hidden = true;
      text("feeding-heading", t("教AI新知识")); text("feeding-back", t("← 返回学习")); return;
    }
    get("learning-panel").hidden = false; get("feeding-panel").hidden = true;
    get("open-feeding").setAttribute("aria-expanded", "false"); get("open-feeding").focus();
  });
  get("feeding-select").addEventListener("click", () => {
    if (button("feeding-select").disabled) return;
    if (nativePicker) nativePicker.postMessage({ action: "choose" }); else input("feeding-files").click();
  });
  get("feeding-select-folder").addEventListener("click", () => { if (!button("feeding-select-folder").disabled) input("feeding-folder").click(); });
  if (typeof window !== "undefined") window.addEventListener("luban-feeding-files", event => {
    const detail = (event as CustomEvent<{ files: { name: string; content?: string; base64?: string }[]; rejected: FeedingFile[] }>).detail;
    run(() => chooseFiles(detail.files.map(file => new File([file.base64 ? Uint8Array.from(atob(file.base64), char => char.charCodeAt(0)) : file.content || ""], file.name)), detail.rejected));
  });
  for (const id of ["feeding-files", "feeding-folder"]) input(id).addEventListener("change", () => run(() => chooseFiles(input(id).files)));
  for (const id of ["feeding-api-consent", "feeding-answer-consent"]) input(id).addEventListener("change", feedingControls);
  get("feeding-select-all").addEventListener("click", () => {
    if (button("feeding-select-all").disabled || button("feeding-select-all").hidden) return;
    const pending = batch?.preview?.filter(item => item.state !== "confirmed") || [];
    const select = !pending.every(item => selected.has(item.id));
    for (const item of pending) { if (select) selected.add(item.id); else selected.delete(item.id); }
    for (const check of get("feeding-preview").querySelectorAll<HTMLInputElement>("input")) {
      if (check.dataset.confirmed !== "true") check.checked = selected.has(check.dataset.itemId!);
    }
    feedingControls();
  });
  get("feeding-clean").addEventListener("click", () => run(clean)); get("feeding-confirm").addEventListener("click", () => run(confirmAnswers));
  get("package-export").addEventListener("click", () => run(() => packageAction(() => api("/api/model-packages/export", {}))));
  const dialog = get<HTMLDialogElement>("package-import-dialog");
  input("package-import").addEventListener("change", () => {
    pendingImport = input("package-import").files?.[0]; input("package-import").value = "";
    if (!pendingImport || activePackage()) { pendingImport = undefined; return; }
    if (!/\.zip$/i.test(pendingImport.name)) { pendingImport = undefined; failure(new Error(t("请选择模型 ZIP 文件"))); return; }
    text("package-import-name", pendingImport.name); dialog.showModal();
  });
  get("package-import-cancel").addEventListener("click", () => { pendingImport = undefined; dialog.close(); });
  dialog.addEventListener("cancel", () => { pendingImport = undefined; });
  get("package-import-confirm").addEventListener("click", () => { const file = pendingImport; pendingImport = undefined; dialog.close(); if (file) run(() => packageAction(() => upload(file))); });
  input("reset-phrase").addEventListener("input", packageControls);
  get("package-reset").addEventListener("click", () => { if (!button("package-reset").disabled) run(() => packageAction(() => api("/api/model-packages/reset", { phrase: input("reset-phrase").value }))); });
  for (const tab of document.querySelectorAll<HTMLButtonElement>("[data-tab]")) tab.addEventListener("click", () => { if (["learning", "packages"].includes(tab.dataset.tab || "")) run(refresh); });
  for (const menu of document.querySelectorAll<HTMLButtonElement>("[data-setting]")) menu.addEventListener("click", () => { if (["reset", "packages"].includes(menu.dataset.setting || "")) run(readPackages); });
  async function prepareLanguageChange() {
    if (media.hasDraft() || feedingBusy || activePackage() || pendingImport || selected.size || batch && ["scanned", "cleaning", "pending_confirmation", "partially_confirmed", "pending"].includes(batch.state))
      throw new Error(t("请先完成当前资料处理或模型包任务，再切换语言"));
  }
  return { refresh, prepareLanguageChange, prepareBundledModels };
}
