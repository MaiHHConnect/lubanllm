import { t } from "./i18n";

type Kind = "image" | "video" | "sound" | "operation";
type Mode = "self" | "teacher";
type Annotation = { id: string; name?: string; source_name?: string; timestamp_seconds?: number; frame_count?: number; timestamps?: number[]; question: string; answer: string };
type Lesson = { id: string; kind: Kind; state: string; round: number; step: number; steps: number;
  total_adam_step: number; phase?: string; error?: string; has_checkpoint: boolean; adopted: false; teacher_modality?: string; video_count?: number; frame_count?: number; samples?: Annotation[] };
type Status = { kind: Kind; round_budget: number; teacher_enabled?: boolean; video_teacher_enabled?: boolean; lesson?: Lesson | null; capability?: { available: boolean; reason: string } };
type Upload = { name: string; content: string; question: string; answer: string; family: string };
type Demonstration = { file_id: string; split: "TRAIN" | "HOLDOUT" };

export async function mediaFiles(files: Iterable<File>, kind: Kind, video = false): Promise<Upload[]> {
  const selected = [...files];
  if (kind === "operation" || video && kind !== "image" || selected.length < 2 || selected.length > (video ? 5 : 20))
    throw Error(t(video ? "请选择2至5个独立来源的视频。" : "请选择2至20个文件，至少来自两个独立来源。"));
  const allowed = video ? /\.(mp4|mov|m4v|webm)$/i : kind === "image" ? /\.(png|jpe?g|webp)$/i : /\.wav$/i;
  if (selected.some(file => !allowed.test(file.name) || !file.size || /[\\/\u0000-\u001f]/.test(file.name))
      || selected.reduce((sum, file) => sum + file.size, 0) > 8 * 1024 * 1024)
    throw Error(t(video ? "视频支持MP4/MOV/M4V/WebM，每批合计最多8MiB；每个视频不超过30秒。" : "图片支持PNG/JPEG/WebP，声音支持WAV；每批合计最多8MiB。"));
  const result: Upload[] = [];
  for (const file of selected) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 32768)
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
    result.push({ name: file.name, content: btoa(binary), question: "", answer: "", family: file.name });
  }
  return result;
}

export function initializeMediaLearning({ api, failure, refreshApp }: {
  api: <T>(path: string, data?: unknown, method?: string) => Promise<T>;
  failure: (error: unknown) => void; refreshApp: () => Promise<void>;
}) {
  const get = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
  const button = (id: string) => get<HTMLButtonElement>(id);
  const consent = () => get<HTMLInputElement>("media-consent");
  let kind: Kind = "image", mode: Mode = "self", active = false, busy = false, status: Status | undefined;
  let annotations: Annotation[] = [], reviewId = "", material: "image" | "video" = "image";
  const teacherMode = () => mode === "teacher" && kind !== "operation";
  const video = () => kind === "image" && material === "video";
  const apiKind = (): Kind => video() ? "video" : kind;
  const teacherEnabled = () => video() ? status?.video_teacher_enabled ?? status?.teacher_enabled : status?.teacher_enabled;
  const protectedMaterial = () => !!status?.lesson && (status.lesson.state.startsWith("teacher_") || status.lesson.teacher_modality === "video" && status.lesson.kind !== "video");
  const reviewing = () => status?.lesson?.state === "teacher_review" && !uploads.length;
  let uploads: Upload[] = [], demonstrations: Demonstration[] = [];
  const draftKey = "luban-reviewed-operation-draft-v1";
  try {
    const saved: unknown = typeof localStorage === "undefined" ? null : JSON.parse(localStorage.getItem(draftKey) || "null");
    if (Array.isArray(saved) && saved.length <= 1024 && saved.every(row => row && /^[a-f0-9]{8}-[a-f0-9-]{27}$/.test(row.file_id) && ["TRAIN", "HOLDOUT"].includes(row.split)))
      demonstrations = saved;
  } catch { /* A missing/corrupt local draft never grants permission or starts work. */ }
  const saveDraft = () => {
    try { if (typeof localStorage !== "undefined") localStorage.setItem(draftKey, JSON.stringify(demonstrations)); }
    catch { failure(Error(t("操作示范草稿无法保存；请在离开页面前提交已核对的资料。"))); }
  };
  const states: Record<string, string> = {
    ready: "资料已保存，可开始本轮。", running: "正在本地训练", paused: "本轮已暂停，完整检查点已保留。",
    interrupted: "上次学习中断，保留最近完整候选。", awaiting_quality: "本轮训练完成，候选已保存；质量与旧能力检查尚未完成，未采纳。",
    error: "本轮未完成，已保存的候选保留。",
    teacher_queued: "已排队，老师将在空闲时标注；尚未训练。",
    teacher_running: "老师正在标注；尚未训练。",
    teacher_review: "请编辑并核对老师标注；老师答案不自动视为正确。",
    teacher_failed: "老师标注失败；不会自动重试或训练。",
    teacher_interrupted: "老师标注已中断；不会自动重试或训练。",
  };
  function controls() {
    const state = status?.lesson?.state, running = state === "running";
    const waiting = state === "teacher_queued" || state === "teacher_running";
    const sending = teacherMode() && uploads.length > 0;
    const draft = uploads.length > 0 || kind === "operation" && demonstrations.length > 0;
    const review = reviewing();
    const permitted = sending ? teacherEnabled() === true && consent().checked
      : review ? annotations.length > 0 && consent().checked && annotations.every(row => row.answer.trim() && (kind !== "image" || row.question.trim()))
      : status?.capability?.available === true && (draft ? consent().checked : !!status?.lesson && !state?.startsWith("teacher_"));
    get("media-type-field").hidden = kind !== "image";
    get<HTMLSelectElement>("media-type").disabled = busy || running || waiting || protectedMaterial();
    button("media-select").hidden = kind === "operation";
    button("media-select").disabled = busy || running || waiting || protectedMaterial();
    get<HTMLInputElement>("media-files").disabled = busy || running || waiting || protectedMaterial();
    button("media-start").disabled = busy || running || waiting || !permitted;
    button("media-pause").hidden = !running;
    button("media-start").textContent = sending || teacherMode() && !status?.lesson ? t("交给老师标注")
      : review ? t("确认标注")
      : status?.lesson?.has_checkpoint && !draft ? t("继续下一轮／断点") : t("开始本轮训练");
    const label = consent().nextSibling;
    if (label?.nodeType === 3) label.textContent = " " + t(sending || teacherMode() && !status?.lesson
      ? video() ? "我允许将每段视频抽取的最多4帧及时间戳一同发送给已配置的API老师，每段视频1次请求，可能产生费用。" : "我允许将这些图片或录音发送给已配置的API老师标注，并了解可能产生费用。"
      : "我已核对标注和独立来源分组，确认资料允许本地训练。");
    consent().disabled = busy || running || waiting;
    for (const field of get("media-files-list").querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>("input,textarea,select"))
      field.disabled = busy || running || waiting;
  }
  function render() {
    const lesson = status?.lesson;
    if (lesson?.state === "teacher_review" && lesson.id !== reviewId) {
      reviewId = lesson.id; annotations = (lesson.samples || []).map(row => ({ ...row })); list();
    }
    const phases: Record<string, string> = { preparing: "准备中", baseline: "检查训练前能力", training: "实际权重训练", validation: "复验候选", validating: "复验候选", saving: "保存检查点" };
    const teaching = lesson?.state.startsWith("teacher_");
    const message = lesson ? `${t(states[lesson.state] || "正在读取学习状态…")}${lesson.state === "running" ? ` · ${t(phases[lesson.phase || ""] || "准备中")}` : ""}${teaching ? "" : ` · ${lesson.step} / ${lesson.steps} · ${t("累计更新")} ${lesson.total_adam_step}`}${lesson.error ? ` · ${t(lesson.error)}` : ""}`
      : t(teacherMode() ? "上传图片或录音交给老师标注；核对答案后再开始本地训练。" : "选择资料并核对标注后开始。模型未采纳前继续使用原版本。");
    const videoSummary = lesson?.kind === "video" || lesson?.teacher_modality === "video" ? ` · ${t("视频数")} ${lesson.video_count ?? 0} · ${t("已提取画面")} ${lesson.frame_count ?? 0}` : "";
    const training = !teaching && !(teacherMode() && (!lesson || uploads.length));
    get("media-status").textContent = teacherMode() && teacherEnabled() === false && (!lesson || uploads.length)
      ? `${t("请先在设置中开启对应的老师标注选项。")} · ${message}${videoSummary}`
      : training && status?.capability?.available === false ? `${t(status.capability.reason)} · ${message}${videoSummary}` : message + videoSummary;
    const progress = get<HTMLProgressElement>("media-progress");
    progress.hidden = !lesson || !!teaching; progress.max = lesson?.steps || 1; progress.value = lesson?.step || 0;
    get<HTMLInputElement>("media-files").accept = video() ? ".mp4,.mov,.m4v,.webm" : kind === "image" ? ".png,.jpg,.jpeg,.webp" : ".wav";
    get("media-source-hint").textContent = video() || lesson?.teacher_modality === "video" && !uploads.length
      ? t(lesson?.teacher_modality === "video" && lesson.kind !== "video" && !uploads.length
        ? "这份旧视频资料按单帧标注，保留原问答与候选；不会自动变为多帧联合训练。"
        : "每段短视频最多4帧，按顺序和真实时间共同输入，一段一条完整问答；不含音轨，也不代表看过全部画面。每批2至5段、合计8MiB，每段不超过30秒。")
      : kind === "operation" ? t("录制并核对至少两个独立软件来源的示范，指定学习与检验；不会自动执行软件操作。")
      : t("同原图的裁剪、同一说话人的录音须同组；最后一个独立来源仅用于检验。每批最多20个文件、8MiB。");
    controls();
  }
  function field(row: HTMLElement, labelText: string, value: string, update: (value: string) => void, long = false) {
    const label = document.createElement("label"); label.textContent = t(labelText);
    const input = document.createElement(long ? "textarea" : "input"); input.value = value;
    input.maxLength = long ? 4096 : 2048; input.required = true;
    input.addEventListener("input", () => { update(input.value); controls(); }); label.append(input); row.append(label);
  }
  function list() {
    const target = get("media-files-list"); target.replaceChildren();
    for (const upload of uploads) {
      const row = document.createElement("article"); row.className = "feeding-qa";
      const name = document.createElement("strong"); name.textContent = upload.name; row.append(name);
      if (kind === "image") field(row, video() ? "视频问题" : "图片问题", upload.question, value => upload.question = value);
      if (!teacherMode()) field(row, kind === "sound" ? "录音的准确转写" : "完整答案", upload.answer, value => upload.answer = value, true);
      field(row, video() ? "来源分组（同一视频及其剪辑使用同名）" : "来源分组（同原图／同说话人使用同名）", upload.family, value => upload.family = value);
      target.append(row);
    }
    for (const sample of reviewing() ? annotations : []) {
      const row = document.createElement("article"); row.className = "feeding-qa";
      const name = document.createElement("strong"); name.textContent = sample.name || sample.source_name || sample.id;
      if (Array.isArray(sample.timestamps)) name.textContent += ` · ${t("画面数")} ${sample.frame_count ?? sample.timestamps.length} · ${sample.timestamps.map(value => value.toFixed(2) + " s").join(", ")}`;
      else if (typeof sample.timestamp_seconds === "number") name.textContent += ` · ${sample.timestamp_seconds.toFixed(2)} s`;
      row.append(name);
      if (kind === "image") field(row, status?.lesson?.kind === "video" ? "视频问题" : "图片问题", sample.question, value => sample.question = value);
      field(row, kind === "sound" ? "录音的准确转写" : "完整答案", sample.answer, value => sample.answer = value, true);
      target.append(row);
    }
    for (const demo of kind === "operation" ? demonstrations : []) {
      const row = document.createElement("label"); row.textContent = demo.file_id;
      const select = document.createElement("select");
      for (const [value, text] of [["TRAIN", "用于学习"], ["HOLDOUT", "独立检验"]]) {
        const option = document.createElement("option"); option.value = value!; option.textContent = t(text!); select.append(option);
      }
      select.value = demo.split;
      select.addEventListener("change", () => { demo.split = select.value as Demonstration["split"]; saveDraft(); });
      row.append(select); target.append(row);
    }
    consent().checked = false; controls();
  }
  async function refresh() {
    if (!active || busy) return;
    const requested = apiKind(), requestedMode = mode;
    const next = await api<Status>(`/api/learning/media?kind=${requested}`);
    if (requested === apiKind() && requestedMode === mode && active) { status = next; render(); }
  }
  const run = (action: () => Promise<void>) => { void action().catch(failure); };
  button("media-select").addEventListener("click", () => get<HTMLInputElement>("media-files").click());
  get<HTMLInputElement>("media-files").addEventListener("change", () => run(async () => {
    if (get<HTMLInputElement>("media-files").disabled) return;
    const requested = kind, requestedMode = mode, requestedMaterial = material;
    const values = await mediaFiles(get<HTMLInputElement>("media-files").files || [], kind, video());
    if (requested !== kind || requestedMode !== mode || requestedMaterial !== material) return;
    uploads = values;
    if (teacherMode() && kind === "image") for (const upload of uploads) upload.question = t(video() ? "这段视频中发生了什么？" : "图片中是什么？");
    list(); render();
  }));
  get<HTMLSelectElement>("media-type").addEventListener("change", () => {
    if (get<HTMLSelectElement>("media-type").disabled) return;
    material = get<HTMLSelectElement>("media-type").value === "video" ? "video" : "image";
    uploads = []; annotations = []; reviewId = ""; status = undefined;
    get<HTMLInputElement>("media-files").value = ""; list(); render(); run(refresh);
  });
  consent().addEventListener("change", controls);
  button("media-start").addEventListener("click", () => run(async () => {
    if (button("media-start").disabled) return;
    busy = true; controls();
    try {
      if (reviewing()) {
        status!.lesson = await api<Lesson>("/api/learning/media/confirm", { id: status!.lesson!.id,
          samples: annotations.map(({ id, question, answer }) => ({ id, question, answer })), answers_reviewed: true });
        annotations = []; reviewId = ""; list(); render(); await refreshApp(); return;
      }
      const sending = teacherMode() && uploads.length > 0;
      if (uploads.length || kind === "operation" && demonstrations.length) {
        const submitted = kind === "operation" ? demonstrations.slice() : [];
        const lesson = await api<Lesson>("/api/learning/media", { kind: apiKind(), files: uploads,
          demonstrations: submitted, training_allowed: consent().checked, answers_reviewed: sending ? false : consent().checked,
          ...(sending ? { teacher_requested: true, teacher_confirmed: true, ...(video() ? { teacher_modality: "video" } : {}) } : {}) });
        status = { ...status!, kind: apiKind(), lesson }; uploads = [];
        if (submitted.length) { demonstrations = demonstrations.filter(row => !submitted.some(saved => saved.file_id === row.file_id)); saveDraft(); }
        list();
        if (sending) { render(); await refreshApp(); return; }
      }
      if (!status?.lesson) throw Error(t("请先选择学习资料。"));
      status.lesson = await api<Lesson>("/api/learning/media/start", { id: status.lesson.id });
      render(); await refreshApp();
    } finally { busy = false; controls(); }
  }));
  button("media-pause").addEventListener("click", () => run(async () => {
    await api("/api/learning/media/cancel", {}); await refresh();
  }));
  if (typeof window !== "undefined") window.addEventListener("luban-operation-recording", event => {
    const detail = (event as CustomEvent<{ state?: string; id?: string }>).detail;
    if (detail?.state !== "saved" || !/^[a-f0-9-]{36}$/.test(detail.id || "")) return;
    if (demonstrations.some(row => row.file_id === detail.id)) return;
    demonstrations.push({ file_id: detail.id!, split: demonstrations.length ? "HOLDOUT" : "TRAIN" });
    saveDraft(); if (kind === "operation") { uploads = []; list(); }
  });
  return {
    refresh,
    select(value: "text" | Kind, selectedMode: Mode = "self") {
      active = value !== "text"; get("feeding-media-panel").hidden = !active;
      if (value === "text") return;
      if (kind !== value || mode !== selectedMode) { kind = value; mode = selectedMode; uploads = []; annotations = []; reviewId = ""; material = "image"; status = undefined; }
      get<HTMLSelectElement>("media-type").value = material;
      list(); render(); run(refresh);
    },
    hasDraft: () => busy || uploads.length > 0 || demonstrations.length > 0,
  };
}
