import { test, expect } from "bun:test";
import { initializeModelTools, inspectFeedingFiles } from "../frontend/model-tools";

class Element {
  textContent = ""; value = ""; hidden = false; disabled = false; checked = false; type = ""; className = ""; open = false;
  href = ""; max = 1; files: File[] | null = null; dataset: Record<string, string> = {}; attributes: Record<string, string> = {};
  children: Element[] = []; listeners: Record<string, (() => void)[]> = {};
  constructor(public tagName = "div") {}
  append(...items: Element[]) { this.children.push(...items); }
  replaceChildren() { this.children = []; }
  setAttribute(key: string, value: string) { this.attributes[key] = value; }
  removeAttribute(key: string) { delete this.attributes[key]; if (key === "href") this.href = ""; }
  addEventListener(kind: string, action: () => void) { (this.listeners[kind] ||= []).push(action); }
  fire(kind: string) { for (const listener of this.listeners[kind] || []) listener(); }
  querySelectorAll(selector: string): Element[] { return this.children.flatMap(child => [...(child.tagName === selector ? [child] : []), ...child.querySelectorAll(selector)]); }
  click() { this.fire("click"); } focus() {}
  showModal() { this.open = true; } close() { this.open = false; }
}

test("latest rejected learning attempt shows its real date while saved-version dates stay historical", async () => {
  const source = await Bun.file(new URL("../frontend/app.ts", import.meta.url)).text();
  const code = new Bun.Transpiler({ loader: "ts" }).transformSync(source.slice(
    source.indexOf("async function loadVersions()"), source.indexOf("async function loadTrainingBackups()")));
  const { t } = await import("../frontend/i18n");
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  const savedTime = "2026-10-16T07:01:28Z", attemptTime = "2026-10-20T04:15:00Z";
  let lastAttempt: { started_at: string; state: string } | null = { started_at: attemptTime, state: "rejected" };
  const run = new Function("api", "$", "text", "document", "t", "locale", "count", `
    let status, backupRestoreBusy=false;
    ${code}
    return loadVersions;
  `);
  const load = run(async () => ({ versions: [{ id: "learned-old", parameters: 96, created_at: savedTime }], last_attempt: lastAttempt }),
    get, (id: string, value: string) => { get(id).textContent = value; }, { createElement: () => new Element() }, t, () => "zh-CN", String);
  await load();
  expect(get("last-learning-time").textContent).toBe(new Date(attemptTime).toLocaleString("zh-CN"));
  expect(get("last-learning-state").textContent).toBe("未通过检查，未更新模型");
  expect(get("versions").children[0].children[0].textContent).toContain(new Date(savedTime).toLocaleString("zh-CN"));
  lastAttempt = { started_at: attemptTime, state: "running" };
  await load();
  expect(get("last-learning-state").textContent).toBe("进行中");
  lastAttempt = null;
  await load();
  expect(get("last-learning-time").textContent).toBe("—");
  expect(get("last-learning-state").hidden).toBe(true);
  expect(get("versions").children[0].children[0].textContent).toContain(new Date(savedTime).toLocaleString("zh-CN"));
});
const batch = () => ({ id: "batch-1", state: "scanned", files: [{ name: "资料.txt", supported: true }], max_teacher_calls: 1, preview: [] as any[], calls: [] as any[] });
const settle = async () => { await new Promise(resolve => setTimeout(resolve, 0)); await new Promise(resolve => setTimeout(resolve, 0)); };
function controller(overrides: Record<string, unknown> = {}) {
  const elements = new Map<string, Element>(), get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  for (const id of ["view-learning", "view-packages", "view-settings", "settings-reset", "feeding-panel", "learning-state"]) get(id).hidden = true;
  const tabs = ["learning", "packages"].map(tab => { const target = new Element("button"); target.dataset.tab = tab; return target; });
  const menu = new Element("button"); menu.dataset.setting = "reset";
  const calls: { path: string; data?: any; method?: string }[] = [], errors: unknown[] = [];
  let current = batch();
  (globalThis as any).document = { getElementById: get, createElement: (tag: string) => new Element(tag), querySelectorAll: (selector: string) => selector === "[data-tab]" ? tabs : [menu] };
  const api = async (path: string, data?: unknown, method?: string) => {
    calls.push({ path, data, method });
    if (path in overrides) { const result = overrides[path]; if (result instanceof Error) throw result; return typeof result === "function" ? (result as Function)(data) : result; }
    if (path === "/api/feeding/scan") return current;
    if (path === "/api/feeding/clean") { current = { ...current, state: "pending_confirmation", preview: [{ id: "qa-1", state: "pending", file_name: "<img src=x>", question: "<script>question</script>", answer: "answer", quote: "answer in source" }, { id: "qa-2", state: "pending", file_name: "资料.txt", question: "other", answer: "other", quote: "other" }], calls: [{ usage_status: "unknown", usage: { total_tokens: 0 } }] }; return current; }
    if (path === "/api/feeding/confirm") return { ...current, state: "partially_confirmed", added: 1, pending: 1, preview: current.preview.map(item => item.id === "qa-1" ? { ...item, state: "confirmed" } : item) };
    if (path.startsWith("/api/feeding")) return path.includes("?") ? current : { batches: [current] };
    if (path === "/api/external-brain") return { needs_teacher: 2, reference_ready: 3, mastered: 4, total: 9, capacity: 4096 };
    if (path === "/api/model-packages") return { task: null, foundation_available: false };
    return {};
  };
  const actions = initializeModelTools({ api: api as any, failure: error => errors.push(error), toast: () => {}, refresh: async () => {} });
  return { get, calls, errors, tabs, menu, ...actions };
}
function nonMediaCalls(calls: { path: string; data?: unknown; method?: string }[]) {
  for (const call of calls.filter(row => row.path.startsWith("/api/learning/media"))) {
    expect(call.data).toBeUndefined();
    expect(call.method || "GET").toBe("GET");
  }
  return calls.filter(call => !call.path.startsWith("/api/learning/media"));
}
async function scan(app: ReturnType<typeof controller>) {
  app.get("feeding-files").files = [new File(["source answer"], "资料.txt")]; app.get("feeding-files").fire("change"); await settle();
}

test("feeding rejects private/executable/oversized files before any byte read and enforces batch bounds", async () => {
  let reads = 0;
  const fake = (name: string, size: number, relative = "") => ({ name, size, webkitRelativePath: relative, arrayBuffer: async () => { reads++; return new TextEncoder().encode("ok").buffer; } }) as File;
  const result = await inspectFeedingFiles([fake(".env", 10), fake("run.ts", 10), fake("safe.txt", 10, "folder/internal/safe.txt"), fake("settings.json", 10), fake("CREDENTIALS_backup.txt", 10), fake("huge.txt", 65537), ...Array.from({ length: 22 }, (_, i) => fake(`item${i}.txt`, 65536))]);
  expect(reads).toBe(8); expect(result.payload.length).toBe(8); expect(result.rejected.length).toBe(20);
  const twenty = await inspectFeedingFiles(Array.from({ length: 21 }, (_, i) => fake(`small${i}.txt`, 2)));
  expect(twenty.payload.length).toBe(20); expect(twenty.rejected[0].reason).toContain("20");
  const invalid = await inspectFeedingFiles([new File([new Uint8Array([0xff])], "bad.txt")]);
  expect(invalid.payload).toEqual([]);
});

test("scan never calls teacher, consent is required, and preview is literal text with selected ids only", async () => {
  const app = controller(); await scan(app);
  expect(app.calls.map(call => call.path)).toEqual(["/api/feeding/scan"]);
  expect(app.get("feeding-clean").disabled).toBe(true);
  app.get("feeding-api-consent").checked = true; app.get("feeding-api-consent").fire("change"); app.get("feeding-clean").fire("click"); await settle();
  expect(app.calls.at(-1)?.data).toEqual({ id: "batch-1", max_calls: 10, confirmed_api_use: true });
  const rows = app.get("feeding-preview").children;
  expect(rows[0].children[1].children[0].textContent).toBe("问题：<script>question</script>");
  expect(rows[0].children[1].children[0].children.length).toBe(0);
  expect(app.get("feeding-usage").textContent).toContain("费用未知");
  const check = rows[0].querySelectorAll("input")[0]; check.checked = true; check.fire("change");
  expect(app.get("feeding-confirm").disabled).toBe(true);
  app.get("feeding-answer-consent").checked = true; app.get("feeding-answer-consent").fire("change"); app.get("feeding-confirm").fire("click"); await settle();
  expect(app.calls.at(-1)?.data).toEqual({ id: "batch-1", ids: ["qa-1"], confirmed_answers: true });
});

test("failed cleanup reads durable status once and never repeats a potentially billed request", async () => {
  const app = controller({ "/api/feeding/clean": new Error("timeout") }); await scan(app);
  app.get("feeding-api-consent").checked = true; app.get("feeding-api-consent").fire("change"); app.get("feeding-clean").fire("click"); await settle();
  expect(app.calls.at(-1)?.path).toBe("/api/feeding?id=batch-1"); expect(app.get("feeding-clean").disabled).toBe(true);
  app.get("feeding-clean").fire("click"); await settle();
  expect(app.calls.filter(call => call.path.endsWith("/clean")).length).toBe(1);
});

test("package progress uses server bytes, downloads use safe anchor paths, and unavailable reset cannot execute", async () => {
  const progress = { task: { state: "running", kind: "export", bytes: 3, total: 12 }, foundation_available: false };
  const app = controller({ "/api/model-packages": progress }); app.get("view-settings").hidden = false; app.get("settings-packages").hidden = false; await app.refresh();
  expect(app.get("package-progress").value).toBe(3 as any); expect(app.get("package-progress").max).toBe(12);
  app.get("reset-phrase").value = "我同意重修"; app.get("reset-phrase").fire("input"); app.get("package-reset").fire("click"); await settle();
  expect(app.calls.some(call => call.path.endsWith("/reset"))).toBe(false);
  Object.assign(progress.task, { state: "complete", result: { download: "/api/model-packages/download/pkg-1" } }); await app.refresh();
  expect(app.get("package-download").href).toBe("/api/model-packages/download/pkg-1");
  Object.assign(progress.task, { result: { download: "https://evil.test/secret" } }); await app.refresh();
  expect(app.get("package-download").hidden).toBe(true); expect(app.get("package-download").href).toBe("");
});

test("idle package service is ready for export without claiming a completed model switch", async () => {
  const app = controller({ "/api/model-packages": { task: { state: "idle" }, foundation_available: false } });
  app.get("view-settings").hidden = false; app.get("settings-packages").hidden = false; await app.refresh();
  expect(app.get("package-status").textContent).toContain("导出后可下载");
  expect(app.get("package-export").disabled).toBe(false);
});

test("ZIP import sends the original File only after the local confirmation dialog", async () => {
  const app = controller(), file = new File(["zip"], "learning.zip");
  let request: any;
  const oldFetch = globalThis.fetch;
  globalThis.fetch = (async (_url, options) => { request = options; return new Response(JSON.stringify({ started: true }), { headers: { "Content-Type": "application/json" } }); }) as typeof fetch;
  try {
    file.arrayBuffer = async () => { throw new Error("must not buffer ZIP"); };
    app.get("package-import").files = [file]; app.get("package-import").fire("change");
    expect(app.get("package-import-dialog").open).toBe(true); expect(request).toBeUndefined();
    app.get("package-import-confirm").fire("click"); await settle();
    expect(request.body).toBe(file); expect(request.credentials).toBe("same-origin"); expect(request.headers["Content-Type"]).toBe("application/zip");
  } finally { globalThis.fetch = oldFetch; }
});

test("feeding opens its own page and separate file/folder buttons open native inputs without paid work", async () => {
  const app = controller(); let files = 0, folders = 0;
  app.get("feeding-files").addEventListener("click", () => files++);
  app.get("feeding-folder").addEventListener("click", () => folders++);
  app.get("open-feeding").click(); await settle();
  expect(app.get("learning-panel").hidden).toBe(true); expect(app.get("feeding-panel").hidden).toBe(false);
  app.get("feeding-select").click(); app.get("feeding-select-folder").click();
  expect([files, folders]).toEqual([1, 1]);
  expect(app.calls.some(call => call.path.endsWith("/clean") || call.path.endsWith("/confirm"))).toBe(false);
  app.get("feeding-back").click(); expect(app.get("learning-panel").hidden).toBe(false); expect(app.get("feeding-panel").hidden).toBe(true);
});

test("native combined picker shares privacy checks and keeps native rejection reasons", async () => {
  const previous = (globalThis as any).window, listeners: Record<string, Function> = {}, messages: unknown[] = [];
  (globalThis as any).window = { webkit: { messageHandlers: { lubanFeeding: { postMessage: (value: unknown) => messages.push(value) } } }, addEventListener: (event: string, action: Function) => { listeners[event] = action; } };
  try {
    const app = controller(); app.get("feeding-select").click();
    expect(messages).toEqual([{ action: "choose" }]); expect(app.get("feeding-select-folder").hidden).toBe(true);
    listeners["luban-feeding-files"]({ detail: { files: [{ name: "资料/notes.txt", content: "verified source" }, { name: "settings.json", content: "secret" }], rejected: [{ name: "photo.png", supported: false, reason: "不支持此格式" }] } }); await settle();
    expect(app.calls[0].data).toEqual({ files: [{ name: "资料/notes.txt", content: "verified source" }], mode: "teacher" });
    const rejected = app.get("feeding-file-list").children.map(row => row.children[1].textContent).join(" ");
    expect(rejected).toContain("私有"); expect(rejected).toContain("不支持此格式");
    expect(app.calls.some(call => call.path.endsWith("/clean"))).toBe(false);
  } finally { if (previous === undefined) delete (globalThis as any).window; else (globalThis as any).window = previous; }
});


test("feeding chooser is clean and self/teacher routes use distinct operations", async () => {
  const app = controller(); app.get("open-feeding").click();
  expect(app.get("feeding-choice").hidden).toBe(false); expect(app.get("feeding-operation").hidden).toBe(true);
  app.get("feeding-self").click(); await settle();
  expect(app.get("feeding-heading").textContent).toBe("自己训练"); expect(app.get("feeding-source-template").hidden).toBe(true);
  app.get("feeding-files").files = [new File(['{"question":"问题","answer":"答案"}'], "教材.jsonl")]; app.get("feeding-files").fire("change"); await settle();
  expect(app.calls.find(call => call.path === "/api/feeding/scan")?.data.mode).toBe("self");
  expect(app.get("feeding-clean-panel").hidden).toBe(true);
  app.get("feeding-back").click(); expect(app.get("feeding-choice").hidden).toBe(false);
  app.get("feeding-teacher").click(); await settle(); expect(app.get("feeding-heading").textContent).toBe("教师训练"); expect(app.get("feeding-source-template").hidden).toBe(false);
});


test("learning capability tabs preserve text drafts and keep media out of the text teacher", async () => {
  const app = controller(); app.get("open-feeding").click(); app.get("feeding-self").click(); await settle();
  await scan(app);
  const preview = app.get("feeding-preview"), before = app.calls.length;
  for (const kind of ["image", "sound", "operation"]) {
    app.get(`feeding-kind-${kind}`).click();
    expect(app.get("feeding-text-panel").hidden).toBe(true);
    expect(app.get("feeding-media-panel").hidden).toBe(false);
    expect(app.get("feeding-operation-panel").hidden).toBe(kind !== "operation");
    expect(app.get("feeding-select").disabled).toBe(true);
    expect(app.get("feeding-confirm").disabled).toBe(true);
    expect(app.get("feeding-clean").disabled).toBe(true);
    let reads = 0;
    app.get("feeding-files").files = [{ name: "sample.png", arrayBuffer: async () => { reads++; return new ArrayBuffer(0); } } as File];
    app.get("feeding-files").fire("change"); await settle();
    expect(reads).toBe(0); expect(nonMediaCalls(app.calls).length).toBe(before);
  }
  app.get("feeding-kind-text").click();
  expect(app.get("feeding-text-panel").hidden).toBe(false);
  expect(app.get("feeding-media-panel").hidden).toBe(true);
  expect(app.get("feeding-heading").textContent).toBe("自己训练");
  expect(app.get("feeding-preview")).toBe(preview);
  expect(app.get("feeding-select").disabled).toBe(false);
  app.get("feeding-files").files = [new File(['question,answer\n问题,答案'], "教材.csv")];
  app.get("feeding-files").fire("change"); await settle();
  expect(app.calls.at(-1)?.data.mode).toBe("self");
});

test("operation demonstration starts only on an explicit native button and sends no teacher/API request", async () => {
  const previous = (globalThis as any).window, listeners: Record<string, Function> = {}, messages: unknown[] = [];
  (globalThis as any).window = { webkit: { messageHandlers: { lubanFeeding: { postMessage: (value: unknown) => messages.push(value) } } },
    addEventListener: (event: string, action: Function) => { listeners[event] = action; } };
  try {
    const app = controller();
    expect(messages).toEqual([]);
    app.get("feeding-kind-operation").click();
    expect(app.get("feeding-operation-panel").hidden).toBe(false);
    expect(app.get("feeding-record-operation").disabled).toBe(false);
    expect(messages).toEqual([]);
    app.get("feeding-record-operation").click();
    expect(messages).toEqual([{ action: "record-operation" }]);
    expect(nonMediaCalls(app.calls)).toEqual([]);
    listeners["luban-operation-recording"]({ detail: { state: "saved", id: "fixture", actions: 4,
      message: "示范已私有保存；可在学习页训练候选，自动执行尚未接通。" } });
    expect(app.get("feeding-operation-status").textContent).toContain("可在学习页训练候选，自动执行尚未接通");
    app.get("feeding-kind-text").click(); app.get("feeding-record-operation").click();
    expect(messages.length).toBe(1);
    expect(nonMediaCalls(app.calls)).toEqual([]);
  } finally { if (previous === undefined) delete (globalThis as any).window; else (globalThis as any).window = previous; }
});

test("ordinary browser operation tab cannot start a recorder", () => {
  const previous = (globalThis as any).window;
  delete (globalThis as any).window;
  try {
    const app = controller(); app.get("feeding-kind-operation").click();
    expect(app.get("feeding-record-operation").disabled).toBe(true);
    app.get("feeding-record-operation").click(); expect(nonMediaCalls(app.calls)).toEqual([]);
  } finally { if (previous !== undefined) (globalThis as any).window = previous; }
});

test("API text draft stays selected across tabs without another paid request", async () => {
  const app = controller(); app.get("open-feeding").click(); app.get("feeding-teacher").click(); await settle();
  await scan(app);
  app.get("feeding-api-consent").checked = true; app.get("feeding-api-consent").fire("change");
  app.get("feeding-clean").click(); await settle();
  const rows = app.get("feeding-preview").children;
  const check = rows[0].querySelectorAll("input")[0]; check.checked = true; check.fire("change");
  app.get("feeding-answer-consent").checked = true; app.get("feeding-answer-consent").fire("change");
  const calls = app.calls.length;
  app.get("feeding-kind-operation").click(); app.get("feeding-kind-text").click();
  expect(app.get("feeding-heading").textContent).toBe("教师训练");
  expect(app.get("feeding-preview").children).toBe(rows);
  expect(check.checked).toBe(true); expect(app.get("feeding-confirm").disabled).toBe(false);
  expect(nonMediaCalls(app.calls).length).toBe(calls);
  app.get("feeding-confirm").click(); await settle();
  expect(app.calls.find(call => call.path === "/api/feeding/confirm")?.data.ids).toEqual(["qa-1"]);
  expect(app.calls.filter(call => call.path === "/api/feeding/clean").length).toBe(1);
});

test('learning cards use independent-proof content, preserve literal text and fail closed on invalid totals', async () => {
  const data: any = { learned: { count: 1, items: [{ id: 'verified', prompt: '<img src=x onerror=alert(1)>' }] }, pending: { count: 2, items: [{ id: 'unverified', prompt: '尚未通过独立测试的问题' }] }, limit: 50, state: 'ready', error: null };
  const app = controller({ '/api/learning/contents': data }); app.get('view-learning').hidden = false;
  await app.refresh();
  expect(app.get('learning-mastered-count').textContent).toBe('1');
  expect(app.get('learning-pending-count').textContent).toBe('2');
  expect(app.get('learning-mastered-list').children[0].textContent).toBe('<img src=x onerror=alert(1)>');
  expect(app.get('learning-content-state').textContent).toContain('独立测试通过');
  expect(app.get('learning-content-state').textContent).toContain('待验证不代表答错');
  data.learned.count = -1; await app.refresh();
  expect(app.get('learning-mastered-count').textContent).toBe('—');
  expect(app.get('learning-mastered-list').children).toHaveLength(0);
});

test('empty learning content explains its scope and preserves a real verification blocker', async () => {
  const data: any = { learned: { count: 0, items: [] }, pending: { count: 0, items: [] }, limit: 50, state: 'ready', error: null };
  const app = controller({ '/api/learning/contents': data }); app.get('view-learning').hidden = false;
  await app.refresh();
  expect(app.get('learning-content-state').textContent).toContain('不代表模型全部学会');
  expect(app.get('learning-content-state').hidden).toBe(false);
  expect(app.get('learning-pending-list').children[0].textContent).toContain('进入学习流程');
  app.get('learning-state').hidden = false; await app.refresh();
  expect(app.get('learning-content-state').hidden).toBe(true);
  app.get('learning-state').hidden = true; await app.refresh();
  expect(app.get('learning-content-state').hidden).toBe(false);
  data.state = 'waiting_validation'; data.error = '专家学习来源已变化，正在重新核对资料。'; await app.refresh();
  expect(app.get('learning-content-state').hidden).toBe(true);
  expect(app.get('learning-content-state').textContent).not.toContain('专家学习来源已变化');
  data.error = '真实MLX验证未通过'; await app.refresh();
  expect(app.get('learning-content-state').hidden).toBe(false);
  expect(app.get('learning-content-state').textContent).toBe('真实MLX验证未通过');
});

test('pending learning items expose a delete action and mastered items do not', async () => {
  const data: any = { learned: { count: 1, items: [{ id: 'verified', prompt: '已掌握' }] }, pending: { count: 1, items: [{ id: 'remove-me', prompt: '不想学习', answer: '<img src=x onerror=alert(1)>\n完整答案' }] }, limit: 50, state: 'ready', error: null };
  const app = controller({ '/api/learning/contents': data, '/api/learning/contents/remove-me': async () => { data.pending = { count: 0, items: [] }; return { ok: true }; } });
  app.get('view-learning').hidden = false; await app.refresh();
  expect(app.get('learning-mastered-list').children[0].children).toHaveLength(0);
  const details = app.get('learning-pending-list').children[0].children[0];
  expect(details.tagName).toBe('details'); expect(details.open).toBe(false);
  expect(details.children[0].tagName).toBe('summary');
  expect(details.children[2].textContent).toBe('<img src=x onerror=alert(1)>\n完整答案');
  details.open = true; await app.refresh();
  expect(app.get('learning-pending-list').children[0].children[0].open).toBe(true);
  const remove = app.get('learning-pending-list').children[0].children[1]; expect(remove.textContent).toBe('删除'); remove.click(); await settle();
  expect(app.calls.some(call => call.path === '/api/learning/contents/remove-me' && call.method === 'DELETE')).toBe(true);
  expect(app.get('learning-pending-list').children[0].textContent).toContain('进入学习流程');
});

test("teacher documents are locally extracted, retain source names, and self mode rejects raw documents", async () => {
  const savedFetch = globalThis.fetch;
  const requests: string[] = [];
  globalThis.fetch = (async (url: string) => {
    requests.push(String(url));
    return new Response(JSON.stringify({ text: "教师资料原文" }), { headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
  try {
    const pdf = new File([new Uint8Array(100000)], "教材.pdf");
    const result = await inspectFeedingFiles([pdf, new File(["x"], "settings.docx")]);
    expect(result.payload).toEqual([{ name: "教材.pdf", content: "教师资料原文" }]);
    expect(requests).toHaveLength(1);
    expect(requests[0]).toStartWith("/api/feeding/extract?");
    expect(result.rejected).toHaveLength(1);
    expect((await inspectFeedingFiles([pdf], "self")).payload).toEqual([]);
    expect(requests).toHaveLength(1);
    globalThis.fetch = (async () => new Response(JSON.stringify({ text: "x".repeat(65537) }))) as typeof fetch;
    expect((await inspectFeedingFiles([pdf])).rejected[0].reason).toContain("64 KiB");
  } finally { globalThis.fetch = savedFetch; }
});


test("teacher preview hides templates, toggles only pending questions, and keeps answer inspection independent of selection", async () => {
  const app = controller();
  app.get("feeding-teacher").click(); await settle();
  expect(app.get("feeding-guide").hidden).toBe(true);
  expect(app.get("feeding-select-all").hidden).toBe(true);
  await scan(app);
  app.get("feeding-api-consent").checked = true;
  app.get("feeding-api-consent").fire("change"); app.get("feeding-clean").click(); await settle();
  const rows = app.get("feeding-preview").children;
  const details = rows[0].children[1];
  expect(details.tagName).toBe("details"); expect(details.open).toBe(false);
  expect(details.children[0].tagName).toBe("summary");
  expect(details.children[1].textContent).toBe("答案：answer");
  expect(rows[0].querySelectorAll("input")[0].checked).toBe(false);
  expect(app.get("feeding-select-all").hidden).toBe(false);
  const count = app.calls.length;
  app.get("feeding-select-all").click();
  expect(rows.every(row => row.querySelectorAll("input")[0].checked)).toBe(true);
  expect(app.get("feeding-select-all").textContent).toBe("取消全选");
  expect(app.get("feeding-answer-consent").checked).toBe(false);
  expect(app.get("feeding-confirm").disabled).toBe(true);
  expect(app.calls.length).toBe(count);
  app.get("feeding-select-all").click();
  expect(rows.every(row => !row.querySelectorAll("input")[0].checked)).toBe(true);
  const first = rows[0].querySelectorAll("input")[0]; first.checked = true; first.fire("change");
  app.get("feeding-answer-consent").checked = true; app.get("feeding-answer-consent").fire("change");
  app.get("feeding-confirm").click(); await settle();
  app.get("feeding-select-all").click();
  const updated = app.get("feeding-preview").children;
  expect(updated[0].querySelectorAll("input")[0].disabled).toBe(true);
  expect(updated[0].querySelectorAll("input")[0].checked).toBe(false);
  expect(updated[1].querySelectorAll("input")[0].checked).toBe(true);
  app.get("feeding-self").click(); await settle();
  expect(app.get("feeding-guide").hidden).toBe(false);
  expect(app.get("feeding-select-all").hidden).toBe(true);
});

test("first install replaces Start with one download action, displays progress and never auto starts the model", async () => {
  const state: any = { task: { state: "idle" }, foundation_available: false, installation: { required: true, download_bytes: 100 } };
  const app = controller({ "/api/model-packages": () => structuredClone(state), "/api/model-packages/install": () => {
    state.task = { state: "running", kind: "install", phase: "download", bytes: 25, total: 100 };
    return { started: true };
  } });
  await app.refresh();
  expect(app.get("first-install").hidden).toBe(false);
  expect(app.get("reload-model").hidden).toBe(true);
  app.get("install-models").click(); await settle();
  expect(app.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(1);
  expect(app.get("install-models").disabled).toBe(true);
  expect(app.get("install-progress").value).toBe(25 as any);
  state.task = { state: "error", kind: "install", error: "download failed" }; await app.refresh();
  expect(app.get("install-status").textContent).toBe("download failed");
  expect(app.get("install-models").disabled).toBe(false);
  state.task = { state: "complete", kind: "install" }; state.installation.required = false; await app.refresh();
  expect(app.get("first-install").hidden).toBe(true);
  expect(app.get("reload-model").hidden).toBe(false);
  expect(app.calls.some(row => row.path === "/api/model/reload")).toBe(false);
});


test("complete installer offers local model installation without download wording", async () => {
  const app = controller({"/api/model-packages":{task:{state:"idle"},foundation_available:false,installation:{required:true,download_bytes:100,bundled:true}}});
  await app.refresh();
  expect(app.get("install-models").textContent).toBe("安装内置模型");
  expect(app.get("install-status").textContent).toBe("模型已内置，同意使用协议后自动安装并启动。");
  expect(app.get("install-description").textContent).toContain("无需联网下载");
});

test("bundled preparation installs once, waits for completion and also recognizes an existing installation", async () => {
  const state: any = { task: { state: "idle" }, foundation_available: false, installation: { required: true, download_bytes: 100, bundled: true } };
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  const app = controller({ "/api/model-packages": () => structuredClone(state), "/api/model-packages/install": async () => {
    await pending;
    state.task = { state: "running", kind: "install", phase: "copy", bytes: 25, total: 100 };
    return { started: true };
  } });
  await app.refresh();
  const preparing = app.prepareBundledModels();
  expect(await app.prepareBundledModels()).toBe(false);
  release();
  expect(await preparing).toBe(false);
  expect(await app.prepareBundledModels()).toBe(false);
  expect(app.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(1);
  expect(app.get("install-status").textContent).toContain("正在安装内置模型");
  state.task.phase = "validate";
  await app.refresh();
  expect(app.get("install-status").textContent).toContain("正在安装内置模型");
  state.installation.required = false;
  state.task = { state: "complete", kind: "install", result: { installed: true } };
  await app.refresh();
  expect(await app.prepareBundledModels()).toBe(true);
  expect(app.get("package-status").textContent).toBe("内置模型已安装。");
  const reopened = controller({ "/api/model-packages": state });
  await reopened.refresh();
  expect(await reopened.prepareBundledModels()).toBe(true);
  expect(reopened.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(0);
});

test("small downloads stay explicit and bundled install failures require an explicit retry", async () => {
  const small = controller({ "/api/model-packages": { task: null, installation: { required: true, bundled: false, download_bytes: 100 } } });
  await small.refresh();
  expect(await small.prepareBundledModels()).toBe(false);
  expect(small.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(0);
  const state: any = { task: { state: "idle" }, installation: { required: true, bundled: true, download_bytes: 100 } };
  const app = controller({ "/api/model-packages": () => structuredClone(state), "/api/model-packages/install": () => {
    state.task = { state: "error", kind: "install", error: "bundled archive failed validation" };
    throw new Error("bundled archive failed validation");
  } });
  await app.refresh();
  expect(await app.prepareBundledModels()).toBe(false);
  await app.refresh();
  expect(await app.prepareBundledModels()).toBe(false);
  expect(app.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(1);
  expect(app.get("install-status").textContent).toBe("bundled archive failed validation");
  expect(app.errors).toHaveLength(1);
  app.get("install-models").click(); await settle();
  expect(app.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(2);
});

test("lost bundled install response never duplicates installation while durable status is polled", async () => {
  const state: any = { task: null, installation: { required: true, bundled: true, download_bytes: 100 } };
  const app = controller({ "/api/model-packages": () => structuredClone(state), "/api/model-packages/install": () => {
    state.task = { state: "running", kind: "install", phase: "copy" };
    throw new Error("connection interrupted");
  } });
  await app.refresh(); await app.prepareBundledModels();
  await app.refresh(); await app.prepareBundledModels();
  expect(app.calls.filter(row => row.path === "/api/model-packages/install")).toHaveLength(1);
  state.task = { state: "complete", kind: "install" }; state.installation.required = false;
  await app.refresh();
  expect(await app.prepareBundledModels()).toBe(true);
});
