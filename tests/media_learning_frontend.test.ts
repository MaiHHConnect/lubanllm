import { afterEach, expect, test } from "bun:test";
import { initializeMediaLearning } from "../frontend/media-learning";

class Element {
  textContent = ""; value: string | number = ""; hidden = false; disabled = false; checked = false;
  className = ""; max = 1; maxLength = 0; required = false; accept = ""; files: File[] = [];
  nextSibling = { nodeType: 3, textContent: "" };
  children: Element[] = []; listeners: Record<string, Function[]> = {};
  constructor(public tagName = "div") {}
  append(...items: Element[]) { this.children.push(...items); }
  replaceChildren() { this.children = []; }
  addEventListener(type: string, callback: Function) { (this.listeners[type] ||= []).push(callback); }
  fire(type: string) { for (const callback of this.listeners[type] || []) callback(); }
  click() { if (!this.disabled) this.fire("click"); }
  querySelectorAll(selector: string): Element[] {
    return this.children.flatMap(child => [...(selector.split(",").includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
}
const original = { document: (globalThis as any).document, window: (globalThis as any).window,
  localStorage: (globalThis as any).localStorage };
afterEach(() => { for (const [key, value] of Object.entries(original)) {
  if (value === undefined) delete (globalThis as any)[key]; else (globalThis as any)[key] = value;
} });
const settle = async () => { await new Promise(resolve => setTimeout(resolve, 0)); await new Promise(resolve => setTimeout(resolve, 0)); };
const draftKey = "luban-reviewed-operation-draft-v1";
const ids = ["12345678-1234-1234-1234-123456789abc", "87654321-1234-1234-1234-123456789abc"];
const lesson = (kind = "image", fields = {}) => ({ id: "fixture", kind, state: "ready", round: 0, step: 0,
  steps: 5, total_adam_step: 0, has_checkpoint: false, adopted: false, ...fields });
function controller(storage = new Map<string, string>(), response?: (path: string, data?: any) => unknown) {
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  const listeners: Record<string, Function[]> = {}, calls: { path: string; data?: any; method?: string }[] = [], errors: unknown[] = [];
  (globalThis as any).document = { getElementById: get, createElement: (tag: string) => new Element(tag) };
  (globalThis as any).window = { addEventListener: (name: string, cb: Function) => (listeners[name] ||= []).push(cb) };
  (globalThis as any).localStorage = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value) };
  const actions = initializeMediaLearning({ api: (async (path: string, data?: any, method?: string) => {
    calls.push({ path, data, method });
    if (response) return response(path, data);
    if (path.includes("?")) return { kind: path.split("=")[1], round_budget: 5, capability: { available: true }, lesson: null };
    if (path === "/api/learning/media") return lesson(data.kind);
    if (path.endsWith("/start")) return lesson("image", { state: "awaiting_quality", step: 5, has_checkpoint: true });
    return {};
  }) as any, failure: error => errors.push(error), refreshApp: async () => {} });
  return { ...actions, get, calls, errors, storage, record(id: string) {
    for (const cb of listeners["luban-operation-recording"] || []) cb({ detail: { state: "saved", id } });
  } };
}
function consent(app: ReturnType<typeof controller>) { app.get("media-consent").checked = true; app.get("media-consent").fire("change"); }

test("selecting each media tab only GETs status, never reads files or starts teacher/training", async () => {
  const app = controller(); let reads = 0;
  app.get("media-files").files = [{ name: "screen.png", arrayBuffer: async () => { reads++; return new ArrayBuffer(0); } } as File];
  for (const kind of ["image", "sound", "operation", "text"] as const) { app.select(kind); await settle(); }
  expect(reads).toBe(0); expect(app.calls).toHaveLength(3);
  for (const call of app.calls) { expect(call.path).toMatch(/^\/api\/learning\/media\?kind=/); expect(call.data).toBeUndefined(); expect(call.method || "GET").toBe("GET"); }
  expect(app.get("feeding-media-panel").hidden).toBe(true); expect(app.errors).toEqual([]);
});

test("start fails closed until capability confirms available, including rejected status", async () => {
  let resolve: (value: unknown) => void = () => {};
  const app = controller(new Map(), () => new Promise(value => resolve = value));
  app.select("image");
  expect(app.get("media-start").disabled).toBe(true); app.get("media-start").click();
  expect(app.calls).toHaveLength(1);
  resolve({ kind: "image", round_budget: 5, lesson: lesson() }); await settle();
  expect(app.get("media-start").disabled).toBe(true);
  const denied = controller(new Map(), async () => { throw Error("status unavailable"); }); denied.select("image"); await settle();
  expect(denied.get("media-start").disabled).toBe(true); expect(denied.errors).toHaveLength(1);
});

test("validation phase and fixed round status render accurately and pause posts only cancel", async () => {
  let state = lesson("image", { state: "running", phase: "validating", step: 5, total_adam_step: 10 });
  const app = controller(new Map(), (path) => path.endsWith("/cancel")
    ? (state = { ...state, state: "paused" })
    : ({ kind: "image", round_budget: 5, capability: { available: true }, lesson: state }));
  app.select("image"); await settle();
  expect(app.get("media-status").textContent).toContain("复验候选");
  expect(app.get("media-status").textContent).toContain("5 / 5");
  expect(app.get("media-status").textContent).toContain("累计更新 10");
  expect(app.get("media-progress").max).toBe(5); expect(app.get("media-progress").value).toBe(5);
  expect(app.get("media-start").disabled).toBe(true);
  app.get("media-pause").click(); await settle();
  expect(app.calls.filter(call => call.data !== undefined).map(call => call.path)).toEqual(["/api/learning/media/cancel"]);
  expect(app.get("media-status").textContent).toContain("完整检查点已保留");
  state = lesson("image", { state: "awaiting_quality", step: 5, has_checkpoint: true }); await app.refresh();
  expect(app.get("media-status").textContent).toContain("候选已保存"); expect(app.get("media-status").textContent).toContain("未采纳");
  expect(app.get("media-start").textContent).toBe("继续下一轮／断点");
});

test("operation UUID drafts survive image upload/create and refresh, clear only submitted operation lesson", async () => {
  const storage = new Map<string, string>(), app = controller(storage);
  app.select("operation"); await settle(); app.record(ids[0]!); app.record(ids[1]!);
  const saved = storage.get(draftKey); expect(JSON.parse(saved!)).toHaveLength(2);
  app.select("image"); await settle(); expect(app.get("media-files-list").children).toHaveLength(0);
  app.get("media-files").files = [new File(["one"], "one.png"), new File(["two"], "two.png")];
  app.get("media-files").fire("change"); await settle(); consent(app); app.get("media-start").click(); await settle();
  const imageRequest = app.calls.find(call => call.path === "/api/learning/media")!;
  expect(imageRequest.data.kind).toBe("image"); expect(imageRequest.data.demonstrations).toEqual([]);
  expect(storage.get(draftKey)).toBe(saved);
  app.select("sound"); await settle(); expect(storage.get(draftKey)).toBe(saved);
  const restored = controller(storage); restored.select("operation"); await settle();
  expect(restored.get("media-files-list").children).toHaveLength(2);
  consent(restored); restored.get("media-start").click(); await settle();
  const operationRequest = restored.calls.find(call => call.path === "/api/learning/media")!;
  expect(operationRequest.data.kind).toBe("operation"); expect(operationRequest.data.demonstrations).toEqual(JSON.parse(saved!));
  expect(JSON.parse(storage.get(draftKey)!)).toEqual([]); expect(restored.errors).toEqual([]);
});

test("operation create failure retains reviewed drafts and makes no start request", async () => {
  const app = controller(new Map(), async (path) => {
    if (!path.includes("?")) throw Error("invalid original demonstration");
    return { kind: "operation", round_budget: 5, capability: { available: true }, lesson: null };
  });
  app.select("operation"); await settle(); app.record(ids[0]!); app.record(ids[1]!); consent(app);
  app.get("media-start").click(); await settle();
  expect(JSON.parse(app.storage.get(draftKey)!)).toHaveLength(2);
  expect(app.calls.some(call => call.path.endsWith("/start"))).toBe(false); expect(app.errors).toHaveLength(1);
});

test("saving an operation lesson clears submitted UUIDs but preserves a newly recorded draft", async () => {
  let finish: (value: unknown) => void = () => {};
  const app = controller(new Map(), (path, data) => {
    if (path.includes("?")) return { kind: "operation", round_budget: 5, capability: { available: true }, lesson: null };
    if (path === "/api/learning/media") return new Promise(resolve => finish = resolve);
    return lesson("operation", { state: "awaiting_quality", step: 5, has_checkpoint: true });
  });
  app.select("operation"); await settle(); app.record(ids[0]!); app.record(ids[1]!); consent(app);
  app.get("media-start").click(); await settle();
  const pending = "abcdef12-1234-1234-1234-123456789abc"; app.record(pending);
  finish(lesson("operation")); await settle();
  expect(JSON.parse(app.storage.get(draftKey)!)).toEqual([{ file_id: pending, split: "HOLDOUT" }]);
  expect(app.errors).toEqual([]);
});

test("teacher uploads queue without local model capability, answers or automatic training", async () => {
  const app = controller(new Map(), (path, data) => path.includes("?")
    ? { kind: "image", teacher_enabled: true, capability: { available: false, reason: "local library missing" }, lesson: null }
    : lesson(data.kind, { state: "teacher_queued" }));
  app.select("image", "teacher"); await settle();
  app.get("media-files").files = [new File(["one"], "one.png"), new File(["two"], "two.png")];
  app.get("media-files").fire("change"); await settle();
  expect(app.get("media-files-list").querySelectorAll("textarea")).toHaveLength(0);
  expect(app.get("media-files-list").querySelectorAll("input")[0]!.value).toBe("图片中是什么？");
  expect(app.get("media-start").disabled).toBe(true); consent(app);
  expect(app.get("media-start").disabled).toBe(false);
  expect(app.get("media-start").textContent).toBe("交给老师标注");
  expect(app.get("media-consent").nextSibling.textContent).toContain("可能产生费用");
  expect(app.get("media-status").textContent).not.toContain("local library missing");
  app.get("media-start").click(); await settle();
  const mutations = app.calls.filter(call => call.data !== undefined);
  expect(mutations.map(call => call.path)).toEqual(["/api/learning/media"]);
  expect(mutations[0]!.data).toMatchObject({ kind: "image", training_allowed: true,
    answers_reviewed: false, teacher_requested: true, teacher_confirmed: true });
  expect(mutations[0]!.data.files.every((row: any) => row.answer === "" && row.question === "图片中是什么？")).toBe(true);
  expect(app.get("media-status").textContent).toContain("空闲时标注");
  expect(app.get("media-status").textContent).not.toContain("累计更新");
  expect(app.get("media-progress").hidden).toBe(true);
  expect(app.get("media-start").disabled).toBe(true); expect(app.errors).toEqual([]);
});

test("teacher sending requires explicit permission and a confirmed enabled setting", async () => {
  let enabled: boolean | undefined;
  const app = controller(new Map(), () => ({ kind: "sound", teacher_enabled: enabled,
    capability: { available: true }, lesson: null }));
  app.select("sound", "teacher"); await settle();
  app.get("media-files").files = [new File(["one"], "one.wav"), new File(["two"], "two.wav")];
  app.get("media-files").fire("change"); await settle(); consent(app);
  expect(app.get("media-start").disabled).toBe(true);
  enabled = false; await app.refresh();
  expect(app.get("media-start").disabled).toBe(true);
  expect(app.get("media-status").textContent).toContain("设置中开启");
  enabled = true; await app.refresh(); expect(app.get("media-start").disabled).toBe(false);
  app.get("media-consent").checked = false; app.get("media-consent").fire("change");
  expect(app.get("media-start").disabled).toBe(true);
  expect(app.calls.every(call => call.data === undefined)).toBe(true);
});

test("teacher answers remain editable across polling and confirmation never auto starts training", async () => {
  let state = lesson("image", { state: "teacher_review", samples: [
    { id: "sample-1", name: "one.png", question: "图片中是什么？", answer: "老师的答案" },
    { id: "sample-2", name: "two.png", question: "图片中是什么？", answer: "" },
  ] });
  const app = controller(new Map(), (path) => {
    if (path.endsWith("/confirm")) return state = lesson("image");
    return { kind: "image", teacher_enabled: true, capability: { available: true }, lesson: state };
  });
  app.select("image", "teacher"); await settle();
  expect(app.get("media-start").textContent).toBe("确认标注");
  expect(app.get("media-consent").nextSibling.textContent).toContain("我已核对标注");
  expect(app.get("media-start").disabled).toBe(true); consent(app);
  expect(app.get("media-start").disabled).toBe(true); // An empty teacher answer is never implicitly approved.
  const answers = app.get("media-files-list").querySelectorAll("textarea");
  answers[0]!.value = "核对后的一号答案"; answers[0]!.fire("input");
  answers[1]!.value = "核对后的二号答案"; answers[1]!.fire("input");
  const questions = app.get("media-files-list").querySelectorAll("input");
  questions[0]!.value = "核对后的问题"; questions[0]!.fire("input");
  await app.refresh();
  expect(app.get("media-files-list").querySelectorAll("textarea")[0]!.value).toBe("核对后的一号答案");
  expect(app.get("media-start").disabled).toBe(false);
  app.get("media-start").click(); await settle();
  const mutations = app.calls.filter(call => call.data !== undefined);
  expect(mutations.map(call => call.path)).toEqual(["/api/learning/media/confirm"]);
  expect(mutations[0]!.data).toEqual({ id: "fixture", answers_reviewed: true, samples: [
    { id: "sample-1", question: "核对后的问题", answer: "核对后的一号答案" },
    { id: "sample-2", question: "图片中是什么？", answer: "核对后的二号答案" },
  ] });
  expect(app.get("media-start").textContent).toBe("开始本轮训练");
  expect(app.get("media-start").disabled).toBe(false); expect(app.errors).toEqual([]);
});

test("teacher review works with training unavailable; ready still requires training capability", async () => {
  let state = lesson("sound", { state: "teacher_review", samples: [{ id: "audio", question: "", answer: "真实转写" }] });
  const app = controller(new Map(), path => {
    if (path.endsWith("/confirm")) return state = lesson("sound");
    return { kind: "sound", teacher_enabled: false, capability: { available: false, reason: "native unavailable" }, lesson: state };
  });
  app.select("sound", "teacher"); await settle(); consent(app);
  expect(app.get("media-start").disabled).toBe(false);
  app.get("media-start").click(); await settle();
  expect(app.get("media-start").disabled).toBe(true);
  expect(app.get("media-status").textContent).toContain("native unavailable");
  expect(app.calls.some(call => call.path.endsWith("/start"))).toBe(false);
});

test("teacher failures and interruptions never retry or start automatically", async () => {
  for (const state of ["teacher_failed", "teacher_interrupted", "teacher_running"] as const) {
    const app = controller(new Map(), () => ({ kind: "image", teacher_enabled: true,
      capability: { available: true }, lesson: lesson("image", { state, error: "original error" }) }));
    app.select("image", "teacher"); await settle(); consent(app); app.get("media-start").click(); await app.refresh();
    expect(app.get("media-start").disabled).toBe(true);
    expect(app.calls.every(call => call.data === undefined)).toBe(true);
    expect(app.get("media-status").textContent).toContain(state === "teacher_running" ? "尚未训练" : "不会自动重试或训练");
  }
});

test("a stale status response from another mode cannot overwrite the current mode", async () => {
  let resolveOld: (value: unknown) => void = () => {};
  let count = 0;
  const app = controller(new Map(), () => ++count === 1 ? new Promise(resolve => resolveOld = resolve)
    : { kind: "image", teacher_enabled: true, capability: { available: false }, lesson: null });
  app.select("image", "self"); app.select("image", "teacher"); await settle();
  resolveOld({ kind: "image", capability: { available: true }, lesson: lesson("image", { state: "running" }) }); await settle();
  expect(app.get("media-status").textContent).toContain("交给老师标注");
  expect(app.get("media-pause").hidden).toBe(true);
});

test("video uses the existing image tab, video-only permission and one request per clip", async () => {
  const app = controller(new Map(), (path, data) => path.includes("?")
    ? { kind: "image", teacher_enabled: false, video_teacher_enabled: true,
      capability: { available: false, reason: "local model unavailable" }, lesson: null }
    : lesson("image", { state: "teacher_queued", teacher_modality: "video", video_count: 2, frame_count: 0 }));
  app.select("image", "teacher"); await settle();
  expect(app.get("media-type-field").hidden).toBe(false);
  expect(app.get("media-type").value).toBe("image");
  app.get("media-type").value = "video"; app.get("media-type").fire("change");
  expect(app.get("media-files").accept).toBe(".mp4,.mov,.m4v,.webm");
  expect(app.get("media-source-hint").textContent).toContain("按顺序和真实时间共同输入");
  app.get("media-files").files = [new File(["one"], "one.mp4"), new File(["two"], "two.mov")];
  app.get("media-files").fire("change"); await settle();
  expect(app.get("media-start").disabled).toBe(true); consent(app);
  expect(app.get("media-start").disabled).toBe(false);
  expect(app.get("media-consent").nextSibling.textContent).toContain("每段视频1次请求");
  app.get("media-start").click(); await settle();
  const mutations = app.calls.filter(call => call.data !== undefined);
  expect(mutations.map(call => call.path)).toEqual(["/api/learning/media"]);
  expect(mutations[0]!.data).toMatchObject({ kind: "video", teacher_modality: "video", training_allowed: true,
    answers_reviewed: false, teacher_requested: true, teacher_confirmed: true });
  expect(mutations[0]!.data.files.map((row: any) => [row.question, row.answer, row.family])).toEqual([
    ["这段视频中发生了什么？", "", "one.mp4"], ["这段视频中发生了什么？", "", "two.mov"],
  ]);
  expect(app.get("media-status").textContent).toContain("视频数 2 · 已提取画面 0");
  expect(app.get("media-type").disabled).toBe(true);
  expect(app.errors).toEqual([]);
});

test("image teacher permission cannot enable video; switching material clears old files and consent", async () => {
  const app = controller(new Map(), () => ({ kind: "image", teacher_enabled: true, video_teacher_enabled: false,
    capability: { available: true }, lesson: null }));
  app.select("image", "teacher"); await settle();
  app.get("media-files").files = [new File(["one"], "one.png"), new File(["two"], "two.png")];
  app.get("media-files").fire("change"); await settle(); consent(app);
  expect(app.get("media-start").disabled).toBe(false);
  app.get("media-type").value = "video"; app.get("media-type").fire("change");
  expect(app.get("media-files-list").children).toHaveLength(0);
  expect(app.get("media-consent").checked).toBe(false);
  app.get("media-files").files = [new File(["one"], "one.webm"), new File(["two"], "two.m4v")];
  app.get("media-files").fire("change"); await settle(); consent(app);
  expect(app.get("media-start").disabled).toBe(true);
  expect(app.get("media-status").textContent).toContain("设置中开启");
  expect(app.calls.every(call => call.data === undefined)).toBe(true);
  app.select("image", "self"); await settle();
  expect(app.get("media-type-field").hidden).toBe(false);
  expect(app.get("media-files").accept).toBe(".png,.jpg,.jpeg,.webp");
  app.select("sound", "teacher"); await settle();
  expect(app.get("media-type-field").hidden).toBe(true); expect(app.get("media-files").accept).toBe(".wav");
});

test("invalid video count, extension and total size reject before any content read", async () => {
  const { mediaFiles } = await import("../frontend/media-learning");
  let reads = 0;
  const file = (name: string, size = 1) => ({ name, size, arrayBuffer: async () => { reads++; return new ArrayBuffer(1); } }) as File;
  for (const files of [[file("one.mp4")], Array.from({ length: 6 }, (_, i) => file(`${i}.mov`)),
    [file("one.png"), file("two.mp4")], [file("one.mp4", 8 * 1024 * 1024), file("two.mov")]])
    await expect(mediaFiles(files, "image", true)).rejects.toThrow();
  await expect(mediaFiles([file("one.mp4"), file("two.mov")], "sound", true)).rejects.toThrow();
  expect(reads).toBe(0);
});

test("legacy per-frame video reviews remain intact and cannot be switched into a joint lesson", async () => {
  const state = lesson("image", { state: "teacher_review", teacher_modality: "video", video_count: 2, frame_count: 4,
    samples: [0, 1, 2, 3].map(i => ({ id: `frame-${i}`, question: "画面中是什么？", answer: `标注${i}`, family: `source-${i < 2 ? 1 : 2}` })) });
  const app = controller(new Map(), path => path.endsWith("/confirm") ? lesson("image", { teacher_modality: "video" })
    : { kind: "image", teacher_enabled: false, video_teacher_enabled: false, capability: { available: true }, lesson: state });
  app.select("image", "teacher"); await settle();
  expect(app.get("media-files-list").querySelectorAll("textarea")).toHaveLength(4);
  expect(app.get("media-files-list").querySelectorAll("input")).toHaveLength(4); // Questions only; source family cannot be edited.
  expect(app.get("media-status").textContent).toContain("已提取画面 4");
  expect(app.get("media-type").disabled).toBe(true);
  app.get("media-type").value = "video"; app.get("media-type").fire("change");
  expect(app.calls.some(call => call.path === "/api/learning/media?kind=video")).toBe(false);
  consent(app); app.get("media-start").click(); await settle();
  const confirm = app.calls.find(call => call.path.endsWith("/confirm"))!;
  expect(confirm.data.samples.map((row: any) => row.id)).toEqual(["frame-0", "frame-1", "frame-2", "frame-3"]);
  expect(confirm.data.samples.every((row: any) => !("family" in row))).toBe(true);
  expect(app.calls.some(call => call.path.endsWith("/start"))).toBe(false);
});

test("self training submits each complete video QA as kind video without teacher permission", async () => {
  const app = controller(); app.select("image", "self"); await settle();
  expect(app.get("media-type-field").hidden).toBe(false);
  app.get("media-type").value = "video"; app.get("media-type").fire("change"); await settle();
  expect(app.calls.at(-1)!.path).toBe("/api/learning/media?kind=video");
  app.get("media-files").files = [new File(["one"], "one.mp4"), new File(["two"], "two.mov")];
  app.get("media-files").fire("change"); await settle();
  const questions = app.get("media-files-list").querySelectorAll("input");
  questions[0]!.value = "整段先后发生什么？"; questions[0]!.fire("input");
  questions[2]!.value = "另一个片段如何结束？"; questions[2]!.fire("input");
  const answers = app.get("media-files-list").querySelectorAll("textarea");
  expect(answers).toHaveLength(2);
  answers[0]!.value = "先打开门，再离开。"; answers[0]!.fire("input");
  answers[1]!.value = "最后坐下。"; answers[1]!.fire("input");
  consent(app); app.get("media-start").click(); await settle();
  const create = app.calls.find(call => call.path === "/api/learning/media")!;
  expect(create.data.kind).toBe("video");
  expect(create.data.files.map((row: any) => [row.question, row.answer])).toEqual([
    ["整段先后发生什么？", "先打开门，再离开。"], ["另一个片段如何结束？", "最后坐下。"],
  ]);
  expect(create.data).not.toHaveProperty("teacher_modality");
  expect(create.data).not.toHaveProperty("teacher_requested");
  expect(app.calls.filter(call => call.data !== undefined).map(call => call.path)).toEqual([
    "/api/learning/media", "/api/learning/media/start",
  ]);
  expect(app.errors).toEqual([]);
});

test("joint teacher video review has one complete QA per clip and shows ordered frame timestamps", async () => {
  const state = lesson("video", { state: "teacher_review", teacher_modality: "video", video_count: 2, frame_count: 7,
    samples: [
      { id: "clip-1", name: "one.mp4", question: "先后发生什么？", answer: "先开门后离开", frame_count: 4, timestamps: [0.2, 1, 2, 3] },
      { id: "clip-2", name: "two.mov", question: "如何结束？", answer: "坐下", frame_count: 3, timestamps: [0.1, 0.5, 1.2] },
    ] });
  const app = controller(new Map(), path => path.endsWith("/confirm") ? lesson("video")
    : { kind: path.endsWith("=video") ? "video" : "image", teacher_enabled: true, capability: { available: true },
      lesson: path.endsWith("=video") ? state : null });
  app.select("image", "teacher"); await settle();
  app.get("media-type").value = "video"; app.get("media-type").fire("change"); await settle();
  expect(app.get("media-files-list").querySelectorAll("textarea")).toHaveLength(2);
  const names = app.get("media-files-list").querySelectorAll("strong");
  expect(names[0]!.textContent).toContain("画面数 4 · 0.20 s, 1.00 s, 2.00 s, 3.00 s");
  expect(app.get("media-status").textContent).toContain("视频数 2 · 已提取画面 7");
  expect(app.get("media-type").disabled).toBe(true);
  consent(app); app.get("media-start").click(); await settle();
  const confirm = app.calls.find(call => call.path.endsWith("/confirm"))!;
  expect(confirm.data.samples).toEqual([
    { id: "clip-1", question: "先后发生什么？", answer: "先开门后离开" },
    { id: "clip-2", question: "如何结束？", answer: "坐下" },
  ]);
  expect(app.calls.some(call => call.path.endsWith("/start"))).toBe(false);
  expect(app.errors).toEqual([]);
});

test("material switch ignores stale image status and cannot replace a pending teacher lesson", async () => {
  let finish: (value: unknown) => void = () => {};
  const app = controller(new Map(), path => path.endsWith("=image") ? new Promise(resolve => finish = resolve)
    : { kind: "video", teacher_enabled: true, capability: { available: true }, lesson: null });
  app.select("image", "teacher");
  app.get("media-type").value = "video"; app.get("media-type").fire("change"); await settle();
  finish({ kind: "image", lesson: lesson("image", { state: "running" }), capability: { available: true } }); await settle();
  expect(app.get("media-pause").hidden).toBe(true);
  const pending = controller(new Map(), () => ({ kind: "image", teacher_enabled: true, capability: { available: true },
    lesson: lesson("image", { state: "teacher_review", samples: [{ id: "old", question: "原问题", answer: "原答案" }] }) }));
  pending.select("image", "teacher"); await settle();
  expect(pending.get("media-type").disabled).toBe(true); expect(pending.get("media-select").disabled).toBe(true);
  pending.get("media-files").files = [new File(["x"], "x.mp4"), new File(["y"], "y.mp4")];
  pending.get("media-files").fire("change"); await settle();
  expect(pending.get("media-files-list").querySelectorAll("textarea")[0]!.value).toBe("原答案");
  expect(pending.calls.every(call => call.data === undefined)).toBe(true);
});
