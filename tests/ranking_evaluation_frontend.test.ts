import { t, locale } from "../frontend/i18n";
import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { COUNTRY_CODES, countryLabel } from "../frontend/countries";

const source = readFileSync(new URL("../frontend/app.ts", import.meta.url), "utf8");
const html = readFileSync(new URL("../frontend/index.html", import.meta.url), "utf8");
const compile = (code: string) => new Bun.Transpiler({ loader: "ts" }).transformSync(code);
const rankingCode = compile(source.slice(source.indexOf("function scoreText"), source.indexOf('\nfor (const id of ["evolution-mode')));
const pollCode = compile(source.slice(source.indexOf("async function pollStatus"), source.indexOf("function applySettings")));
const capacityCode = compile(source.slice(source.indexOf("function renderEvolutionCapacity"), source.indexOf("function renderEvolution()")));
class Element {
  value: string | number = "";
  textContent = "";
  hidden = false;
  disabled = false;
  max = 1;
  children: Element[] = [];
  attributes: Record<string, string> = {};
  listeners: Record<string, (event: { preventDefault(): void }) => void> = {};
  append(child: Element) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  setAttribute(key: string, value: string) { this.attributes[key] = value; }
  removeAttribute(key: string) { delete this.attributes[key]; }
  reportValidity() { return true; }
  addEventListener(type: string, action: (event: { preventDefault(): void }) => void) { this.listeners[type] = action; }
}
function controller(total = 101) {
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  get("view-evolution").hidden = true;
  get("leaderboard-sort").value = "growth";
  const calls: { path: string; data?: any; method?: string }[] = [];
  let task: any = { state: "idle" };
  let row: any = { username: "row", country_code: "US", evolution_count: 2, token_total: null,
    growth: { domain: "mathematics", eligible: true, net_delta: null, retention: null, question_count: 24 },
    benchmark: null, generalization: null, benchmark_delta: null, generalization_delta: null };
  let profile: any = { username: "tester", country_code: "CN", evolution_count: 1, token_total: null,
    count_error: null, benchmark: null, generalization: null, benchmark_delta: null,
    generalization_delta: null, elapsed_seconds: null, published_profile: null };
  const api = async (path: string, data?: any, method?: string) => {
    calls.push({ path, data, method });
    if (path === "/api/status") return { runtime: { state: task.state === "running" ? "evaluating" : "ready" }, benchmark: task };
    if (path === "/api/leaderboard/profile") { profile = { ...profile, ...data }; return { profile }; }
    if (path === "/api/benchmark/cancel") { task = { ...task, state: "error", error: "评测已取消，没有提交分数" }; return { ok: true }; }
    if (path === "/api/benchmark/start") { task = { state: "queued", completed: 0, total: 48 }; return task; }
    const page = Number(new URL(path, "http://localhost").searchParams.get("page"));
    const entries = Array.from({ length: Math.max(0, Math.min(100, total - (page - 1) * 100)) }, (_, i) => ({ ...row, rank: (page - 1) * 100 + i + 1 }));
    return { profile, entries, total, page, page_size: 100, error: null };
  };
  const run = new Function("$", "input", "select", "button", "text", "count", "api", "COUNTRY_CODES", "countryLabel", "document", "t", "locale", `
    let rankingBusy=false, rankingLoaded=false, rankingCountryEdited=false, rankingPage=1, rankingTotal=0, rankingCountError=null, rankingSort="growth", rankingDomain="mathematics", growthState;
    let benchmarkState, benchmarkRefreshPending=false, status, modelTools, home;
    function renderStatus(next) { status=next; renderBenchmark(next.benchmark); renderGrowth(next.growth); }
    async function maybeRemindEvolution() {}
    async function loadEvolution() { throw new Error("Evolution tab was not opened"); }
    ${rankingCode}
    ${pollCode}
    return { loadRanking, pollStatus, renderBenchmark };
  `);
  const actions = run(get, get, get, get, (id: string, value: string) => { get(id).textContent = value; },
    (n: number) => String(n), api, COUNTRY_CODES, countryLabel, { createElement: () => new Element() }, t, locale);
  return { get, calls, ...actions, setTask: (value: any) => { task = value; if (value.state === "complete" && value.result) profile = { ...profile, ...value.result }; }, setRow: (value: any) => { row = { ...row, ...value }; },
    setProfile: (value: any) => { profile = { ...profile, ...value }; } };
}
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

const apiCode = compile(source.slice(source.indexOf("async function api<"), source.indexOf("function failure(")));
const apiWith = (fetcher: typeof fetch) => new Function("fetch", "t", `${apiCode}; return api;`)(fetcher, t);

test("local service disconnect reports an actionable error without retrying writes", async () => {
  let calls = 0;
  const api = apiWith(async () => { calls++; throw new TypeError("Load failed"); });
  await expect(api("/api/leaderboard/profile", { username: "tester" }, "PUT"))
    .rejects.toThrow("无法连接本地应用服务");
  expect(calls).toBe(1);
});

test("disconnect while reading an evaluation response uses the same local service message", async () => {
  const api = apiWith(async () => ({ ok: true, text: async () => { throw new TypeError("Load failed"); } }) as Response);
  await expect(api("/api/benchmark/start", {})).rejects.toThrow("无法连接本地应用服务");
});

test("local service error handling preserves validation and authentication messages", async () => {
  const api = apiWith(async () => Response.json({ error: { message: "模型忙碌，请在空闲时评测" } }, { status: 409 }));
  await expect(api("/api/benchmark/start", {})).rejects.toThrow("模型忙碌，请在空闲时评测");
  const unauthorized = apiWith(async () => new Response("", { status: 401 }));
  await expect(unauthorized("/api/benchmark/start", {})).rejects.toThrow("请重新打开鲁班零号应用");
});

test("evaluation queues once, shows real progress, and refreshes once after backend reporting", async () => {
  const app = controller();
  await app.loadRanking(2);
  app.get("benchmark-start").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.calls.at(-2)!.path).toBe("/api/leaderboard/profile");
  expect(app.calls.at(-1)).toEqual({ path: "/api/benchmark/start", data: {}, method: undefined });
  expect(app.get("leaderboard-publish").disabled).toBe(true);
  expect(app.get("benchmark-state").textContent).toBe("通用能力评测已排队");
  const count = app.calls.length;
  app.get("benchmark-start").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.calls.length).toBe(count);
  app.setTask({ state: "running", phase: "generalization", completed: 27, total: 48 });
  await app.pollStatus();
  expect(app.get("benchmark-state").textContent).toBe("正在测通用能力变式 · 27 / 48");
  expect(app.get("benchmark-progress").value).toBe(27);
  expect(app.get("benchmark-progress").max).toBe(48);
  app.setTask({ state: "running", phase: "核验成绩", completed: 48, total: 48 });
  await app.pollStatus();
  expect(app.get("benchmark-state").textContent).toBe("正在测通用能力 · 48 / 48");
  app.setTask({ state: "complete", completed: 48, total: 48,
    result: { benchmark: 75, generalization: 62.5, benchmark_delta: 4.2, generalization_delta: -1, elapsed_seconds: 12.3 } });
  await app.pollStatus();
  expect(app.calls.at(-1)!.path).toBe("/api/leaderboard?page=1&sort=growth&domain=mathematics");
  expect(app.get("leaderboard-benchmark").textContent).toBe("75.0 · +4.2 pp");
  expect(app.get("leaderboard-generalization").textContent).toBe("");
  expect(app.get("leaderboard-elapsed").textContent).toBe("12.3 秒");
  expect(app.get("benchmark-state").textContent).toBe("通用能力评测完成，已上报");
  expect(app.get("leaderboard-publish").disabled).toBe(false);
  expect(app.get("benchmark-progress").hidden).toBe(true);
  const refreshes = app.calls.filter(call => call.path.startsWith("/api/leaderboard?")).length;
  await app.pollStatus();
  expect(app.calls.filter(call => call.path.startsWith("/api/leaderboard?")).length).toBe(refreshes);
  expect(app.calls.some(call => call.path === "/api/leaderboard/publish")).toBe(false);
});

test("unknown scores/tokens stay unknown and the submitted payload never contains scores", async () => {
  const app = controller(1);
  await app.loadRanking();
  expect(app.get("leaderboard-benchmark").textContent).toBe("—");
  expect(app.get("leaderboard-generalization").textContent).toBe("");
  expect(app.get("leaderboard-elapsed").textContent).toBe("—");
  expect(app.get("leaderboard-tokens").textContent).toBe("—");
  const cells = app.get("leaderboard-entries").children[0].children;
  expect([cells[3].textContent, cells[5].textContent, cells[6].textContent, cells[7].textContent]).toEqual(["—", "—", "—", "—"]);
  app.get("benchmark-start").listeners.click({ preventDefault() {} });
  await settle();
  const body = app.calls.find(call => call.path === "/api/leaderboard/profile")!.data;
  expect(Object.keys(body).sort()).toEqual(["country_code", "username"]);
  expect(app.calls.find(call => call.path === "/api/benchmark/start")!.data).toEqual({});
});

test("table places real cumulative tokens before counts and delta in its own column", async () => {
  const app = controller(1);
  app.setProfile({ token_total: 0 });
  app.setRow({ token_total: 1500, benchmark: 75, generalization: 62.5, benchmark_delta: -4.2, generalization_delta: 2 });
  await app.loadRanking();
  const cells = app.get("leaderboard-entries").children[0].children;
  expect(cells.length).toBe(8);
  expect(cells.map((cell: Element) => cell.textContent).slice(3)).toEqual(["1500", "2", "—", "—", "75.0"]);
  expect(app.get("leaderboard-tokens").textContent).toBe("0");
  const header = html.slice(html.indexOf('<table class="leaderboard-table"'), html.indexOf('</thead>', html.indexOf('<table class="leaderboard-table"')));
  expect(header.indexOf("累计Token")).toBeLessThan(header.indexOf("进化次数"));
  expect(header.indexOf("进化次数")).toBeLessThan(header.indexOf("本次成长"));
  expect(header.indexOf("本次成长")).toBeLessThan(header.indexOf("通用Benchmark"));
});

test("changing among the three sorts resets the page without prefetching", async () => {
  const app = controller();
  expect(app.calls).toEqual([]);
  await app.loadRanking(2);
  for (const sort of ["benchmark", "evolution_count", "growth"]) {
    app.get("leaderboard-sort").value = sort;
    app.get("leaderboard-sort").listeners.change({ preventDefault() {} });
    await settle();
    expect(app.calls.at(-1)!.path).toBe(`/api/leaderboard?page=1&sort=${sort}&domain=mathematics`);
  }
  expect(app.calls.length).toBe(4);
  app.get("leaderboard-sort").value = "tokens";
  app.get("leaderboard-sort").listeners.change({ preventDefault() {} });
  await settle();
  expect(app.calls.length).toBe(4);
});

test("evaluation errors do not fabricate success or zero scores and allow retry", async () => {
  const app = controller(1);
  await app.loadRanking();
  app.setTask({ state: "error", error: "评测未完成：存储盘未连接", result: null });
  await app.pollStatus();
  expect(app.get("benchmark-state").textContent).toBe("评测未完成：存储盘未连接");
  expect(app.get("benchmark-state").attributes.role).toBe("alert");
  expect(app.get("leaderboard-benchmark").textContent).toBe("—");
  expect(app.get("leaderboard-publish").disabled).toBe(false);
  expect(app.calls.filter(call => call.path.startsWith("/api/leaderboard?")).length).toBe(1);
  expect(app.calls.some(call => call.path === "/api/leaderboard/publish")).toBe(false);
});

test("count verification failure still blocks queuing an evaluation report", async () => {
  const app = controller(1);
  app.setProfile({ evolution_count: null, count_error: "升级记录损坏" });
  await app.loadRanking();
  app.get("benchmark-start").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.get("leaderboard-publish").disabled).toBe(true);
  expect(app.calls.length).toBe(1);
});

test("running progress polling skips the evolution endpoint that holds the model lock", async () => {
  const app = controller(1);
  app.get("view-evolution").hidden = false;
  app.setTask({ state: "running", completed: 10, total: 48 });
  await app.pollStatus();
  expect(app.calls.map(call => call.path)).toEqual(["/api/status"]);
  expect(app.get("benchmark-progress").value).toBe(10);
});

test("stopping evaluation preserves the last completed score and never uploads partial work", async () => {
  const app = controller(1);
  app.setProfile({ benchmark: 75, generalization: 50, elapsed_seconds: 12 });
  await app.loadRanking();
  app.setTask({ state: "running", completed: 10, total: 48 });
  await app.pollStatus();
  expect(app.get("benchmark-stop").hidden).toBe(false);
  app.get("benchmark-stop").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.calls.at(-2)!.path).toBe("/api/benchmark/cancel");
  expect(app.get("benchmark-stop").hidden).toBe(true);
  expect(app.get("benchmark-state").textContent).toBe("评测已取消，没有提交分数");
  expect(app.get("leaderboard-benchmark").textContent).toBe("75.0");
  expect(app.get("leaderboard-generalization").textContent).toBe("");
  expect(app.calls.some(call => call.path === "/api/leaderboard/publish")).toBe(false);
});

test("a validated profile invalidates an older model's cached status result", async () => {
  const app = controller(1);
  app.setTask({ state: "running", completed: 48, total: 48 });
  await app.pollStatus();
  app.setTask({ state: "complete", result: { benchmark: 75, generalization: 50 } });
  await app.pollStatus();
  expect(app.get("leaderboard-benchmark").textContent).toBe("75.0");
  app.setProfile({ benchmark: null, generalization: null, benchmark_delta: null, generalization_delta: null });
  await app.loadRanking();
  await app.pollStatus();
  expect(app.get("leaderboard-benchmark").textContent).toBe("—");
  expect(app.get("leaderboard-generalization").textContent).toBe("");
});

test("chat can interrupt evaluation and still preserves the existing automatic-learning path", () => {
  const code = compile(source.slice(source.indexOf("function chatReady"), source.indexOf("const chat = initializeChat")));
  const ready = new Function("status", `let backupRestoreBusy=false,teacherFirst=false; ${code}; return chatReady();`);
  expect(ready({ runtime: { state: "evaluating" } })).toBe(true);
  expect(ready({ runtime: { state: "learning", job: { kind: "learning", state: "running" } }, automatic: { state: "running" } })).toBe(true);
  expect(ready({ runtime: { state: "stopped" } })).toBe(false);

});

test("capacity uses actual parameter counts and does not invent growth or memory", () => {
  const nodes = new Map<string, Element>();
  const get = (id: string) => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id)!; };
  const render = new Function("$", "text", "count", "evolution", "status", "t", `${capacityCode}; renderEvolutionCapacity();`);
  const call = (evolution: any, runtime: any = { state: "ready" }) => render(get, (id: string, value: string) => { get(id).textContent = value; }, (n: number) => String(n), evolution, { runtime }, t);
  const current = 2032489600;
  const value = { ready: true, capacity: { current_parameters: current, after_parameters: current, additional_weight_bytes: 0, will_grow: false, eligible: true } };
  call(value);
  expect(get("evolution-current-parameters").textContent).toBe(`${current} 参数`);
  expect(get("evolution-after-parameters").textContent).toBe(`${current} 参数`);
  expect(get("evolution-additional-weights").textContent).toBe("0 B");
  expect(get("evolution-additional-memory").textContent).toBe("待测");
  expect(get("evolution-growth-state").textContent).toContain("🔒 结构扩容暂未开放");
  expect(get("evolution-ready").hidden).toBe(false);
  call({ ...value, ready: false });
  expect(get("evolution-ready").hidden).toBe(true);
  call(value, { state: "evolving", job: { state: "running" } });
  expect(get("evolution-ready").hidden).toBe(true);
  call(undefined);
  expect(get("evolution-current-parameters").textContent).toBe("—");
  expect(get("evolution-after-parameters").textContent).toBe("—");
  expect(get("evolution-additional-weights").textContent).toBe("待测");
  expect(get("evolution-ready").hidden).toBe(true);
});
