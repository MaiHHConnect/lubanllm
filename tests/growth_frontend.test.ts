import { t, locale } from "../frontend/i18n";
import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { COUNTRY_CODES, countryLabel } from "../frontend/countries";

const source = readFileSync(new URL("../frontend/app.ts", import.meta.url), "utf8");
const html = readFileSync(new URL("../frontend/index.html", import.meta.url), "utf8");
const transpile = (code: string) => new Bun.Transpiler({ loader: "ts" }).transformSync(code);
const rankingCode = transpile(source.slice(source.indexOf("function scoreText"), source.indexOf('\nfor (const id of ["evolution-mode')));
const pollCode = transpile(source.slice(source.indexOf("async function pollStatus"), source.indexOf("function applySettings")));
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
const summary = { domain: "mathematics", baseline_score: 50, current_score: 58.3, net_delta: 8.3,
  gap_recovery: 25, retention: 91.7, question_count: 24, eligible: true, retention_check: null };
function controller() {
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  get("view-evolution").hidden = true;
  const calls: { path: string; data?: any; method?: string }[] = [];
  let task: any = { state: "idle" };
  let benchmark: any = { state: "idle" };
  let profile: any = { username: "tester", country_code: "CN", evolution_count: 1, token_total: 1500,
    count_error: null, published_profile: null, growth: null,
    growth_profile: ["mathematics", "programming", "data"].map(domain => ({ domain, growth: null })) };
  let entries: any[] = [{ username: "ranked", country_code: "US", evolution_count: 2, token_total: 5000,
    benchmark: 75, benchmark_delta: -4.2, generalization: 62.5, growth: summary, rank: 1 }];
  const api = async (path: string, data?: any, method?: string) => {
    calls.push({ path, data, method });
    if (path === "/api/status") return { runtime: { state: task.state === "running" ? "evaluating" : "ready" }, growth: task, benchmark };
    if (path === "/api/leaderboard/profile") { profile = { ...profile, ...data }; return { profile }; }
    if (path === "/api/growth/start") { task = { state: "queued", total: 24, completed: 0 }; return task; }
    if (path === "/api/growth/cancel") { task = { state: "error", error: "成长评测已取消，未上报本次结果" }; return { ok: true }; }
    if (path === "/api/benchmark/start") { benchmark = { state: "queued", total: 48 }; return benchmark; }
    return { profile, entries, total: entries.length, page: 1, page_size: 100, error: null };
  };
  const run = new Function("$", "input", "select", "button", "text", "count", "api", "COUNTRY_CODES", "countryLabel", "document", "t", "locale", `
    let rankingBusy=false, rankingLoaded=false, rankingCountryEdited=false, rankingPage=1, rankingTotal=0, rankingCountError=null;
    let rankingSort="growth", rankingDomain="mathematics", growthState, benchmarkState, benchmarkRefreshPending=false, status, modelTools, home;
    function renderStatus(next) { status=next; renderBenchmark(next.benchmark); renderGrowth(next.growth); }
    async function maybeRemindEvolution() {}
    async function loadEvolution() { throw new Error("Should not read evolution while evaluating"); }
    ${rankingCode}
    ${pollCode}
    return { loadRanking, pollStatus, renderGrowth, renderCapabilities };
  `);
  const actions = run(get, get, get, get, (id: string, value: string) => { get(id).textContent = value; },
    (n: number) => String(n), api, COUNTRY_CODES, countryLabel, { createElement: () => new Element() }, t, locale);
  return { get, calls, ...actions, setTask: (value: any) => { task = value; }, setEntries: (value: any[]) => { entries = value; },
    setProfile: (value: any) => { profile = { ...profile, ...value }; } };
}
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

test("growth is the default and changing domains resets the page with explicit domain isolation", async () => {
  const app = controller();
  expect(app.calls).toEqual([]);
  await app.loadRanking();
  expect(app.calls[0].path).toBe("/api/leaderboard?page=1&sort=growth&domain=mathematics");
  for (const domain of ["programming", "data", "mathematics"]) {
    app.get("leaderboard-domain").value = domain;
    app.get("leaderboard-domain").listeners.change({ preventDefault() {} });
    await settle();
    expect(app.calls.at(-1)!.path).toBe(`/api/leaderboard?page=1&sort=growth&domain=${domain}`);
  }
  const before = app.calls.length;
  for (const invalid of ["programming&sort=benchmark", "__proto__", "constructor"]) {
    app.get("leaderboard-domain").value = invalid;
    app.get("leaderboard-domain").listeners.change({ preventDefault() {} });
    await settle();
  }
  expect(app.calls.length).toBe(before);
  expect(html).toContain('<option value="growth">本次成长</option>');
  expect(html).not.toContain('<option value="generalization">');
});

test("wrong-domain or unqualified receipts cannot appear in the growth ranking", async () => {
  const app = controller();
  app.setEntries([
    { username: "math", rank: 1, growth: summary },
    { username: "programming", rank: 2, growth: { ...summary, domain: "programming" } },
    { username: "unqualified", rank: 3, growth: { ...summary, eligible: false } },
    { username: "unmeasured", rank: 4, growth: null },
  ]);
  await app.loadRanking();
  expect(app.get("leaderboard-entries").children.length).toBe(1);
  expect(app.get("leaderboard-entries").children[0].children[1].textContent).toBe("math");
});

test("unknown profiles stay unmeasured across the capability profile", async () => {
  const app = controller();
  await app.loadRanking();
  let cards = app.get("capability-profile").children;
  expect(cards.map(card => card.children[0].textContent)).toEqual(["数学", "编程与计算机", "数据与分析", "语言与翻译", "写作与表达", "自然科学", "医疗与健康", "法律", "财经与金融", "教育与学习", "工程与技术", "商业与管理", "历史与人文", "艺术与设计", "生活与实用知识", "宠物与动物"]);
  expect(cards.every(card => card.children[1].textContent === "待验证")).toBe(true);
  expect(cards.every(card => card.children[2].textContent.includes("尚不计分"))).toBe(true);
  app.setProfile({ growth: summary, growth_profile: [{ domain: "mathematics", growth: summary }] });
  await app.loadRanking();
  cards = app.get("capability-profile").children;
  expect(cards[0].children[1].textContent).toBe("50.0% → 58.3%");
  expect(cards[0].children[2].textContent).toBe("缺口解决 25.0% · 旧能力保持 91.7%");
  expect(cards[0].children[3].textContent).toBe("本次成长 +8.3 pp · 有效题数 24");
  expect(cards[1].children[1].textContent).toBe("待验证");
  app.setProfile({ growth_profile: [{ domain: "mathematics", growth: null }] });
  await app.loadRanking();
  expect(app.get("capability-profile").children[0].children[1].textContent).toBe("待验证");
});

test("personal growth and core Benchmark changes occupy separate columns and outputs", async () => {
  const app = controller();
  app.setProfile({ benchmark: 75, benchmark_delta: -4.2, growth: summary,
    growth_profile: [{ domain: "mathematics", growth: summary }] });
  await app.loadRanking();
  const cells = app.get("leaderboard-entries").children[0].children;
  expect(cells[5].textContent).toBe("+8.3 pp · 24 题");
  expect(cells[6].textContent).toBe("91.7%");
  expect(cells[7].textContent).toBe("75.0");
  expect(app.get("leaderboard-benchmark").textContent).toBe("75.0 · -4.2 pp");
  expect(html).not.toContain('id="leaderboard-generalization"');
});

test("one opt-in action queues automatically selected tests without scores or domain input", async () => {
  const app = controller();
  await app.loadRanking();
  app.get("leaderboard-form").listeners.submit({ preventDefault() {} });
  await settle();
  expect(app.calls.at(-2)!.path).toBe("/api/leaderboard/profile");
  expect(Object.keys(app.calls.at(-2)!.data).sort()).toEqual(["country_code", "username"]);
  expect(app.calls.at(-1)).toEqual({ path: "/api/growth/start", data: {}, method: undefined });
  expect(app.get("leaderboard-publish").disabled).toBe(true);
  expect(app.get("benchmark-start").disabled).toBe(true);
  const count = app.calls.length;
  app.get("leaderboard-form").listeners.submit({ preventDefault() {} });
  await settle();
  expect(app.calls.length).toBe(count);
});

test("baseline completion explains automatic follow-up and cannot fabricate a zero score", async () => {
  const app = controller();
  await app.loadRanking();
  app.setTask({ state: "baseline", result: null });
  await app.pollStatus();
  expect(app.get("growth-state").textContent).toBe("已建立成长基线，学习后自动评测");
  expect(app.get("growth-progress").hidden).toBe(true);
  expect(app.get("capability-profile").children[0].children[1].textContent).toBe("待验证");
  expect(app.calls.some(call => call.path === "/api/leaderboard/publish")).toBe(false);
  app.setTask({ state: "idle", eligible_skills: [] });
  await app.pollStatus();
  expect(app.get("growth-state").textContent).toBe("暂缺可独立验证的新题，尚不计分");
});

test("real progress and cancellation preserve last evidence without partial uploads", async () => {
  const app = controller();
  app.setProfile({ growth: summary, growth_profile: [{ domain: "mathematics", growth: summary }] });
  await app.loadRanking();
  app.setTask({ state: "running", phase: "after", completed: 9, total: 24 });
  await app.pollStatus();
  expect(app.get("growth-progress").value).toBe(9);
  expect(app.get("growth-progress").max).toBe(24);
  expect(app.get("growth-state").textContent).toBe("正在评测学习后的成长 · 9 / 24");
  expect(app.get("growth-stop").hidden).toBe(false);
  app.get("growth-stop").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.calls.at(-2)!.path).toBe("/api/growth/cancel");
  expect(app.get("growth-stop").hidden).toBe(true);
  expect(app.get("growth-state").attributes.role).toBe("alert");
  expect(app.get("capability-profile").children[0].children[1].textContent).toBe("50.0% → 58.3%");
});

test("exam uses actual elapsed time and budget across baseline, after and retention", async () => {
  const app = controller();
  app.setTask({ state: "queued", elapsed_seconds: 0, budget_seconds: 300 });
  await app.pollStatus();
  expect(app.get("growth-exam").hidden).toBe(false);
  expect(app.get("growth-exam").attributes["data-state"]).toBe("queued");
  expect(app.get("growth-exam-phase").textContent).toBe("等待考试开始");
  expect(app.get("growth-time").textContent).toBe("已用 0:00 · 上限 5:00");
  for (const [phase, label] of [["baseline", "正在建立成长基线"], ["after", "正在评测学习后的成长"], ["retention", "正在核验延迟保持"], ["original_retest", "正在核验旧能力保持"]]) {
    app.setTask({ state: "running", phase, completed: 9, total: 24, elapsed_seconds: 72.9, budget_seconds: 300 });
    await app.pollStatus();
    expect(app.get("growth-exam-phase").textContent).toBe(label);
    expect(app.get("growth-state").textContent).toBe(`${label} · 9 / 24`);
    expect(app.get("growth-time").textContent).toBe("已用 1:12 · 上限 5:00");
    expect(app.get("growth-exam").attributes["data-state"]).toBe("running");
  }
  // Completion ends the visual without guessing that the budget was fully used.
  app.setTask({ state: "baseline", elapsed_seconds: 90, budget_seconds: 300 });
  await app.pollStatus();
  expect(app.get("growth-exam").hidden).toBe(true);
  expect(app.get("growth-time").textContent).toBe("已用 1:30 · 上限 5:00");
});

test("missing or invalid timer data remains unknown and timeout never invents a score", async () => {
  const app = controller();
  await app.loadRanking();
  app.setTask({ state: "running", elapsed_seconds: -1, budget_seconds: Number.NaN });
  await app.pollStatus();
  expect(app.get("growth-time").textContent).toBe("已用 — · 上限 —");
  app.setTask({ state: "error", error: "达到5分钟上限，未完成题目不计分", elapsed_seconds: 301.2, budget_seconds: 300 });
  await app.pollStatus();
  expect(app.get("growth-exam").hidden).toBe(true);
  expect(app.get("growth-time").textContent).toBe("已用 5:01 · 上限 5:00");
  expect(app.get("growth-state").textContent).toContain("不计分");
  expect(app.get("capability-profile").children.every(card => card.children[1].textContent === "待验证")).toBe(true);
});

test("completion refreshes evidence once and delayed retention remains separate", async () => {
  const app = controller();
  await app.loadRanking();
  app.setTask({ state: "running", completed: 24, total: 24 });
  await app.pollStatus();
  const growth = { ...summary, retention_check: { retention: 83.3 } };
  app.setProfile({ growth, growth_profile: [{ domain: "mathematics", growth }] });
  app.setTask({ state: "complete", result: growth });
  await app.pollStatus();
  expect(app.calls.at(-1)!.path).toBe("/api/leaderboard?page=1&sort=growth&domain=mathematics");
  expect(app.get("capability-profile").children[0].children[4].textContent).toBe("延迟保持 83.3%");
  const before = app.calls.filter(call => call.path.startsWith("/api/leaderboard?")).length;
  await app.pollStatus();
  expect(app.calls.filter(call => call.path.startsWith("/api/leaderboard?")).length).toBe(before);
});

test("terminal phases replace stale initialization text while material details stay intact", async () => {
  const app = controller();
  const stale = "学习前建立个人成长基线，学习后比较独立新题";
  for (const [state, phase, expected] of [
    ["baseline", "已建立成长基线，学习后自动评测", "已建立成长基线，学习后自动评测"],
    ["complete", "个人成长对照已完成", "个人成长对照已完成"],
    ["complete", "已有本版本成绩，未重复计分", "已有本版本成绩，未重复计分"],
    ["cancelled", "评测已停止，未发布部分分数", "评测已停止，未发布部分分数"],
    ["error", "成长评测未完成", "成长评测未完成"],
  ]) {
    app.setTask({ state, phase, message: stale, result: state === "complete" ? summary : null });
    await app.pollStatus();
    expect(app.get("growth-state").textContent).toBe(expected);
    expect(app.get("growth-state").textContent).not.toBe(stale);
  }
  app.setTask({ state: "error", phase: "成长评测未完成", message: stale, error: "达到5分钟上限，未发布部分分数" });
  await app.pollStatus();
  expect(app.get("growth-state").textContent).toBe("达到5分钟上限，未发布部分分数");
  app.setTask({ state: "waiting_materials", phase: "", message: "当前只有开放问答，暂缺可独立验证的新题，尚不计分" });
  await app.pollStatus();
  expect(app.get("growth-state").textContent).toBe("当前只有开放问答，暂缺可独立验证的新题，尚不计分");
  app.setTask({ state: "running", phase: "核对学习后的独立新题", completed: 4, total: 24, message: stale });
  await app.pollStatus();
  expect(app.get("growth-state").textContent).toBe("核对学习后的独立新题 · 4 / 24");
  expect(app.get("growth-exam-phase").textContent).toBe("核对学习后的独立新题");
});

test("domain filter hides for general Benchmark and upgrade sorts while API domain remains explicit", async () => {
  const app = controller();
  await app.loadRanking();
  for (const sort of ["benchmark", "evolution_count", "growth"]) {
    app.get("leaderboard-sort").value = sort;
    app.get("leaderboard-sort").listeners.change({ preventDefault() {} });
    await settle();
    expect(app.get("leaderboard-domain-filter").hidden).toBe(sort !== "growth");
    expect(app.calls.at(-1)!.path).toBe(`/api/leaderboard?page=1&sort=${sort}&domain=mathematics`);
  }
});


test("a later IP lookup fills an untouched country without replacing manual choices", async () => {
  const app = controller();
  app.setProfile({ country_code: null });
  await app.loadRanking();
  expect(app.get("leaderboard-country").value).toBe("");
  app.setProfile({ country_code: "CN" });
  await app.loadRanking();
  expect(app.get("leaderboard-country").value).toBe("CN");
  app.get("leaderboard-country").value = "";
  app.get("leaderboard-country").listeners.change({ preventDefault() {} });
  await app.loadRanking();
  expect(app.get("leaderboard-country").value).toBe("");
});

test("saving a nickname after failed detection does not turn blank into a manual country choice", async () => {
  const app = controller();
  app.setProfile({ country_code: null });
  await app.loadRanking();
  app.get("leaderboard-form").listeners.submit({ preventDefault() {} });
  await settle();
  expect(app.calls.find(call => call.path === "/api/leaderboard/profile")!.data).toEqual({ username: "tester" });
});
