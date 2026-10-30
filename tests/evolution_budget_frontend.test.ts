import { t, LEARNING_SOURCE_RECHECK_NOTICE } from "../frontend/i18n";
import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../frontend/app.ts", import.meta.url), "utf8");
const code = new Bun.Transpiler({ loader: "ts" }).transformSync(source.slice(
  source.indexOf("function automaticCount"), source.indexOf("let evolutionStartBusy")));
function controller(automatic: any) {
  const elements = new Map<string, any>();
  const get = (id: string) => {
    if (!elements.has(id)) elements.set(id, { hidden: false });
    return elements.get(id);
  };
  const run = new Function("$", "text", "count", "t", "LEARNING_SOURCE_RECHECK_NOTICE", `
    let status={automatic: ${JSON.stringify(automatic)}};
    ${code}
    return { renderAutomatic, update(value) {status.automatic=value;renderAutomatic();} };
  `);
  const actions = run(get, (id: string, value: string) => { get(id).textContent = value; }, String, t, LEARNING_SOURCE_RECHECK_NOTICE);
  actions.renderAutomatic();
  return { ...actions, get, elements };
}

test("unknown status does not invent counters or learning progress", () => {
  const app = controller(undefined);
  expect(app.get("learning-state").hidden).toBe(true);
  expect(app.get("learning-count").textContent).toBe("—");
  expect(app.get("evolution-experience").hidden).toBe(true);
  app.update({ learning_count: -1, experience: { current: 3, target: 16 } });
  expect(app.get("learning-count").textContent).toBe("—");
  expect(app.get("evolution-experience").hidden).toBe(false);
  expect(app.get("evolution-experience").value).toBe(3);
  expect(app.get("evolution-experience").max).toBe(16);
});

test("verification and material blockers stay visible until learning resumes", () => {
  const app = controller({ state: "waiting_validation", reason: "当前专家来源尚未通过验证" });
  expect(app.get("learning-state").hidden).toBe(false);
  expect(app.get("learning-state").textContent).toBe("当前专家来源尚未通过验证");
  app.update({ state: "waiting_materials", reason: "等待足够的已核验学习资料：训练 3 / 16 条，独立验证 0 / 2 条。" });
  expect(app.get("learning-state").hidden).toBe(false);
  expect(app.get("learning-state").textContent).toBe("等待足够的已核验学习资料：训练 3 / 16 条，独立验证 0 / 2 条。");
  app.update({ state: "waiting_materials" });
  expect(app.get("learning-state").textContent).toContain("独立验证资料");
  app.update({ state: "idle" });
  expect(app.get("learning-state").hidden).toBe(true);
});

test("source verification failure stays visible without claiming an automatic recheck", () => {
  const app = controller({ state: "waiting_validation", reason: LEARNING_SOURCE_RECHECK_NOTICE, fastbrain_updates: 2, learning_count: 3 });
  expect(app.get("learning-state").hidden).toBe(false);
  expect(app.get("learning-state").textContent).toBe("学习暂不可用：引擎更新后需要重新验证。");
  expect(app.get("learning-state").textContent).not.toContain("正在");
  expect(app.get("fastbrain-updates").textContent).toBe("2");
  expect(app.get("learning-count").textContent).toBe("3");
  app.update({ state: "running", reason: "" });
  expect(app.get("learning-state").textContent).toBe("后台正在学习");
});

test("legacy status budgets do not add controls or a per-step panel", () => {
  const app = controller({ evolution_budget: "full" });
  expect([...app.elements.keys()].some(id => id.includes("budget"))).toBe(false);
  const html = readFileSync(new URL("../frontend/index.html", import.meta.url), "utf8");
  for (const id of ["evolution-round", "learning-round", "evolution-budget-recommended", "evolution-budget-full"])
    expect(html).not.toContain(`id="${id}"`);
  expect(source).not.toContain("evolution_budget");
});

function evolutionController(initial: any, ready = true) {
  const elements = new Map<string, any>();
  const get = (id: string) => {
    if (!elements.has(id)) elements.set(id, {});
    return elements.get(id);
  };
  const messages: string[] = [], requests: string[] = [];
  const evolutionCode = new Bun.Transpiler({ loader: "ts" }).transformSync(source.slice(
    source.indexOf("let evolutionStartBusy"), source.indexOf("function renderEvolution()")));
  const run = new Function("initial", "ready", "button", "text", "count", "t", "toast", "api", "LEARNING_SOURCE_RECHECK_NOTICE", `
    let status=initial, evolution={ready,capacity:{eligible:ready}};
    const renderEvolution=()=>renderEvolutionCapacity(), pollStatus=async()=>{}, failure=()=>{};
    ${evolutionCode}
    return {startEvolution, renderEvolutionCapacity, update(next) {status=next;renderEvolutionCapacity();}};
  `);
  const actions = run(initial, ready, get, (id: string, value: string) => { get(id).textContent = value; }, String, t,
    (message: string) => messages.push(message), async (path: string) => { requests.push(path); }, LEARNING_SOURCE_RECHECK_NOTICE);
  actions.renderEvolutionCapacity();
  return { ...actions, get, messages, requests };
}
const evolutionStatus = (current: number, runtime = "ready") => ({ runtime: { state: runtime },
  automatic: { experience: { current, target: 16 }, pending_signature: current >= 16 ? "verified" : null } });

test("insufficient evolution materials remain clickable and report exact shortfall without a start request", async () => {
  const app = evolutionController(evolutionStatus(3));
  expect(app.get("evolution-start").disabled).toBe(false);
  await app.startEvolution();
  expect(app.messages).toEqual(["还差 13 条合格新知识才能进化（已有 3 / 16 条）。"]);
  expect(app.get("evolution-start-hint").textContent).toBe(app.messages[0]);
  expect(app.requests).toEqual([]);
  app.update(evolutionStatus(16));
  await app.startEvolution();
  expect(app.requests).toEqual(["/api/evolution/start"]);
});

test("unknown counts and stopped models do not invent a shortfall or start evolution", async () => {
  const app = evolutionController({ runtime: { state: "ready" } });
  await app.startEvolution();
  expect(app.messages.pop()).toBe("正在统计合格新知识，请稍后重试。");
  app.update(evolutionStatus(3, "stopped"));
  await app.startEvolution();
  expect(app.messages.pop()).toBe("请先启动模型，再开始进化。");
  app.update(evolutionStatus(16, "loading"));
  await app.startEvolution();
  expect(app.messages.pop()).toBe("模型尚未就绪，请稍后再开始进化。");
  expect(app.requests).toEqual([]);
});

test("active work stays disabled and reaching the count does not bypass other eligibility checks", async () => {
  const app = evolutionController({ ...evolutionStatus(16), runtime: { state: "ready", job: { state: "running" } } });
  expect(app.get("evolution-start").disabled).toBe(true);
  await app.startEvolution();
  expect(app.requests).toEqual([]);
  const blocked = evolutionController({ runtime: { state: "ready" }, automatic: {
    experience: { current: 16, target: 16 }, reason: "独立验证资料不足" } }, false);
  await blocked.startEvolution();
  expect(blocked.messages).toEqual(["独立验证资料不足"]);
  expect(blocked.requests).toEqual([]);
});

function learningController(initial: any, respond: (path: string, body: any) => any = (_, body) => ({ automatic: body?.automatic })) {
  const elements = new Map<string, any>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, { checked: true, disabled: false }); return elements.get(id); };
  const requests: any[] = [], messages: string[] = [], failures: unknown[] = [];
  const code = new Bun.Transpiler({ loader: "ts" }).transformSync(source.slice(
    source.indexOf("let learningStartBusy"), source.indexOf("let evolutionStartBusy")));
  const run = new Function("initial", "button", "input", "api", "toast", "failure", "t", "LEARNING_SOURCE_RECHECK_NOTICE", `
    let status=initial;
    const pollStatus=async()=>{}, loadVersions=async()=>{};
    ${code}
    return {startLearning, saveAutomaticLearning, update(next){status=next;renderLearningControls();}};
  `);
  const actions = run(initial, get, get, async (path: string, body: any, method: string) => {
    requests.push({ path, body, method }); return respond(path, body);
  }, (message: string) => messages.push(message), (error: unknown) => failures.push(error), t, LEARNING_SOURCE_RECHECK_NOTICE);
  return { ...actions, get, requests, messages, failures };
}

test("learn now explains missing training or validation materials without starting, and preserves real blockers", async () => {
  const app = learningController(evolutionStatus(3));
  await app.startLearning();
  expect(app.messages.pop()).toBe("资料不足");
  expect(app.get("learning-start").disabled).toBe(false);
  app.update({ ...evolutionStatus(16), automatic: { experience: { current: 16, target: 16 },
    state: "waiting_materials", reason: "等待足够的已核验学习资料：训练 16 / 16 条，独立验证 0 / 2 条。" } });
  await app.startLearning();
  expect(app.messages.pop()).toBe("资料不足");
  app.update({ runtime: { state: "ready" } });
  await app.startLearning();
  expect(app.messages.pop()).toBe("正在统计合格新知识，请稍后重试。");
  app.update(evolutionStatus(16, "stopped"));
  await app.startLearning();
  expect(app.messages.pop()).toBe("请先启动模型，待就绪后开始学习。");
  app.update({ ...evolutionStatus(16), automatic: { ...evolutionStatus(16).automatic,
    state: "waiting_validation", reason: LEARNING_SOURCE_RECHECK_NOTICE } });
  await app.startLearning();
  expect(app.messages.pop()).toBe("学习暂不可用：引擎更新后需要重新验证。");
  expect(app.requests).toEqual([]);
});

test("manual learning works with automatic learning off and duplicate clicks never create a second request", async () => {
  let complete!: () => void;
  const app = learningController({ ...evolutionStatus(16), automatic: { ...evolutionStatus(16).automatic,
    learning_enabled: false } }, () => new Promise<void>(resolve => { complete = resolve; }));
  const first = app.startLearning();
  expect(app.get("learning-start").disabled).toBe(true);
  await app.startLearning();
  expect(app.requests).toEqual([{ path: "/api/learn", body: {}, method: undefined }]);
  complete(); await first;
  expect(app.get("learning-start").disabled).toBe(false);
  app.update({ ...evolutionStatus(16), runtime: { state: "ready", job: { state: "running" } } });
  await app.startLearning();
  expect(app.requests).toHaveLength(1);
});

test("automatic learning switch saves only its setting and rolls back after a failed save", async () => {
  const app = learningController(evolutionStatus(16));
  app.get("learning-automatic").checked = false;
  await app.saveAutomaticLearning();
  expect(app.requests).toEqual([{ path: "/api/settings", body: { automatic: { learning_enabled: false } }, method: "PATCH" }]);
  expect(app.get("learning-automatic").checked).toBe(false);
  const failed = learningController(evolutionStatus(16), () => { throw new Error("save failed"); });
  failed.get("learning-automatic").checked = false;
  await expect(failed.saveAutomaticLearning()).rejects.toThrow("save failed");
  expect(failed.get("learning-automatic").checked).toBe(true);
  expect(failed.get("learning-automatic").disabled).toBe(false);
  const html = readFileSync(new URL("../frontend/index.html", import.meta.url), "utf8");
  expect(html).toContain('id="learning-automatic" type="checkbox" role="switch" checked');
  expect(html.indexOf('id="learning-start"')).toBeLessThan(html.indexOf('id="open-feeding"'));
});
