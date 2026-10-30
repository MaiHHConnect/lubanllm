import { expect, test } from "bun:test";
import { initializeHome, modelStartPresentation } from "../frontend/home";

test("model start shows immediate busy state and keeps polling from enabling duplicate requests", () => {
  for (const state of ["stopped", "ready", "error"]) {
    const value = modelStartPresentation(state, false, true);
    expect(value).toEqual({ state: "loading", disabled: true, label: "启动中", detail: "正在加载模型…" });
  }
  expect(modelStartPresentation("loading", false, false).disabled).toBe(true);
});
test("start returns genuine ready, failed, busy and stopped actions", () => {
  expect(modelStartPresentation("ready", false, false)).toMatchObject({ label: "停止模型", detail: "模型已启动", disabled: false });
  expect(modelStartPresentation("ready", false, "stopping")).toEqual({ state: "stopping", label: "停止中", detail: "正在停止模型…", disabled: true });
  expect(modelStartPresentation("ready", true, false).disabled).toBe(true);
  expect(modelStartPresentation("error", false, false)).toMatchObject({ label: "重新启动", detail: "启动失败，点击重试", disabled: false });
  expect(modelStartPresentation("stopped", false, false)).toMatchObject({ label: "启动模型", disabled: false });
  for (const state of ["generating", "evolving", "training", "evaluating"]) expect(modelStartPresentation(state, false, false).disabled).toBe(true);
});

async function bundledStartupFixture() {
  const source = await Bun.file(new URL("../frontend/app.ts", import.meta.url)).text();
  const transpile = (start: string, end: string) => new Bun.Transpiler({ loader: "ts" }).transformSync(source.slice(source.indexOf(start), source.indexOf(end)));
  const automatic = transpile("async function prepareBundledStartup()", "function applySettings(");
  const manual = transpile("async function changeModelState(", '$("reload-model").addEventListener("click"');
  return new Function(`
    let bundledStartupBusy=false, bundledStartupAttempted=false, modelStarting=false, startupReadyRequested=false;
    let agreement={accepted:false}, status={runtime:{state:'stopped',job:null}}, backupRestoreBusy=false;
    let installed=true, preparations=0, prepare=async()=>{ preparations++; return installed; }, failStart=false;
    const calls=[], errors=[], control={disabled:false};
    const button=()=>control, $=()=>({hidden:true}), renderStatus=()=>{}, loadVersions=async()=>{};
    const modelTools={prepareBundledModels:()=>prepare()};
    const api=async(path)=>{calls.push(path); if(failStart)throw Error('startup failed'); status.runtime.state=path.endsWith('/stop')?'stopped':'ready';};
    const pollStatus=async()=>prepareBundledStartup();
    const operation=async(target, action)=>{try{await action();}catch(error){errors.push(error);}};
    ${automatic}
    ${manual}
    return {automatic:prepareBundledStartup,manual:changeModelState,calls,errors,
      configure(value){if('accepted'in value)agreement.accepted=value.accepted;if('state'in value)status.runtime.state=value.state;
        if('job'in value)status.runtime.job=value.job;if('installed'in value)installed=value.installed;
        if('prepare'in value)prepare=value.prepare;if('failStart'in value)failStart=value.failStart;},
      get preparations(){return preparations;},get readyDialog(){return startupReadyRequested;}};
  `)() as {
    automatic: () => Promise<void>; manual: (stopping: boolean) => Promise<void>; calls: string[]; errors: Error[];
    configure: (value: Record<string, unknown>) => void; preparations: number; readyDialog: boolean;
  };
}

test("complete startup waits for agreement and installed models, then starts once without a connection dialog", async () => {
  const app = await bundledStartupFixture();
  await app.automatic();
  expect(app.preparations).toBe(0); expect(app.calls).toEqual([]);
  app.configure({ accepted: true, installed: false });
  await app.automatic(); expect(app.calls).toEqual([]);
  app.configure({ installed: true, job: { state: "running" } });
  await app.automatic(); expect(app.calls).toEqual([]);
  app.configure({ job: null });
  await app.automatic(); await app.automatic();
  expect(app.calls).toEqual(["/api/model/reload"]);
  expect(app.readyDialog).toBe(false);
  await app.manual(true); await app.automatic();
  expect(app.calls).toEqual(["/api/model/reload", "/api/model/stop"]);
  const reopened = await bundledStartupFixture();
  reopened.configure({ accepted: true }); await reopened.automatic();
  expect(reopened.calls).toEqual(["/api/model/reload"]);
});

test("automatic start cannot duplicate an in-flight preparation or retry failed loading", async () => {
  const app = await bundledStartupFixture();
  let release!: (value: boolean) => void;
  app.configure({ accepted: true, prepare: () => new Promise<boolean>(resolve => { release = resolve; }) });
  const pending = app.automatic();
  await app.automatic(); release(true); await pending;
  expect(app.calls).toEqual(["/api/model/reload"]);
  const failed = await bundledStartupFixture();
  failed.configure({ accepted: true, failStart: true });
  await failed.automatic(); await failed.automatic();
  expect(failed.calls).toEqual(["/api/model/reload"]);
  expect(failed.errors.map(error => error.message)).toEqual(["startup failed"]);
  failed.configure({ failStart: false }); await failed.manual(false);
  expect(failed.calls).toEqual(["/api/model/reload", "/api/model/reload"]);
  expect(failed.readyDialog).toBe(true);
});

test("connection protocol persists locally and keeps Claude SDK base distinct from OpenAI guides", async () => {
  const previousDocument = globalThis.document, previousStorage = globalThis.localStorage;
  const nodes = new Map<string, any>(), saved = new Map([["luban.connection-protocol", "claude"]]);
  const makeNode = () => ({ value: "", textContent: "", dataset: {}, hidden: false,
    listeners: {} as Record<string, () => void>, classList: { toggle() {} },
    append() {}, replaceChildren() {}, setAttribute() {}, getAttribute() { return ""; },
    addEventListener(event: string, callback: () => void) { this.listeners[event] = callback; },
  });
  const node = (id: string) => { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id)!; };
  globalThis.document = { getElementById: node, createElement: makeNode } as unknown as Document;
  globalThis.localStorage = { getItem: (key: string) => saved.get(key) ?? null, setItem: (key: string, value: string) => { saved.set(key, value); } } as Storage;
  const copies: string[] = [], calls: string[] = [];
  try {
    const home = initializeHome({ api: async <T>(path: string) => { calls.push(path); return { key: "test-key" } as T; },
      failure: error => { throw error; }, toast() {}, copy: async value => { copies.push(value); } });
    expect(node("connection-protocol").value).toBe("claude");
    expect(node("home-api-base").textContent).toBe("http://127.0.0.1:8765");
    home.renderRuntime({ state: "ready" }, "http://127.0.0.1:9000/v1");
    expect(node("home-api-base").textContent).toBe("http://127.0.0.1:9000");
    node("home-copy-connection").listeners.click(); await Bun.sleep(0);
    expect(copies[0]).toBe("协议：Claude Messages\nAPI 地址：http://127.0.0.1:9000\nAPI 密钥：test-key\n模型 ID：luban0");
    node("connection-protocol").value = "openai"; node("connection-protocol").listeners.change();
    expect(saved.get("luban.connection-protocol")).toBe("openai");
    expect(node("home-api-base").textContent).toBe("http://127.0.0.1:9000/v1");
    node("home-copy-connection").listeners.click(); await Bun.sleep(0);
    expect(copies[1]).toContain("OpenAI Chat Completions\nAPI 地址：http://127.0.0.1:9000/v1");
    expect(calls).toEqual(["/api/access-key", "/api/access-key"]);
    const html = await Bun.file(new URL("../frontend/index.html", import.meta.url)).text();
    expect(html).toContain('for="connection-protocol"');
    expect(html).toContain("以下应用教程使用 OpenAI Chat Completions。");
    expect(html).toContain("OpenAI 与 Claude 均支持文字与工具调用，工具由连接的应用执行。");
  } finally {
    globalThis.document = previousDocument; globalThis.localStorage = previousStorage;
  }
});
