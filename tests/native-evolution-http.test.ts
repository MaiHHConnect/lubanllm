import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBackend } from "../backend/server.ts";
import { Settings } from "../backend/settings.ts";
import { NativeEvolutionService, type NativeEvolutionRoundResult } from "../backend/native-evolution-service.ts";

function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done; }); return { promise, resolve }; }
async function fixture(closeGate?: Promise<void>, inspectNativeEvolution?: any) {
  const directory = mkdtempSync(join(tmpdir(), "native-evolution-http-"));
  const settings = new Settings(join(directory, "home")); settings.update({ storage_root: join(directory, "storage") });
  const calls: string[] = [], entered = deferred<void>(), final = deferred<NativeEvolutionRoundResult>();
  let progress!: (value: { step: number; phase: string }) => void, ordinaryJob: unknown = null;
  const computeToken = "private-native-compute-test-key";
  const compute = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
    expect(request.headers.get("authorization")).toBe("Bearer " + computeToken);
    const path = new URL(request.url).pathname; calls.push(path);
    if (path.endsWith("/acquire")) {
      const body = await request.json() as any;
      return Response.json({ lease_id: body.lease_id, was_ready: true, identity: "a".repeat(64), root: directory,
        store: join(directory, "modules/luban0-v1"), model: join(directory, "model"), activeCore: null });
    }
    if (path.endsWith("/release")) return Response.json({ released: true, was_ready: true });
    if (path === "/api/status") return Response.json({ runtime: { state: "stopped", job: ordinaryJob },
      automatic: { state: "paused", reason: "stale paused reason", learning_count: 4, knowledge_learned: 9 } });
    if (path === "/api/model/reload") return Response.json({ state: "loading" });
    if (path === "/api/settings" && request.method === "PATCH") return Response.json(settings.update(await request.json()));
    if (path === "/internal/data-scope") return Response.json({ root: settings.snapshot().storage_root,
      store: join(directory, "modules/luban0-v1"), identity: "a".repeat(64) });
    if (path === "/api/learn") { ordinaryJob = { id: "python-next", state: "running", kind: "learning" }; return Response.json({ job_id: "python-next" }); }
    return Response.json({ ok: true });
  } });
  const computeURL = `http://127.0.0.1:${compute.port}`;
  const nativeEvolution = new NativeEvolutionService({
    compute: (path, body) => fetch(computeURL + path, { method: "POST", headers: { authorization: "Bearer " + computeToken, "content-type": "application/json" }, body: JSON.stringify(body) }),
    async open() { return { async runRound(input) {
      progress = input.onProgress; entered.resolve();
      return Promise.race([final.promise, new Promise<never>((_, reject) => {
        if (input.signal.aborted) reject(input.signal.reason);
        else input.signal.addEventListener("abort", () => reject(input.signal.reason), { once: true });
      })]);
    }, async close() { if (closeGate) await closeGate; calls.push("native_closed"); } }; },
  });
  const backend = createBackend({ settings, port: 8765, computeURL, computeToken, rpcToken: "native-rpc-key",
    webRoot: directory, version: "fixture", build: 1, nativeEvolution, inspectNativeEvolution });
  const request = (path: string, method = "GET", body?: unknown, authorized = true) => backend.fetch(new Request("http://127.0.0.1:8765" + path, {
    method, headers: { ...(authorized ? path.startsWith("/v1/") ? { authorization: "Bearer " + settings.secret("access_key") } : { cookie: "luban_session=" + settings.secret("ui_token") } : {}),
      "content-type": "application/json" }, body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body),
  }));
  const rpc = (authorized: boolean) => backend.fetch(new Request("http://127.0.0.1:8765/internal/rpc", {
    method: "POST", headers: { authorization: "Bearer " + (authorized ? "native-rpc-key" : "wrong"), "content-type": "application/json" },
    body: JSON.stringify({ target: "native_evolution", method: "tick", args: [], kwargs: {},
      root: settings.snapshot().storage_root, store: join(directory, "modules/luban0-v1"), identity: "a".repeat(64) }),
  }));
  return { calls, settings, nativeEvolution, request, rpc, entered, final, progress: (step: number) => progress({ step, phase: "training" }),
    async close() { await nativeEvolution.close(); backend.close(); compute.stop(true); rmSync(directory, { recursive: true, force: true }); } };
}

test("the real public start route authenticates and accepts only the existing empty body", async () => {
  const f = await fixture();
  try {
    expect((await f.request("/api/evolution/start", "POST", {}, false)).status).toBe(401);
    for (const body of [undefined, "", "[]", { steps: 1 }, { continuous: false }]) expect((await f.request("/api/evolution/start", "POST", body)).status).toBe(400);
    expect(f.calls).toEqual([]);
    const response = await f.request("/api/evolution/start", "POST", {});
    expect(response.status).toBe(200); expect((await response.json() as any).job_id).toMatch(/^[a-f0-9]{12}$/);
    await f.entered.promise; expect(f.calls).toContain("/internal/native-evolution/acquire");
    expect(f.calls).not.toContain("/api/evolution/start");
  } finally { await f.close(); }
});

test("unqualified runtime blocks public start and private automatic tick before model handoff", async () => {
  const f = await fixture(undefined, async () => { throw new Error("原生进化运行程序尚未完成接线验收"); });
  try {
    expect((await f.rpc(false)).status).toBe(401);
    const tick = await (await f.rpc(true)).json() as any;
    expect(tick.ok).toBe(true); expect(tick.result.state).toBe("waiting_validation");
    const start = await f.request("/api/evolution/start", "POST", {});
    expect(start.status).toBe(400); expect((await start.json() as any).error.message).toContain("尚未完成接线验收");
    expect(f.nativeEvolution.status().state).toBe("idle");
    expect(f.calls).toEqual(["/internal/data-scope"]);
  } finally { await f.close(); }
});

test("native progress reaches status and blocks mutations/chat while read routes remain available", async () => {
  const f = await fixture();
  try {
    await f.request("/api/evolution/start", "POST", {}); await f.entered.promise; f.progress(16);
    const status = await (await f.request("/api/status")).json() as any;
    expect(status.runtime.state).toBe("evolving");
    expect(status.runtime.job).toMatchObject({ state: "running", kind: "evolution", step: 16, steps: 450, round: 1 });
    expect(status.automatic).toEqual({ state: "running", reason: "", learning_count: 4, knowledge_learned: 9 });
    expect((await f.request("/api/settings")).status).toBe(200);
    for (const [path, method] of [["/api/settings", "PATCH"], ["/api/model/reload", "POST"], ["/api/model/stop", "POST"],
      ["/api/chat", "POST"], ["/v1/chat/completions", "POST"], ["/api/samples", "POST"], ["/api/evolution/start", "POST"]]) {
      expect((await f.request(path!, method!, {})).status).toBe(409); expect(f.calls).not.toContain(path!);
    }
    const close = await (await f.request("/api/automatic/close-status")).json() as any;
    expect(close.running).toBe(true); expect(close.resumable).toBe(false);
  } finally { await f.close(); }
});

test("native busy status mirrors queued and cleanup errors without losing learning statistics", async () => {
  const f = await fixture(), original = f.nativeEvolution.status.bind(f.nativeEvolution);
  try {
    await f.request("/api/evolution/start", "POST", {}); await f.entered.promise;
    for (const [nativeState, expected, failure] of [["preparing", "queued", null], ["quarantined", "error", "restore failed"]] as const) {
      f.nativeEvolution.status = () => ({ ...original(), state: nativeState, running: false, lease_held: true, cleanup_error: failure });
      const calls = f.calls.length, response = await (await f.request("/api/status")).json() as any;
      expect(response.runtime.job.state).toBe(expected);
      expect(response.automatic).toEqual({ state: expected, reason: failure ?? "", learning_count: 4, knowledge_learned: 9 });
      expect(f.calls.slice(calls)).toEqual(["/api/status"]);
    }
  } finally { f.nativeEvolution.status = original; await f.close(); }
});

test("active native lease permits only the isolated automatic learning switch without stopping its job", async () => {
  const f = await fixture();
  try {
    await f.request("/api/evolution/start", "POST", {}); await f.entered.promise; f.progress(7);
    const before = f.nativeEvolution.status();
    expect((await f.request("/api/settings", "PATCH", { automatic: { learning_enabled: false } }, false)).status).toBe(401);
    for (const learning_enabled of [false, true, false]) {
      expect((await f.request("/api/settings", "PATCH", { automatic: { learning_enabled } })).status).toBe(200);
      expect(new Settings(f.settings.home).snapshot().automatic.learning_enabled).toBe(learning_enabled);
      expect(f.nativeEvolution.status()).toMatchObject({ id: before.id, running: true, step: 7 });
    }
    for (const body of [{ automatic: { learning_enabled: false }, temperature: .5 },
      { automatic: { learning_enabled: false, mode: "auto", scheduled_time: "02:00" } },
      { automatic: { learning_enabled: 0 } }, { automatic: { learning_enabled: "false" } },
      { teacher: { learning_enabled: false } }]) {
      expect((await f.request("/api/settings", "PATCH", body)).status).toBe(409);
    }
    expect(f.calls.filter(path => path === "/api/settings")).toHaveLength(3);
    expect(f.calls).not.toContain("/internal/native-evolution/release");
    expect(f.calls).not.toContain("/api/model/reload");
  } finally { await f.close(); }
});

test("existing cancel endpoint cancels native execution, waits for cleanup, then restores ordinary inference", async () => {
  const f = await fixture();
  try {
    await f.request("/api/evolution/start", "POST", {}); await f.entered.promise;
    expect((await f.request("/api/learn/cancel", "POST", {})).status).toBe(200);
    expect((await f.nativeEvolution.wait()).state).toBe("paused");
    expect(f.calls.indexOf("native_closed")).toBeLessThan(f.calls.indexOf("/internal/native-evolution/release"));
    expect(f.calls.indexOf("/internal/native-evolution/release")).toBeLessThan(f.calls.indexOf("/api/model/reload"));
    expect(f.calls).not.toContain("/api/learn/cancel");
    const status = await (await f.request("/api/status")).json() as any;
    expect(status.runtime.job.state).toBe("paused");
  } finally { await f.close(); }
});

test("awaiting quality remains unaccepted, and subsequent Python work supersedes the terminal native job", async () => {
  const f = await fixture();
  try {
    await f.request("/api/evolution/start", "POST", {}); await f.entered.promise;
    f.final.resolve({ step: 450, phase: "awaiting_quality", quality_checked: false, adopted: false, pending: false });
    await f.nativeEvolution.wait();
    let status = await (await f.request("/api/status")).json() as any;
    expect(status.runtime.job.state).toBe("rejected"); expect(status.runtime.job.result.promoted).toBe(false);
    await f.request("/api/learn", "POST", { steps: 16 });
    status = await (await f.request("/api/status")).json() as any;
    expect(status.runtime.job.id).toBe("python-next");
  } finally { await f.close(); }
});

test("the macOS quit handshake returns success only after native teardown and lease release", async () => {
  const closed = deferred<void>(), f = await fixture(closed.promise);
  try {
    await f.request("/api/evolution/start", "POST", {}); await f.entered.promise;
    let returned = false;
    const pause = f.request("/api/automatic/pause", "POST").then(response => { returned = true; return response; });
    await Promise.resolve(); await Promise.resolve();
    expect(returned).toBe(false); expect(f.calls).not.toContain("/internal/native-evolution/release");
    closed.resolve(); expect((await pause).status).toBe(200);
    expect(f.nativeEvolution.status().lease_held).toBe(false);
  } finally { closed.resolve(); await f.close(); }
});

test("public expert learning uses native handoff and rejects custom budgets before dispatch", async () => {
  const sample = (id: string, purpose: string) => ({ id, prompt: id, answer: "ok", purpose, approved: true, training_allowed: true });
  const f = await fixture(undefined, async () => ({ material_kind: "expert", core: "core-abcdef123456", expert: null,
    records: { train: Array.from({ length: 16 }, (_, i) => sample("train" + i, "training")),
      valid: [sample("valid0", "validation"), sample("valid1", "validation")], protection: [] },
    steps: 192, material_target: 16, arguments: { steps: 64 }, execution_ready: true }));
  try {
    expect((await f.request("/api/learn", "POST", { steps: 450 })).status).toBe(400);
    expect((await f.request("/api/learn", "POST", {}, false)).status).toBe(401);
    expect((await f.request("/api/learn", "POST", {})).status).toBe(200);
    await f.entered.promise;
    expect(f.calls).toContain("/internal/native-evolution/acquire");
    expect(f.calls).not.toContain("/api/learn");
  } finally { await f.close(); }
});
