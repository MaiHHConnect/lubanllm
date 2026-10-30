import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { homeLedger, modelName, readModelName, saveModelName } from "../backend/home";
import { Memory } from "../backend/memory";
import { Settings } from "../backend/settings";
import { createBackend } from "../backend/server";

const identity = "a".repeat(64), other = "b".repeat(64);
let root: string, memory: Memory;
beforeEach(() => { root = mkdtempSync(join(tmpdir(), "luban-home-test-")); memory = new Memory(join(root, "data")); });
afterEach(() => { memory.close(); rmSync(root, { recursive: true, force: true }); });

test("nickname validates Unicode length, isolates base identities and persists privately", () => {
  const settings = new Settings(join(root, "private"));
  expect(readModelName(settings.home, identity)).toBe("luban0");
  expect(saveModelName(settings.home, identity, "  小鲁班 🚀  ")).toBe("小鲁班 🚀");
  expect(readModelName(settings.home, identity)).toBe("小鲁班 🚀");
  expect(readModelName(settings.home, other)).toBe("luban0");
  expect(modelName("鲁".repeat(32))).toBe("鲁".repeat(32));
  for (const value of [null, {}, "", " ", "鲁".repeat(33), "a\nb", "a\u202eb", "a\ud800b"]) expect(() => modelName(value)).toThrow();
  expect(statSync(join(settings.home, "model-names.json")).mode & 0o777).toBe(0o600);
  expect(readFileSync(join(settings.home, "settings.json"), "utf8")).not.toContain("小鲁班");
  const alias = join(root, "name-alias"); symlinkSync(join(settings.home, "model-names.json"), alias);
  rmSync(join(settings.home, "model-names.json")); symlinkSync(alias, join(settings.home, "model-names.json"));
  expect(() => readModelName(settings.home, identity)).toThrow();
});

test("daily/weekly counters use recorded current-base calls and attributed feedback, never pending materials or aggregate usage", async () => {
  const now = new Date(2026, 9, 11, 12), today = new Date(2026, 9, 11, 0).getTime() / 1000;
  memory.connect(db => {
    const insert = db.query("INSERT INTO events VALUES (?,?,?,?)");
    const call = (id: string, base: string, route: string, created: number) => insert.run(id, "decision", JSON.stringify({ base_identity: base, decision: { route } }), created);
    call("today-local", identity, "local", today + 1);
    call("week-local", identity, "local", today - 86400);
    call("today-teacher", identity, "teacher", today + 2);
    call("other-base", other, "local", today + 3);
    call("last-week", identity, "local", today - 86400 * 8);
    call("future", identity, "local", today + 86400);
    const opinion = (id: string, candidate: string, correct: boolean | null, status = "complete") => insert.run(id, "teacher_review", JSON.stringify({ candidate_id: candidate, status, result: { correct } }), today + 4);
    opinion("false1", "route:today-local", false); opinion("false2", "route:today-local", false);
    opinion("correct", "route:today-teacher", true);
    opinion("other-opinion", "route:other-base", false);
    opinion("unknown", "route:week-local", null, "pending");
    opinion("failed", "route:week-local", false, "failed");
    opinion("no-evidence", "route:missing", false);
  });
  memory.record_usage("local", { total_tokens: 500 });
  memory.capture_teacher([{ role: "user", content: "No evidence this is a gap" }], "pending", {}, false);
  const ledger = await homeLedger(memory, identity, now);
  expect(ledger.today).toEqual({ local_calls: 1, teacher_calls: 1, feedback_received: 2, teacher_flagged_gaps: 1, corrections: null, mastered: null });
  expect(ledger.week.local_calls).toBe(2);
  expect(ledger.application_attribution).toBe(false);
  expect((await homeLedger(memory, null, now)).today.local_calls).toBe(0);
});

test("personal evolution excludes factory identity, research, rejected, uncommitted, foreign and missing-event runs", async () => {
  memory.connect(db => {
    db.exec("CREATE TABLE training_runs (id TEXT, identity TEXT, kind TEXT, state TEXT, body TEXT, updated REAL)");
    const run = (id: string, report = {}, changes: any = {}, base = identity, event = true) => {
      const body = { scope: "product", commit: { report: { identity: base, promoted: true, publication_state: "published", version: "core-" + id, ...report } }, ...changes };
      db.query("INSERT INTO training_runs VALUES (?,?,?,?,?,?)").run(id, base, "evolution", changes.state ?? "committed", JSON.stringify(body), 1);
      if (event) db.query("INSERT INTO events VALUES (?,?,?,?)").run("training:" + id, "evolution", JSON.stringify(body.commit.report), 1);
    };
    run("factory", { foundation_install: true, training_updates_executed: 0 });
    run("original-factory", { native_identity_verified: true });
    run("good"); run("foreign", {}, {}, other); run("research", {}, { scope: "research" });
    run("pending", {}, { state: "committing" }); run("rejected", { promoted: false });
    run("no-event", {}, {}, identity, false);
  });
  const ledger = await homeLedger(memory, identity);
  expect(ledger.evolution_count).toBe(1);
  expect(ledger.recent.some(row => row.id.includes("factory"))).toBe(false);
  expect(ledger.recent.find(row => row.id === "training:pending")?.promoted).toBe(false);
  expect(ledger.recent.find(row => row.id === "training:research")?.promoted).toBe(false);
});

test("real public home routes enforce session/Origin and verified current-base rename without changing model id", async () => {
  const settings = new Settings(join(root, "private")); settings.update({ storage_root: memory.root });
  let current = identity, verifiedCount = 0, verifiedIdentity = identity;
  const received: string[] = [];
  const compute = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    expect(request.headers.get("authorization")).toBe("Bearer private-compute");
    const path = new URL(request.url).pathname; received.push(path);
    if (path === "/api/health") return Response.json({ state: "stopped" });
    if (["/internal/data-scope", "/internal/home-profile"].includes(path)) return Response.json({ root: memory.root, identity: current, growth_profile: [{ domain: "mathematics", growth: null }] });
    if (path === "/api/status") return Response.json({ runtime: { language_parameters: 234567 }, automatic: { evolution_count: 999 } });
    if (path === "/internal/home-evolution") return Response.json({ root: memory.root, identity: verifiedIdentity, evolution_count: verifiedCount });
    throw new Error("unexpected compute call " + path);
  } });
  const backend = createBackend({ settings, port: 8765, computeURL: compute.url.origin, computeToken: "private-compute", rpcToken: "private-rpc", webRoot: root, version: "fixture", build: 1 });
  const request = (path: string, method = "GET", body?: any, extra = {}) => backend.fetch(new Request("http://127.0.0.1:8765" + path, { method,
    headers: { cookie: "luban_session=" + settings.secret("ui_token"), ...extra }, body: body === undefined ? undefined : JSON.stringify(body) }));
  try {
    expect((await backend.fetch(new Request("http://127.0.0.1:8765/api/home"))).status).toBe(401);
    expect((await request("/api/home", "GET", undefined, { origin: "https://evil.example" })).status).toBe(403);
    const home = await (await request("/api/home")).json() as any;
    expect(home.model).toEqual({ id: "luban0", name: "luban0", can_rename: false, evolution_count: 0, core_parameters: 234567 });
    expect(home.growth_profile).toEqual([{ domain: "mathematics", growth: null }]);
    expect((await request("/api/home/model-name", "PATCH", { name: "名字", evolution_count: 100 })).status).toBe(409);
    verifiedCount = 1; verifiedIdentity = other;
    expect((await request("/api/home/model-name", "PATCH", { name: "名字" })).status).toBe(409);
    verifiedIdentity = identity;
    expect((await (await request("/api/home/model-name", "PATCH", { name: "名字" })).json()) as any).toEqual({ id: "luban0", name: "名字" });
    expect((await (await request("/v1/models", "GET", undefined, { authorization: "Bearer " + settings.secret("access_key") })).json()) as any).toEqual({ object: "list", data: [{ id: "luban0", object: "model", owned_by: "maihaohuo" }] });
    current = other;
    expect(((await (await request("/api/home")).json()) as any).model.name).toBe("luban0");
    expect(received.every(path => ["/api/health", "/api/status", "/internal/data-scope", "/internal/home-profile", "/internal/home-evolution"].includes(path))).toBe(true);
    expect((await request("/internal/home-evolution")).status).toBe(404);
  } finally { backend.close(); compute.stop(true); }
});

test("loaded busy models keep status, home and conversations readable; startup remains guarded", async () => {
  const settings = new Settings(join(root, "private")); settings.update({ storage_root: memory.root });
  let state = "ready";
  const conversation = memory.save_conversation(null, [{ role: "user", content: "hello" }]);
  const compute = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    expect(request.headers.get("authorization")).toBe("Bearer private-compute");
    expect(request.method).toBe("GET");
    const path = new URL(request.url).pathname;
    if (path === "/api/health") return Response.json({ state });
    if (["/internal/data-scope", "/internal/home-profile"].includes(path)) return Response.json({ root: memory.root, identity });
    if (path === "/api/status") return Response.json({ runtime: { state, language_parameters: 234567, context: { configured_limit: 262144 } } });
    throw new Error("unexpected compute call " + path);
  } });
  const backend = createBackend({ settings, port: 8765, computeURL: compute.url.origin, computeToken: "private-compute", rpcToken: "private-rpc", webRoot: root, version: "fixture", build: 1 });
  const request = (path: string) => backend.fetch(new Request("http://127.0.0.1:8765" + path, {
    headers: { cookie: "luban_session=" + settings.secret("ui_token") },
  }));
  try {
    for (state of ["ready", "generating", "learning", "evolving", "evaluating"]) {
      const status = await (await request("/api/status")).json() as any;
      expect(status.runtime.state).toBe(state);
      expect(status.runtime.context.configured_limit).toBe(262144);
      const home = await request("/api/home"); expect(home.status).toBe(200);
      expect(((await home.json()) as any).model.core_parameters).toBe(234567);
      const list = await request("/api/conversations"); expect(list.status).toBe(200);
      expect(((await list.json()) as any).conversations.map((row: any) => row.id)).toContain(conversation);
      const detail = await request("/api/conversations/" + conversation); expect(detail.status).toBe(200);
      expect(((await detail.json()) as any).messages).toEqual([{ role: "user", content: "hello" }]);
    }
    for (state of ["loading", "error", "unknown"]) {
      for (const path of ["/api/home", "/api/conversations", "/api/conversations/" + conversation]) expect((await request(path)).status).toBe(409);
      expect(((await (await request("/api/status")).json()) as any).runtime.context).toBeNull();
    }
    state = "stopped";
    expect((await request("/api/home")).status).toBe(200);
    expect((await request("/api/conversations")).status).toBe(409);
  } finally { backend.close(); compute.stop(true); }
});
