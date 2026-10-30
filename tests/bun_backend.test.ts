import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createBackend } from "../backend/server";
import { Settings } from "../backend/settings";
import { decodeWire, encodeWire } from "../backend/wire";

let directory: string, settings: Settings, backend: ReturnType<typeof createBackend>, server: ReturnType<typeof Bun.serve>;
let oracle: any, worker: any, oracleInfo: any, workerInfo: any, publicURL: string;
const rpcToken = "fixture-private-rpc-key", computeToken = "fixture-private-compute-key";
const processes: any[] = [];
async function reference(mode: "oracle" | "private", home: string, environment = {}) {
  const process = Bun.spawn([resolve(".venv/bin/python"), resolve("tests/bun_backend_reference.py"), "--mode", mode, "--home", home], {
    env: { ...globalThis.process.env, PYTHONPATH: resolve("src") + ":" + resolve("tests"),
      LUBAN_BUN_RPC_URL: "", LUBAN_BUN_RPC_TOKEN: "", LUBAN_COMPUTE_TOKEN: "", ...environment },
    stdout: "pipe", stderr: "pipe",
  });
  processes.push(process);
  const reader = process.stdout.getReader(); let text = "";
  const deadline = AbortSignal.timeout(20000);
  while (!text.includes("\n")) {
    const result: any = await Promise.race([reader.read(), new Promise((_, reject) => deadline.addEventListener("abort", () => reject(new Error("Python fixture startup timeout")), { once: true }))]);
    if (result.done) throw new Error("Python fixture exited: " + await new Response(process.stderr).text());
    text += new TextDecoder().decode(result.value);
  }
  reader.releaseLock();
  return { process, info: JSON.parse(text.split("\n")[0]!) };
}
async function ready(url: string, headers: Record<string, string> = {}) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try { if ((await fetch(url + "/api/health", { headers })).ok) return; } catch {}
    await Bun.sleep(25);
  }
  throw new Error("HTTP fixture never became healthy");
}
function headers(side: "bun" | "oracle", path: string) {
  const credentials = side === "bun" ? { ui_token: settings.secret("ui_token"), access_key: settings.secret("access_key") } : oracleInfo;
  return path.startsWith("/v1/") ? { authorization: "Bearer " + credentials.access_key } : { cookie: "luban_session=" + credentials.ui_token };
}
async function request(side: "bun" | "oracle", path: string, method = "GET", body?: any, extra: Record<string, string> = {}) {
  return fetch((side === "bun" ? publicURL : `http://127.0.0.1:${oracleInfo.port}`) + path, {
    method, headers: { ...headers(side, path), ...(body === undefined ? {} : { "content-type": "application/json" }), ...extra },
    body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body),
  });
}
async function rpc(method: string, args: any[] = [], kwargs = {}, transaction?: string, target = "memory") {
  const payload = { target, method, args, kwargs, transaction, identity: null,
    root: settings.snapshot().storage_root, home: settings.home };
  return decodeWire(await (await fetch(publicURL + "/internal/rpc", { method: "POST", headers: {
    authorization: "Bearer " + rpcToken, "content-type": "application/json" }, body: JSON.stringify(encodeWire(payload)) })).json());
}
function normalize(value: any): any {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !["id", "created", "updated", "accepted_at", "started", "pid"].includes(key))
    .map(([key, item]) => [key, normalize(item)]));
  if (typeof value === "string") {
    if (value.startsWith("{") || value.startsWith("[")) { try { return normalize(JSON.parse(value)); } catch {} }
    return value.replaceAll(settings.snapshot().storage_root, "<storage>").replaceAll(join(directory, "oracle-storage"), "<storage>")
      .replaceAll(settings.home, "<home>").replaceAll(join(directory, "oracle-home"), "<home>")
      .replace(/http:\/\/127\.0\.0\.1:\d+/g, "http://127.0.0.1:<port>");
  }
  return value;
}
async function differential(path: string, method = "GET", body?: any, extra = {}) {
  const pair = await Promise.all([request("bun", path, method, body, extra), request("oracle", path, method, body, extra)]);
  expect(pair[0]!.status, method + " " + path).toBe(pair[1]!.status);
  const data = await Promise.all(pair.map(response => response.json()));
  expect(normalize(data[0]), method + " " + path).toEqual(normalize(data[1]));
  return { status: pair[0]!.status, data };
}

beforeAll(async () => {
  directory = mkdtempSync(join(tmpdir(), "luban-public-http-"));
  settings = new Settings(join(directory, "bun-home")); settings.update({ storage_root: join(directory, "bun-storage") });
  const original = new Settings(join(directory, "oracle-home")); original.update({ storage_root: join(directory, "oracle-storage") });
  const webRoot = join(directory, "web"); mkdirSync(webRoot); writeFileSync(join(webRoot, "index.html"), "<html>isolated UI</html>");
  const first = await reference("oracle", original.home); oracle = first.process; oracleInfo = first.info;
  await ready(`http://127.0.0.1:${oracleInfo.port}`);
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response("starting", { status: 503 }) });
  publicURL = `http://127.0.0.1:${server.port}`;
  const options = { settings, port: server.port!, computeURL: "http://127.0.0.1:1", computeToken, rpcToken,
    webRoot, version: "0.3.10", build: 18 };
  const release = JSON.parse(readFileSync(resolve("release.json"), "utf8")); options.version = release.version; options.build = release.build;
  backend = createBackend(options); server.reload({ fetch: backend.fetch });
  const second = await reference("private", settings.home, { LUBAN_BUN_RPC_URL: publicURL + "/internal/rpc", LUBAN_BUN_RPC_TOKEN: rpcToken,
    LUBAN_COMPUTE_TOKEN: computeToken, LUBAN_PUBLIC_PORT: String(server.port) });
  worker = second.process; workerInfo = second.info; options.computeURL = `http://127.0.0.1:${workerInfo.port}`;
  await ready(options.computeURL, { authorization: "Bearer " + computeToken }); await ready(publicURL);
  writeFileSync(resolve("research/bun_backend_route_inventory.json"), JSON.stringify({
    oracle_route_count: oracleInfo.routes.length, private_compute_route_count: workerInfo.routes.length,
    routes: oracleInfo.routes, evidence_scope: "Real create_app routes; every route authenticated/strict-JSON boundary checked. Selected non-neural handlers and proxy flows checked with a CPU FakeRuntime. Model quality, real ASR and full training are not claimed.",
    private_compute_sqlite_connect_forbidden: true,
  }, null, 2) + "\n");
}, 45000);
afterAll(async () => {
  backend?.beginClose();
  for (const process of processes) process.kill("SIGTERM");
  await Promise.all(processes.map(async process => {
    await Promise.race([process.exited, Bun.sleep(10000).then(() => process.kill("SIGKILL"))]);
    await process.exited; expect(() => globalThis.process.kill(process.pid, 0)).toThrow();
  }));
  backend?.close(); server?.stop(true);
  if (directory) rmSync(directory, { recursive: true, force: true });
});

describe("actual Bun public HTTP and actual FastAPI CPU oracle", () => {
  test("teacher-first pending material resumes through real Bun RPC without a nested lease", async () => {
    const home = join(directory, "teacher-practice-home");
    const isolatedSettings = new Settings(home);
    isolatedSettings.update({ storage_root: join(directory, "teacher-practice-storage"), teacher: {
      base_url: "http://offline.invalid/v1", api_key: "offline-teacher-key", model: "fixture", answer_mode: "teacher_first",
    } });
    const isolatedServer = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response("starting", { status: 503 }) });
    const isolatedBackend = createBackend({ settings: isolatedSettings, port: isolatedServer.port!,
      computeURL: "http://127.0.0.1:1", computeToken, rpcToken, webRoot: join(directory, "web"), version: "fixture", build: 1 });
    isolatedServer.reload({ fetch: isolatedBackend.fetch });
    const fixture = Bun.spawn([resolve(".venv/bin/python"), resolve("tests/bun_teacher_practice_fixture.py"), "--home", home], {
      env: { ...process.env, PYTHONPATH: resolve("src") + ":" + resolve("tests"),
        LUBAN_BUN_RPC_URL: `http://127.0.0.1:${isolatedServer.port}/internal/rpc`, LUBAN_BUN_RPC_TOKEN: rpcToken,
        LUBAN_COMPUTE_TOKEN: computeToken }, stdout: "pipe", stderr: "pipe",
    });
    processes.push(fixture);
    const output = new Response(fixture.stdout).text(), errors = new Response(fixture.stderr).text();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const exit = await Promise.race([fixture.exited, new Promise<never>((_, reject) => {
        timer = setTimeout(() => { fixture.kill("SIGKILL"); reject(new Error("Teacher practice RPC deadlock or timeout")); }, 12000);
      })]);
      expect(exit, await errors).toBe(0);
      const result = JSON.parse((await output).trim());
      expect(result).toMatchObject({ remote_memory: true, python_sqlite_forbidden: true, teacher_calls: 2,
        material_id_preserved: true, split_preserved: true, sample_eligible_for_learning: false, state: "complete",
        teacher_usage: { calls: 2, unknown_calls: 1, prompt_tokens: null, completion_tokens: null, total_tokens: null } });
    } finally {
      clearTimeout(timer); if (fixture.exitCode === null) fixture.kill("SIGKILL");
      await fixture.exited; isolatedBackend.close(); isolatedServer.stop(true);
    }
  }, 20000);
  test("all product route identities survive the private split", async () => {
    expect(workerInfo.routes).toEqual(oracleInfo.routes);
    expect(oracleInfo.routes.length).toBeGreaterThanOrEqual(66);
    for (const entry of oracleInfo.routes as string[]) {
      const [method, template] = entry.split(" "); const path = template!.replace(/\{[^}]+\}/g, "missing-fixture");
      // Strict JSON and authentication are exercised against each real route.
      // This is a boundary check, not a claim that CPU fixtures run every model operation.
      if (["POST", "PATCH", "PUT"].includes(method!)) await differential(path, method, "[]");
      else if (!path.startsWith("/api/health")) {
        const pair = await Promise.all([fetch(publicURL + path, { method }), fetch(`http://127.0.0.1:${oracleInfo.port}` + path, { method })]);
        expect(pair[0]!.status, entry).toBe(pair[1]!.status);
        expect(pair[0]!.status, entry).toBe(401);
      }
    }
  });
  test("Host, Origin, cookie, Bearer and security headers", async () => {
    expect((await request("bun", "/api/settings", "GET", undefined, { host: "evil.example" })).status).toBe(403);
    await differential("/api/settings", "GET", undefined, { origin: "https://evil.example" });
    await differential("/api/settings", "GET", undefined, { cookie: "luban_session=wrong" });
    await differential("/v1/models", "GET", undefined, { authorization: "Bearer wrong" });
    const response = await request("bun", "/api/settings");
    expect(response.headers.get("cache-control")).toBe("no-store"); expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(response.headers.get("content-security-policy")).toContain("frame-ancestors 'none'");
  });
  test("authenticated compute status, catalogue and validation handlers match the oracle", async () => {
    for (const path of ["/api/health", "/api/version", "/api/status", "/api/experts", "/api/training/backups", "/api/model-packages",
      "/api/feeding", "/api/external-brain", "/api/automatic/close-status", "/api/learning/contents", "/api/evolution", "/api/evolution/status", "/api/model-quiz", "/api/leaderboard?page=0"]) {
      const result = await differential(path); expect(result.status, path).toBeLessThan(500);
    }
    for (const [path, body] of [["/api/feeding/scan", {}], ["/api/feeding/clean", {}], ["/api/feeding/confirm", {}],
      ["/api/training/backups/restore", { id: "fixture-backup" }], ["/api/learn", { steps: 16.0 }],
      ["/api/evolution/start", { unrequested: true }], ["/api/evolution/review", { ids: [] }]] as const) {
      const result = await differential(path, "POST", body); expect(result.status, path).toBeLessThan(500);
    }
  });
  test("private routes and keys stay inaccessible to public credentials", async () => {
    for (const path of ["/internal/data-scope", "/internal/unknown"]) expect((await request("bun", path)).status).toBe(404);
    expect((await request("bun", "/internal/rpc", "POST", {})).status).toBe(401);
    expect((await fetch(`http://127.0.0.1:${workerInfo.port}/api/health`)).status).toBe(401);
    for (const token of [settings.secret("ui_token"), settings.secret("access_key"), rpcToken]) {
      expect((await fetch(`http://127.0.0.1:${workerInfo.port}/api/health`, { headers: { authorization: "Bearer " + token } })).status).toBe(401);
    }
    for (const path of ["/api/health", "/api/settings", "/api/access-key", "/api/status"]) {
      const text = await (await request("bun", path)).text(); expect(text).not.toContain(rpcToken); expect(text).not.toContain(computeToken);
    }
  });
  test("session, settings, agreement and integer-versus-float validation", async () => {
    for (const side of ["bun", "oracle"] as const) {
      const info = side === "bun" ? { ui_token: settings.secret("ui_token") } : oracleInfo;
      const response = await request(side, "/api/session", "POST", { token: info.ui_token }); expect(response.status).toBe(200);
      expect(response.headers.get("set-cookie")).toContain("HttpOnly");
    }
    await differential("/api/settings"); await differential("/api/agreement");
    await differential("/api/agreement/accept", "POST", {}); await differential("/api/agreement");
    await differential("/api/settings", "PATCH", { temperature: 0.8, max_new_tokens: 32 });
    for (const body of ['{"max_new_tokens":16.0}', '{"max_new_tokens":true}', '[]', '{']) {
      await differential("/api/settings", "PATCH", body);
    }
    // Bun intentionally rejects non-finite JSON during parsing; Python's
    // legacy decoder reaches field validation. Both reject these HTTP inputs.
    for (const body of ['{"temperature":NaN}', '{"max_new_tokens":1e999}']) {
      for (const side of ["bun", "oracle"] as const) expect((await request(side, "/api/settings", "PATCH", body)).status).toBe(400);
    }
  });
  test("empty session body preserves HTTP 400 with a normalized JSON diagnostic", async () => {
    const bun = await request("bun", "/api/session", "POST", ""), original = await request("oracle", "/api/session", "POST", "");
    expect(bun.status).toBe(400); expect(original.status).toBe(400);
    expect((await bun.json() as any).error.message).toBe("Invalid JSON");
    expect((await original.json() as any).error.message).toContain("Expecting value");
  });
  test("streamed bodies enforce real bytes and malformed UTF8", async () => {
    const chunks = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(100001)); controller.enqueue(new Uint8Array(100001)); controller.close(); } });
    const response = await fetch(publicURL + "/api/settings", { method: "PATCH", headers: headers("bun", "/api/settings"), body: chunks });
    expect(response.status).toBe(413);
    const invalid = await fetch(publicURL + "/api/settings", { method: "PATCH", headers: headers("bun", "/api/settings"), body: new Uint8Array([0xff]) }); expect(invalid.status).toBe(400);
  });
  test("Bun sample creation and data listings match the original ledger", async () => {
    await differential("/api/samples", "POST", { prompt: "独立问题", answer: "独立答案", approved: true, training_allowed: true });
    await differential("/api/samples");
    await differential("/api/samples", "POST", { prompt: "独立问题", answer: "独立答案", purpose: "validation", approved: true, training_allowed: true });
    await differential("/api/conversations"); await differential("/api/materials"); await differential("/api/events");
    await differential("/api/conversations/missing-fixture");
    await differential("/api/conversations/missing-fixture", "DELETE");
  });
  test("real private RPC owns transactions, blocks public reads and permits readonly audit", async () => {
    const begin = await rpc("db_begin"); expect(begin.ok).toBe(true); const transaction = begin.result.transaction;
    expect((await rpc("db_execute", [], { sql: "INSERT INTO events VALUES (?,?,?,?)", parameters: ["atomic", "fixture", "{}", 0] }, transaction)).ok).toBe(true);
    let completed = false; const pending = request("bun", "/api/events").then(response => { completed = true; return response; });
    await Bun.sleep(50); expect(completed).toBe(false);
    await rpc("db_commit", [], {}, transaction); await rpc("db_close", [], {}, transaction);
    expect((await pending).status).toBe(200);
    const read = await rpc("db_begin", [], { readonly: true }); const lease = read.result.transaction;
    const rows = await rpc("db_execute", [], { sql: "SELECT id FROM events WHERE id='atomic'" }, lease); expect(rows.result.rows).toEqual([["atomic"]]);
    expect((await rpc("db_execute", [], { sql: "DELETE FROM events" }, lease)).ok).toBe(false);
    await rpc("db_close", [], {}, lease);
    expect(Number((await rpc("snapshot", [], {}, undefined, "settings")).result.temperature)).toBe(0.8);
  });
  test("real RPC rejects foreign Settings owners and transaction identity changes", async () => {
    const mismatch = await fetch(publicURL + "/internal/rpc", { method: "POST", headers: { authorization: "Bearer " + rpcToken },
      body: JSON.stringify({ target: "settings", method: "snapshot", home: join(directory, "foreign-home") }) });
    expect((await mismatch.json() as any).ok).toBe(false);
    const begin = await rpc("db_begin"), lease = begin.result.transaction;
    try {
      const mismatch = await fetch(publicURL + "/internal/rpc", { method: "POST", headers: { authorization: "Bearer " + rpcToken },
        body: JSON.stringify({ target: "memory", root: settings.snapshot().storage_root, method: "db_execute", transaction: lease,
          identity: "foreign-model", kwargs: { sql: "SELECT 1" } }) });
      expect((await mismatch.json() as any).ok).toBe(false);
    } finally { await rpc("db_close", [], {}, lease); }
  });
  test("POST chat and SSE pass through real compute and write events/conversations through Bun", async () => {
    const body = { messages: [{ role: "user", content: "你好" }], route: "local", thinking: "off" };
    const response = await request("bun", "/api/chat", "POST", body); expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    const text = await response.text(); expect(text).toContain("本地"); expect(text).toContain("回答"); expect(text).toContain('"type": "done"');
    const conversations = await (await request("bun", "/api/conversations")).json() as any;
    expect(conversations.conversations.length).toBe(1);
    const conversation = await (await request("bun", "/api/conversations/" + conversations.conversations[0].id)).json() as any;
    expect(conversation.messages.at(-1).content).toBe("本地回答");
    expect((await rpc("usage")).result.local.total_tokens).toBe(4);
    expect((await rpc("events")).result.some((event: any) => event.kind === "decision")).toBe(true);
    const completion = await request("bun", "/v1/chat/completions", "POST", body); expect(completion.status).toBe(200);
    expect((await completion.json() as any).choices[0].message.content).toBe("本地回答");
  });
  test("client SSE cancellation reaches compute without retaining a request", async () => {
    const controller = new AbortController();
    const response = await fetch(publicURL + "/api/chat", { method: "POST", headers: { ...headers("bun", "/api/chat"), "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "slow-stream" }], route: "local" }), signal: controller.signal });
    const reader = response.body!.getReader(); await reader.read(); controller.abort(); try { await reader.cancel(); } catch {}
    await Bun.sleep(150); const close = await request("bun", "/api/automatic/close-status"); expect(close.status).toBe(200);
    const exclusive = await request("bun", "/api/update/prepare", "POST", {}); expect(exclusive.status).toBe(200);
    expect((await request("bun", "/api/update/cancel", "POST", {})).status).toBe(200);
    const next = await request("bun", "/api/chat", "POST", { messages: [{ role: "user", content: "next" }], route: "local" });
    expect(next.status).toBe(200); await next.text();
  });
  test("audio API remains closed through the public proxy", async () => {
    const response = await request("bun", "/api/audio/transcriptions?language=en", "POST", "raw-wav-boundary-fixture", { "content-type": "audio/wav" });
    expect(response.status).toBe(400); expect((await response.json() as any).error.message).toContain("音频接口暂不开放");
  });
  test("public text chat rejects media before model or teacher for JSON SSE and compaction", async () => {
    const before = await rpc("usage");
    for (const path of ["/api/chat", "/v1/chat/completions"]) for (const mode of [{}, { stream: true }, { compact_only: true }]) {
      const response = await request("bun", path, "POST", { messages: [{ role: "user", content: [{ type: "input_audio", input_audio: { data: "AAAA", format: "wav" } }] }], route: "teacher", ...mode });
      expect(response.status).toBe(400); expect((await response.json() as any).error.message).toContain("仅支持文字");
    }
    expect(await rpc("usage")).toEqual(before);
    const { data } = await differential("/api/chat/capabilities");
    expect(data[0].text).toBe(true); expect(data[0].images).toBe(false); expect(data[0].audio).toBe(false);
  });
  test("package upload streams to the real import endpoint with an isolated invalid ZIP", async () => {
    const stream = new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode("invalid")); controller.enqueue(new TextEncoder().encode("ZIP")); controller.close(); } });
    const response = await fetch(publicURL + "/api/model-packages/import", { method: "POST", headers: { ...headers("bun", "/api/model-packages/import"), "content-type": "application/zip" }, body: stream });
    expect(response.status).toBe(200); expect((await response.json() as any).started).toBe(true);
    let body: any; for (let i = 0; i < 40; i++) { body = await (await request("bun", "/api/model-packages")).json(); if (body.task.state === "error") break; await Bun.sleep(25); }
    expect(body.task.state).toBe("error"); expect(body.task.bytes).toBe(10);
  });
  test("access-key rotation invalidates the old public key and never exposes internal keys", async () => {
    const old = settings.secret("access_key"), response = await request("bun", "/api/access-key/rotate", "POST", {});
    const body = await response.json() as any; expect(response.status).toBe(200); expect(body.key).not.toBe(old); expect(body.key).not.toBe(computeToken); expect(body.key).not.toBe(rpcToken);
    expect((await request("bun", "/v1/models", "GET", undefined, { authorization: "Bearer " + old })).status).toBe(401);
    expect((await request("bun", "/v1/models")).status).toBe(200);
  });
  test("Claude JSON and SSE reuse the real compute chat path with x-api-key or Bearer", async () => {
    const payload = { model: "luban0", max_tokens: 32, system: [{ type: "text", text: "简短回答" }],
      messages: [{ role: "user", content: [{ type: "text", text: "你好" }] }] };
    const claude = (body: any, extra = {}) => fetch(publicURL + "/v1/messages", { method: "POST", headers: {
      "x-api-key": settings.secret("access_key"), "anthropic-version": "2023-06-01", "content-type": "application/json", ...extra,
    }, body: JSON.stringify(body) });
    const json = await claude(payload); expect(json.status).toBe(200);
    expect(await json.json()).toMatchObject({ type: "message", model: "luban0", role: "assistant",
      content: [{ type: "text", text: "本地回答" }], stop_reason: "end_turn", stop_sequence: null,
      usage: { input_tokens: 2, output_tokens: 2 } });
    const streamed = await claude({ ...payload, stream: true }, { authorization: "Bearer " + settings.secret("access_key") });
    expect(streamed.headers.get("content-type")).toBe("text/event-stream");
    const raw = await streamed.text();
    const frames = raw.split("\n\n").filter(Boolean).map(frame => {
      const lines = frame.split("\n"), event = JSON.parse(lines[1]!.slice(6));
      expect(lines[0]).toBe("event: " + event.type); return event;
    });
    expect(frames.map(frame => frame.type)).toEqual(["message_start", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
    expect(frames.filter(frame => frame.type === "content_block_delta").map(frame => frame.delta.text).join("")).toBe("本地回答");
    expect(frames.at(-2)).toMatchObject({ delta: { stop_reason: "end_turn" }, usage: { input_tokens: 2, output_tokens: 2 } });
    expect(raw).not.toContain("思考"); expect(raw).not.toContain("[DONE]");
  });
  test("Claude boundary validates credentials, version, capabilities and the 8 MiB body limit", async () => {
    const payload = { model: "luban0", max_tokens: 32, messages: [{ role: "user", content: "你好" }] };
    const claude = (body: any, extra = {}, path = "/v1/messages") => fetch(publicURL + path, { method: "POST", headers: {
      "x-api-key": settings.secret("access_key"), "anthropic-version": "2023-06-01", "content-type": "application/json", ...extra,
    }, body: typeof body === "string" ? body : JSON.stringify(body) });
    for (const [body, extra, path, expected] of [
      [payload, { "x-api-key": "wrong" }, "/v1/messages", 401],
      [payload, { "anthropic-version": "wrong" }, "/v1/messages", 400],
      [{ ...payload, tools: [{ name: "shell", input_schema: { type: "array" } }] }, {}, "/v1/messages", 400],
      [{ ...payload, messages: [{ role: "user", content: [{ type: "tool_result", tool_use_id: "x", content: "ok" }] }] }, {}, "/v1/messages", 400],
      ["[]", {}, "/v1/messages", 400], ["{invalid", {}, "/v1/messages", 400],
      [payload, {}, "/v1/messages/count_tokens", 404],
      [{ ...payload, messages: [{ role: "user", content: "x".repeat(8 * 1024 * 1024) }] }, {}, "/v1/messages", 413],
    ] as const) {
      const response = await claude(body, extra, path); expect(response.status).toBe(expected);
      expect(await response.json()).toMatchObject({ type: "error", error: { message: expect.any(String), type: expect.any(String) } });
    }
    // A recognized Claude route must not fall through to the old 200 KB default.
    const large = await claude({ ...payload, messages: [{ role: "user", content: "x".repeat(210_000) }] });
    expect(large.status, await large.clone().text()).toBe(200);
    expect((await claude(payload, {}, "/v1/models")).status).toBe(401);
  }, 20000);

});


// Exercise the actual Bun -> private FastAPI -> Bun RPC boundary, not an adapter mock.
describe("real public HTTP tool loops with isolated CPU runtime", () => {
  const schema = { type: "object", properties: { city: { type: "string" } }, required: ["city"], additionalProperties: false };
  const tools = [{ type: "function", function: { name: "lookup", description: "查天气", parameters: schema } }];
  const firstMessage = { role: "user", content: "查上海天气" };
  const openAI = { model: "luban0", messages: [firstMessage], max_tokens: 64, tools, thinking: "off", route: "local" };
  const claude = { model: "luban0", messages: [firstMessage], max_tokens: 64,
    tools: [{ name: "lookup", description: "查天气", input_schema: schema }] };
  const claudeRequest = (body: any) => request("bun", "/v1/messages", "POST", body, { "anthropic-version": "2023-06-01" });
  const sse = (text: string) => text.split("\n\n").filter(Boolean).flatMap(frame => {
    const line = frame.split("\n").find(line => line.startsWith("data: "));
    return !line || line === "data: [DONE]" ? [] : [JSON.parse(line.slice(6))];
  });
  async function learningObservationState() {
    // A filtered run may reach this helper before any ordinary ledger request.
    expect((await request("bun", "/api/samples")).status).toBe(200);
    const begin = await rpc("db_begin", [], { readonly: true });
    expect(begin.ok, JSON.stringify(begin)).toBe(true);
    const lease = begin.result.transaction;
    let rows: any[], practiceRows: any[];
    try {
      rows = (await rpc("db_execute", [], { sql: "SELECT id,kind FROM events WHERE kind IN ('decision','tool_decision') ORDER BY id" }, lease)).result.rows;
      const exists = (await rpc("db_execute", [], { sql: "SELECT name FROM sqlite_master WHERE type='table' AND name='teacher_practice'" }, lease)).result.rows;
      practiceRows = exists.length ? (await rpc("db_execute", [], { sql: "SELECT request_id,state,decision_id FROM teacher_practice ORDER BY request_id" }, lease)).result.rows : [];
    } finally { await rpc("db_close", [], {}, lease); }
    const response = await request("bun", "/api/evolution");
    expect(response.status, await response.clone().text()).toBe(200);
    const candidates = (await response.json() as any).candidates;
    expect(Array.isArray(candidates)).toBe(true);
    // This product projection calls the real review._candidates with real RPC data.
    const toolIDs = new Set(rows.filter(row => row[1] === "tool_decision").map(row => "route:" + row[0]));
    expect(candidates.some((candidate: any) => toolIDs.has(candidate.id))).toBe(false);
    return { decisionIDs: rows.filter(row => row[1] === "decision").map(row => row[0]),
      toolCount: toolIDs.size, practiceRows, candidates };
  }
  test("buffered tool generation survives a shorter public idle timeout on both SSE protocols", async () => {
    const shortServer = Bun.serve({ hostname: "127.0.0.1", port: 0, idleTimeout: 1, fetch: () => new Response("starting") });
    const shortBackend = createBackend({ settings, port: shortServer.port!, computeURL: `http://127.0.0.1:${workerInfo.port}`,
      computeToken, rpcToken, webRoot: join(directory, "web"), version: "fixture", build: 1 });
    shortServer.reload({ fetch: shortBackend.fetch });
    try {
      for (const [path, body] of [["/v1/chat/completions", openAI], ["/v1/messages", claude]] as const) {
        const response = await fetch(shortServer.url.origin + path, { method: "POST", headers: {
          authorization: "Bearer " + settings.secret("access_key"), "content-type": "application/json", "anthropic-version": "2023-06-01",
        }, body: JSON.stringify({ ...body, stream: true, messages: [{ role: "user", content: "slow-tools" }] }) });
        expect(response.status).toBe(200);
        const text = await response.text(), values = sse(text);
        expect(values.some(value => value.error || value.type === "error")).toBe(false);
        if (path === "/v1/chat/completions") {
          expect(text).toContain(": keepalive");
          expect(values.some(value => value.choices?.[0]?.finish_reason === "tool_calls")).toBe(true);
          expect(text.trimEnd().endsWith("data: [DONE]")).toBe(true);
        } else {
          expect(values.some(value => value.type === "ping")).toBe(true);
          expect(values.some(value => value.delta?.stop_reason === "tool_use")).toBe(true);
          expect(values.at(-1).type).toBe("message_stop");
        }
      }
    } finally { shortBackend.close(); shortServer.stop(true); }
  }, 10000);
  test("OpenAI JSON and SSE complete call/result/answer and keep tool observations out of learning", async () => {
    const beforeObservation = await learningObservationState();
    const beforeSamples = await (await request("bun", "/api/samples")).json();
    const beforeMaterials = await (await request("bun", "/api/materials")).json();
    const beforeTeacher = (await rpc("usage")).result.teacher;
    for (const stream of [false, true]) {
      const response = await request("bun", "/v1/chat/completions", "POST", { ...openAI, stream, tool_choice: "required", parallel_tool_calls: false });
      expect(response.status).toBe(200);
      let call: any;
      if (stream) {
        const events = sse(await response.text());
        call = events.flatMap(event => event.choices?.[0]?.delta?.tool_calls ?? [])[0];
        expect(events.find(event => event.choices?.[0]?.finish_reason)?.choices[0].finish_reason).toBe("tool_calls");
        expect(events.find(event => event.usage)?.usage).toMatchObject({ prompt_tokens: 11, completion_tokens: 7 });
      } else {
        const body = await response.json() as any;
        expect(body.choices[0].finish_reason).toBe("tool_calls"); call = body.choices[0].message.tool_calls[0];
      }
      expect(typeof call.id).toBe("string");
      expect(call).toMatchObject({ type: "function", function: { name: "lookup" } });
      expect(JSON.parse(call.function.arguments)).toEqual({ city: "上海" });
      const follow = await request("bun", "/v1/chat/completions", "POST", { ...openAI, stream,
        messages: [firstMessage, { role: "assistant", content: null, tool_calls: [{ id: call.id, type: call.type, function: call.function }] },
          { role: "tool", tool_call_id: call.id, content: "晴" }] });
      expect(follow.status, JSON.stringify(call) + " " + await follow.clone().text()).toBe(200);
      if (stream) {
        const events = sse(await follow.text());
        expect(events.map(event => event.choices?.[0]?.delta?.content ?? "").join("")).toBe("工具结果：晴");
        expect(events.find(event => event.choices?.[0]?.finish_reason)?.choices[0].finish_reason).toBe("stop");
      } else expect((await follow.json() as any).choices[0].message.content).toBe("工具结果：晴");
    }
    expect(await (await request("bun", "/api/samples")).json()).toEqual(beforeSamples);
    expect(await (await request("bun", "/api/materials")).json()).toEqual(beforeMaterials);
    expect((await rpc("usage")).result.teacher).toEqual(beforeTeacher);
    const afterObservation = await learningObservationState();
    expect(afterObservation.decisionIDs).toEqual(beforeObservation.decisionIDs);
    expect(afterObservation.toolCount - beforeObservation.toolCount).toBe(4);
    expect(afterObservation.practiceRows).toEqual(beforeObservation.practiceRows);
    expect(afterObservation.candidates).toEqual(beforeObservation.candidates);
  });
  test("Claude JSON and SSE complete tool_use/tool_result/answer with exact usage", async () => {
    const beforeObservation = await learningObservationState();
    for (const stream of [false, true]) {
      const response = await claudeRequest({ ...claude, stream, tool_choice: { type: "tool", name: "lookup", disable_parallel_tool_use: true } });
      expect(response.status).toBe(200);
      let call: any;
      if (stream) {
        const events = sse(await response.text());
        call = events.find(event => event.content_block?.type === "tool_use").content_block;
        call.input = JSON.parse(events.filter(event => event.delta?.type === "input_json_delta").map(event => event.delta.partial_json).join(""));
        expect(events.at(-2)).toMatchObject({ delta: { stop_reason: "tool_use" }, usage: { input_tokens: 11, output_tokens: 7 } });
        expect(events.at(-1).type).toBe("message_stop");
      } else {
        const body = await response.json() as any;
        call = body.content.find((part: any) => part.type === "tool_use");
        expect(body).toMatchObject({ stop_reason: "tool_use", usage: { input_tokens: 11, output_tokens: 7 } });
      }
      expect(typeof call.id).toBe("string");
      expect(call).toMatchObject({ type: "tool_use", name: "lookup", input: { city: "上海" } });
      const follow = await claudeRequest({ ...claude, stream,
        messages: [firstMessage, { role: "assistant", content: [call] },
          { role: "user", content: [{ type: "tool_result", tool_use_id: call.id, content: "晴" }] }] });
      expect(follow.status, JSON.stringify(call) + " " + await follow.clone().text()).toBe(200);
      if (stream) {
        const events = sse(await follow.text());
        expect(events.filter(event => event.delta?.type === "text_delta").map(event => event.delta.text).join("")).toBe("工具结果：晴");
        expect(events.at(-2).delta.stop_reason).toBe("end_turn");
      } else expect((await follow.json() as any).content).toEqual([{ type: "text", text: "工具结果：晴" }]);
    }
    const afterObservation = await learningObservationState();
    expect(afterObservation.decisionIDs).toEqual(beforeObservation.decisionIDs);
    expect(afterObservation.toolCount - beforeObservation.toolCount).toBe(4);
    expect(afterObservation.practiceRows).toEqual(beforeObservation.practiceRows);
    expect(afterObservation.candidates).toEqual(beforeObservation.candidates);
  });
  test("bad tool history, definitions and choices return 400 before inference", async () => {
    for (const body of [
      { ...openAI, messages: [{ role: "tool", tool_call_id: "missing", content: "晴" }] },
      { ...openAI, messages: [firstMessage, { role: "assistant", content: null, tool_calls: [{ id: "pending", type: "function", function: { name: "lookup", arguments: "{}" } }] }, { role: "user", content: "forgot result" }] },
      { ...openAI, tool_choice: { type: "function", function: { name: "unknown" } } },
      { ...openAI, tools: [{ type: "function", function: { name: "lookup", parameters: { type: "array" } } }] },
      { ...openAI, parallel_tool_calls: "false" },
    ]) expect((await request("bun", "/v1/chat/completions", "POST", body)).status).toBe(400);
    for (const body of [
      { ...claude, tool_choice: { type: "tool", name: "unknown" } },
      { ...claude, messages: [{ role: "user", content: [{ type: "tool_result", tool_use_id: "missing", content: "晴" }] }] },
    ]) expect((await claudeRequest(body)).status).toBe(400);
  });
  test("truncated tool generation never exposes an executable call in JSON or SSE", async () => {
    for (const stream of [false, true]) {
      const response = await request("bun", "/v1/chat/completions", "POST", { ...openAI, stream, messages: [{ role: "user", content: "tool-truncated" }] });
      const text = await response.text();
      expect(text).not.toContain('"tool_calls":');
      if (stream) {
        const events = sse(text);
        expect(events.some(event => event.choices?.[0]?.finish_reason === "tool_calls")).toBe(false);
        expect(events.find(event => event.choices?.[0]?.finish_reason)?.choices[0].finish_reason).toBe("length");
      } else {
        expect(response.status).toBe(200);
        expect(JSON.parse(text).choices[0].finish_reason).toBe("length");
      }
      const adapted = await claudeRequest({ ...claude, stream, messages: [{ role: "user", content: "tool-truncated" }] });
      const adaptedText = await adapted.text();
      expect(adaptedText).not.toContain('"type":"tool_use"');
      if (stream) {
        const events = sse(adaptedText);
        expect(events.at(-2).delta.stop_reason).toBe("max_tokens");
        expect(events.at(-1).type).toBe("message_stop");
      } else {
        expect(adapted.status).toBe(200);
        expect(JSON.parse(adaptedText).stop_reason).toBe("max_tokens");
      }
    }
  });
});
