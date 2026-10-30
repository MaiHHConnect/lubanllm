import { timingSafeEqual } from "node:crypto";
import { isAbsolute, join, relative, resolve } from "node:path";
import { existsSync, lstatSync, realpathSync } from "node:fs";
import { Memory, pythonStrip } from "./memory.ts";
import { Settings } from "./settings.ts";
import { decodeWire, encodeWire, parseJSON, stringifyJSON } from "./wire.ts";
import { claudeError, claudePath, claudeRequest, claudeResponse } from "./anthropic.ts";
import { homeLedger, modelName, readModelName, saveModelName } from "./home.ts";
import type { NativeEvolutionService } from "./native-evolution-service.ts";
import { NativeEvolutionAutomatic, type NativeEvolutionScope, type NativeEvolutionInspection } from "./native-evolution-automatic.ts";

const security = {
  "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
  "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https://ai.maihh.net; connect-src 'self' https://ai.maihh.net; frame-ancestors 'none'",
};
const MAX_RPC = 64 * 1024 * 1024;
const loadedStates = ["ready", "generating", "learning", "evolving", "evaluating"];
const UPLOAD_LIMIT = 16 * 1024 * 1024 * 1024 + 32 * 1024 * 1024;
const object = (value: any): value is Record<string, any> => value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof Number);
const equal = (a: string, b: string) => /^[\x00-\x7f]*$/.test(a) && a.length === b.length && timingSafeEqual(Buffer.from(a), Buffer.from(b));
class HTTPFailure extends Error { constructor(readonly status: number, message: string) { super(message); } }

async function boundedBody(request: Request, limit: number): Promise<Uint8Array> {
  const length = request.headers.get("content-length");
  if (length !== null && (!/^\d+$/.test(length) || BigInt(length) > BigInt(limit))) throw new HTTPFailure(413, "Request body too large");
  if (!request.body) return new Uint8Array();
  const reader = request.body.getReader(), parts: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) { await reader.cancel(); throw new HTTPFailure(413, "Request body too large"); }
      parts.push(value);
    }
  } finally { reader.releaseLock(); }
  const result = new Uint8Array(size); let at = 0;
  for (const part of parts) { result.set(part, at); at += part.length; }
  return result;
}

export function createBackend(options: {
  settings: Settings; port: number; computeURL: string; computeToken: string; rpcToken: string;
  webRoot: string; version: string; build: number;
  nativeEvolution?: NativeEvolutionService;
  inspectNativeEvolution?: (scope: NativeEvolutionScope, memory: Memory) => Promise<NativeEvolutionInspection>;
}) {
  const { settings } = options;
  const memories = new Map<string, Memory>();
  let closing = false;
  const memory = (root: string) => {
    if (typeof root !== "string" || !isAbsolute(root)) throw new TypeError("存储目录必须是绝对路径");
    const path = resolve(root); let result = memories.get(path);
    if (!result) { result = new Memory(path); memories.set(path, result); }
    return result;
  };
  const nativeAutomatic = options.nativeEvolution && options.inspectNativeEvolution ? new NativeEvolutionAutomatic({
    service: options.nativeEvolution, settings, memory, inspect: options.inspectNativeEvolution,
  }) : undefined;
  const masked = (error: any) => settings.mask(String(error?.message ?? error)).replaceAll(options.rpcToken, "[redacted]").replaceAll(options.computeToken, "[redacted]");
  const json = (value: any, status = 200, headers: Record<string, string> = {}) => new Response(stringifyJSON(value), { status, headers: { "Content-Type": "application/json", ...headers } });
  async function compute(path: string, request?: Request, body?: BodyInit) {
    const headers = new Headers(request?.headers);
    for (const name of ["host", "cookie", "connection", "content-length", "transfer-encoding", "authorization", "proxy-authorization", "x-api-key"]) headers.delete(name);
    headers.set("authorization", "Bearer " + options.computeToken);
    try {
      return await fetch(options.computeURL + path, { method: request?.method ?? "GET", headers,
        body, signal: request?.signal, redirect: "manual", proxy: null });
    } catch (error) {
      if (request?.signal.aborted) throw new HTTPFailure(499, "请求已停止");
      throw new HTTPFailure(409, closing ? "应用正在退出" : "模型计算服务尚未就绪，请稍后重试");
    }
  }
  async function scope(path = "/internal/data-scope") {
    const response = await compute(path);
    if (!response.ok) throw new HTTPFailure(409, "模型数据范围尚未就绪，请稍后重试");
    const value = await response.json() as { root: string; identity: string | null; growth_profile?: any[] };
    if (!object(value) || typeof value.root !== "string" || (value.identity !== null && typeof value.identity !== "string")) throw new HTTPFailure(409, "模型数据范围无效");
    return { store: memory(value.root), identity: value.identity, growth_profile: value.growth_profile ?? [] };
  }
  async function rpc(request: Request) {
    if (request.method !== "POST" || !equal(request.headers.get("authorization") ?? "", "Bearer " + options.rpcToken)) throw new HTTPFailure(401, "Invalid private data key");
    let body: any;
    try { body = decodeWire(parseJSON(new TextDecoder("utf-8", { fatal: true }).decode(await boundedBody(request, MAX_RPC)))); }
    catch (error) { if (error instanceof HTTPFailure) throw error; throw new HTTPFailure(400, "Invalid JSON"); }
    if (!object(body) || typeof body.method !== "string" || !Array.isArray(body.args ?? []) || !object(body.kwargs ?? {})) throw new HTTPFailure(400, "RPC参数无效");
    try {
      let result: any;
      if (body.target === "settings") {
        if (body.home !== undefined && resolve(body.home) !== resolve(settings.home)) throw new TypeError("设置目录与服务不一致");
        const methods = ["snapshot", "public", "secret", "agreement", "accept_agreement", "validated", "update", "rotate_key", "mask"];
        if (!methods.includes(body.method) || Object.keys(body.kwargs ?? {}).length) throw new TypeError("未知设置方法或参数");
        result = (settings as any)[body.method](...(body.args ?? []));
      } else if (body.target === "native_evolution") {
        if (!nativeAutomatic || (body.args ?? []).length) throw new TypeError("原生进化调度入口尚未就绪");
        const now = body.method === "tick" && typeof body.kwargs?.now === "string" ? new Date(body.kwargs.now) : new Date();
        if (!Number.isFinite(now.getTime())) throw new TypeError("原生进化调度时间无效");
        result = await nativeAutomatic.request(body.method, body.kwargs ?? {}, { root: body.root, store: body.store, identity: body.identity }, now);
      } else if (body.target === "memory") {
        result = await memory(body.root ?? settings.snapshot().storage_root).dispatch(body.method, body.args ?? [], body.kwargs ?? {}, body.identity ?? null, body.transaction, request.signal);
      } else throw new TypeError("未知数据服务");
      return json({ ok: true, result: encodeWire(result) });
    } catch (error: any) {
      const code = String(error.code ?? "");
      const type = code.startsWith("SQLITE_CONSTRAINT") ? "IntegrityError" : code.startsWith("SQLITE_") ? "OperationalError"
        : error instanceof TypeError ? "TypeError" : code || /数据服务已关闭|数据库事务已结束|存储盘未连接/.test(error.message) ? "RuntimeError" : "ValueError";
      return json({ ok: false, error: { type, message: masked(error) } });
    }
  }
  async function route(request: Request) {
    const url = new URL(request.url), path = url.pathname;
    if (path.startsWith("/internal/")) {
      if (path !== "/internal/rpc") throw new HTTPFailure(404, "Not Found");
      return rpc(request);
    }
    if (closing) throw new HTTPFailure(409, "应用正在退出");
    const origin = request.headers.get("origin");
    if (path.startsWith("/api/") && origin && ![`http://127.0.0.1:${options.port}`, `http://localhost:${options.port}`].includes(origin)) throw new HTTPFailure(403, "Origin is not allowed");
    if (path.startsWith("/api/") && !["/api/session", "/api/health"].includes(path)) {
      const cookie = request.headers.get("cookie")?.split(";").map(part => part.trim()).find(part => part.startsWith("luban_session="))?.slice(14) ?? "";
      if (!cookie || !equal(cookie, settings.secret("ui_token"))) throw new HTTPFailure(401, "请从鲁班应用启动页面");
    }
    if (path.startsWith("/v1/")) {
      const authorization = request.headers.get("authorization");
      const authenticated = authorization ? equal(authorization, "Bearer " + settings.secret("access_key"))
        : claudePath(path) && equal(request.headers.get("x-api-key") ?? "", settings.secret("access_key"));
      if (!authenticated) return claudePath(path) ? claudeError(401, "Invalid local API key")
        : json({ error: { message: "Invalid local API key", type: "authentication_error" } }, 401);
    }
    if (claudePath(path) && path !== "/v1/messages") throw new HTTPFailure(404, "Unsupported Claude endpoint; use POST /v1/messages");
    if (path === "/v1/messages" && request.method !== "POST") throw new HTTPFailure(405, "Use POST /v1/messages");
    const native = options.nativeEvolution?.status();
    const nativeBusy = native && (native.running || native.lease_held || native.state === "quarantined");
    let body: Uint8Array | undefined, decoded: any;
    const upload = path === "/api/model-packages/import" && request.method === "POST";
    if (["POST", "PATCH", "PUT"].includes(request.method) && !upload) {
      const audio = ["/api/audio/transcriptions", "/v1/audio/transcriptions", "/api/chat/attachments/extract", "/api/feeding/extract"].includes(path);
      const limit = path === "/api/learning/media" ? 12 * 1024 * 1024 : audio || ["/api/chat", "/v1/chat/completions", "/v1/messages"].includes(path) ? 8 * 1024 * 1024 : path === "/api/feeding/scan" ? 2 * 1024 * 1024 : 200_000;
      body = await boundedBody(request, limit);
      if (body.length && !audio) {
        try { decoded = parseJSON(new TextDecoder("utf-8", { fatal: true }).decode(body)); }
        catch { throw new HTTPFailure(400, "Invalid JSON"); }
        if (!object(decoded)) throw new HTTPFailure(400, "请求需要 JSON 对象");
      }
    }
    const key = request.method + " " + path;
    const learningToggle = key === "PATCH /api/settings" && object(decoded)
      && Object.keys(decoded).length === 1 && object(decoded.automatic)
      && Object.keys(decoded.automatic).length === 1 && typeof decoded.automatic.learning_enabled === "boolean";
    if (nativeBusy && ["POST", "PATCH", "PUT", "DELETE"].includes(request.method) && !learningToggle
        && !["/api/learn/cancel", "/api/automatic/pause", "/api/session"].includes(path)) {
      throw new HTTPFailure(409, "原生进化正在占用模型，请先完成或取消进化");
    }
    if (nativeAutomatic && key === "POST /api/learn") {
      if (!decoded || Object.keys(decoded).length) throw new TypeError("专家学习预算由验证方法自动决定，学会后提前结束");
      const response = await compute("/internal/data-scope");
      if (!response.ok) throw new HTTPFailure(409, "模型数据范围尚未就绪，请稍后重试");
      return json(await nativeAutomatic.request("learn", {}, await response.json() as NativeEvolutionScope));
    }
    if (options.nativeEvolution && key === "POST /api/evolution/start") {
      if (!decoded || Object.keys(decoded).length) throw new TypeError("进化参数由已验证方法自动决定");
      if (nativeAutomatic) {
        const response = await compute("/internal/data-scope");
        if (!response.ok) throw new HTTPFailure(409, "模型数据范围尚未就绪，请稍后重试");
        return json(await nativeAutomatic.request("start", {}, await response.json() as NativeEvolutionScope));
      }
      const job = options.nativeEvolution.start();
      return json({ job_id: job.id });
    }
    // The existing UI cancellation endpoint is /api/learn/cancel, not /api/job/cancel.
    if (nativeBusy && key === "POST /api/automatic/pause") {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([options.nativeEvolution!.pause(), new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new HTTPFailure(409, "正在保存有效训练进度，请稍后再退出")), 55_000);
        })]);
      } finally { if (timer) clearTimeout(timer); }
      return json({ ok: true, paused: true });
    }
    if (nativeBusy && key === "POST /api/learn/cancel") {
      const job = options.nativeEvolution!.cancel();
      return json({ ok: true, job_id: job.id, state: job.state });
    }
    if (key === "GET /api/status" && !native?.id) {
      const health = await compute("/api/health");
      if (!health.ok) return health;
      const healthValue = await health.json() as any;
      if (!object(healthValue) || ![...loadedStates, "stopped"].includes(healthValue.state)) {
        return json({
          runtime: { state: healthValue.state || "loading", error: null, device: "mps", model: "luban0",
            model_path: "", base_parameters: 0, language_parameters: 0, language_layers: 0,
            routing_layers: 0, core_training_layer: null, core_version: null, core_requires_review: null,
            active_expert: null, expert_parameters: 0, paging: { enabled: false }, context: null,
            inference: { engine: "loading" }, training_engine: { name: "Luban 训练引擎", core: "loading", routing_and_experts: "loading" },
            native_process: null, last_generation_usage: null,
            router: { supported_actions: [], calibrated: false, gates_expert: false }, job: null, rss_bytes: 0, metal_bytes: 0 },
          speech: { state: "unloaded", error: "" }, teacher_calls: 0,
          samples: { training: 0, validation: 0 }, usage: {}, teacher_materials: { total: 0, pending: 0, eligible: 0 },
          automatic: { state: "waiting_model", experience: { current: 0, target: 350 } },
          benchmark: { state: "idle" }, growth: { state: "idle", domains: [], eligible_skills: [] },
          api_base: `http://127.0.0.1:${options.port}/v1`,
        });
      }
    }
    if (native?.id && key === "GET /api/status") {
      const response = await compute(path);
      if (!response.ok) return response;
      const value = await response.json() as any;
      if (!object(value) || !object(value.runtime)) throw new HTTPFailure(409, "模型状态响应无效");
      value.native_evolution = native;
      // Unloading clears Python's old job; a later Python job must not be hidden by our terminal result.
      if (!nativeBusy && native.round > 0 && value.runtime.job) return json(value);
      const state = native.state === "preparing" ? "queued" : native.running ? "running"
        : ["failed", "quarantined"].includes(native.state) ? "error" : native.state === "unaccepted" ? "rejected" : native.state;
      value.runtime = { ...value.runtime, ...(nativeBusy ? { state: "evolving" } : {}), job: {
        id: native.id, kind: native.kind, state, step: native.step, steps: native.steps, phase: native.phase,
        round: native.round, cancel_requested: native.state === "cancelling", error: native.cleanup_error ?? native.error,
        result: native.last_result ? { ...native.last_result, promoted: native.last_result.adopted === true } : null,
      } };
      if (nativeBusy) value.automatic = { ...value.automatic, state,
        reason: native.cleanup_error ?? native.error ?? "" };
      return json(value);
    }
    if (nativeBusy && key === "GET /api/automatic/close-status") {
      return json({ running: true, resumable: false, pausing: native.state === "cancelling", state: native.state, package_busy: false, job_id: native.id });
    }
    if (key === "GET /api/health") {
      let state = "loading";
      try { const r = await compute("/api/health"); if (r.ok) state = (await r.json() as any).state; } catch {}
      return json({ app: "luban", state, version: options.version, build: options.build });
    }
    if (key === "GET /api/version") return json({ version: options.version, build: options.build });
    if (key === "POST /api/session") {
      if (decoded === undefined) throw new HTTPFailure(400, "Invalid JSON");
      if (typeof decoded?.token !== "string" || !equal(decoded.token, settings.secret("ui_token"))) return json({ detail: "Invalid launch session" }, 401);
      return json({ ok: true }, 200, { "Set-Cookie": `luban_session=${decoded.token}; HttpOnly; Path=/; SameSite=strict` });
    }
    if (key === "GET /api/settings") return json(settings.public());
    if (key === "GET /api/agreement") return json(settings.agreement());
    if (key === "POST /api/agreement/accept") return json(settings.accept_agreement());
    if (key === "GET /api/access-key") return json({ key: settings.secret("access_key") });
    if (key === "POST /api/access-key/rotate") return json({ key: settings.rotate_key() });
    if (key === "GET /api/home") {
      const health = await compute("/api/health");
      if (!health.ok) return health;
      const healthValue = await health.json() as any;
      if (!object(healthValue) || ![...loadedStates, "stopped"].includes(healthValue.state)) {
        throw new HTTPFailure(409, "模型状态尚未就绪，请稍后重试");
      }
      const [{ store, identity, growth_profile }, response] = await Promise.all([scope("/internal/home-profile"), compute("/api/status")]);
      if (!response.ok) throw new HTTPFailure(409, "模型状态尚未就绪，请稍后重试");
      const state = await response.json() as any;
      const { evolution_count, recent, last_evolution, ...activity } = await homeLedger(store, identity);
      return json({ model: { id: "luban0", name: readModelName(settings.home, identity),
        can_rename: evolution_count > 0, evolution_count,
        core_parameters: state.runtime?.language_parameters ?? null }, activity, growth_profile, recent, last_evolution });
    }
    if (key === "PATCH /api/home/model-name") {
      const name = modelName(decoded?.name);
      const response = await compute("/internal/home-evolution");
      if (!response.ok) throw new HTTPFailure(409, "个人进化记录尚未通过核验");
      const verified = await response.json() as any;
      const current = await scope();
      if (!Number.isSafeInteger(verified.evolution_count) || verified.evolution_count < 1
          || verified.identity !== current.identity || verified.root !== current.store.root) throw new HTTPFailure(409, "完成一次个人进化后才可修改模型名称");
      return json({ id: "luban0", name: saveModelName(settings.home, current.identity!, name) });
    }
    if (key === "GET /v1/models") return json({ object: "list", data: [{ id: "luban0", object: "model", owned_by: "maihaohuo" }] });
    if (["GET /api/conversations", "GET /api/samples", "GET /api/materials", "GET /api/events", "POST /api/samples"].includes(key) || (request.method === "GET" && path.startsWith("/api/conversations/"))) {
      if (key === "GET /api/conversations" || (request.method === "GET" && path.startsWith("/api/conversations/"))) {
        const health = await compute("/api/health");
        if (!health.ok) return health;
        const healthValue = await health.json() as any;
        if (!object(healthValue) || !loadedStates.includes(healthValue.state)) {
          throw new HTTPFailure(409, "模型状态尚未就绪，请稍后重试");
        }
      }
      const { store, identity } = await scope();
      if (key === "GET /api/conversations") return json({ conversations: await store.dispatch("conversations", [], {}, identity) });
      if (key === "GET /api/samples") return json({ samples: (await store.dispatch("samples", [], {}, identity)).filter((row: any) => row.approved && row.training_allowed) });
      if (key === "GET /api/materials") return json({ materials: await store.dispatch("teacher_materials", [], {}, identity) });
      if (key === "GET /api/events") return json({ events: await store.dispatch("events", [], {}, identity) });
      if (request.method === "GET") {
        const id = decodeURIComponent(path.slice("/api/conversations/".length));
        if (!id || id.includes("/")) return json({ detail: "Not Found" }, 404);
        const result = await store.dispatch("conversation", [id], {}, identity);
        return result ? json(result) : json({ detail: "Conversation not found" }, 404);
      }
      const value = decoded ?? {}, source = value.source ?? "human";
      const provenance: Record<string, any> = { verified_by: "local requester", created: Date.now() / 1000 };
      if (source === "synthetic") provenance.scope = "manually supplied synthetic research material; not a teacher response";
      for (const name of ["family_group", "split", "split_family", "evolution_role"]) if (name in value) {
        if (typeof value[name] !== "string" || !pythonStrip(value[name])) throw new TypeError(name + " 需要非空文本");
        provenance[name] = value[name];
      }
      if (source === "human") provenance.evolution_role ??= "new";
      if (value.route_action) {
        if (!["local_core", "neural_expert", "teacher"].includes(value.route_action)) throw new TypeError("路由标注应为 local_core、neural_expert 或 teacher");
        provenance.route_action = value.route_action;
      }
      if (source === "teacher") {
        const config = settings.snapshot().teacher, agreement = settings.agreement();
        if (!agreement.accepted) throw new TypeError("请先阅读并接受使用协议，再提交教师学习材料；您需自行处理对应服务条款与第三方权利");
        Object.assign(provenance, { provider: config.base_url, model: config.model, rights_basis: "user responsibility under agreement", agreement_version: agreement.version, user_responsibility: true });
      }
      return json({ id: await store.dispatch("add_sample", [value.prompt ?? "", value.answer ?? "", source, value.purpose ?? "training", value.approved === true, source === "teacher" || value.training_allowed === true, provenance], {}, identity) });
    }
    if (key === "POST /v1/messages") {
      const chat = claudeRequest(decoded, request.headers.get("anthropic-version"));
      // Size only, never content: lets a long agent prompt be told apart from a prefill problem.
      console.log(`Claude入口请求 字符=${chat.messages.reduce((n: number, m: any) => n + String(m.content ?? "").length, 0)} 工具=${chat.tools?.length ?? 0} 消息=${chat.messages.length}`);
      return claudeResponse(await compute("/v1/chat/completions", request, stringifyJSON(chat)), chat.stream, masked);
    }
    if (path.startsWith("/api/") || path.startsWith("/v1/")) {
      if (upload) {
        const length = request.headers.get("content-length");
        if (length && (!/^\d+$/.test(length) || BigInt(length) > BigInt(UPLOAD_LIMIT))) throw new HTTPFailure(413, "Request body too large");
        let total = 0, tooLarge = false;
        const stream = request.body?.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({ transform(chunk, controller) {
          total += chunk.length;
          if (total > UPLOAD_LIMIT) { tooLarge = true; controller.error(new Error("Request body too large")); } else controller.enqueue(chunk);
        } }));
        try { return await compute(path + url.search, request, stream); }
        catch (error) { if (tooLarge) throw new HTTPFailure(413, "Request body too large"); throw error; }
      }
      return compute(path + url.search, request, body);
    }
    if (!["GET", "HEAD"].includes(request.method)) return json({ detail: "Method Not Allowed" }, 405, { Allow: "GET, HEAD" });
    if (!existsSync(options.webRoot)) return json({ message: "前端尚未构建，请执行 bun run build" }, 503);
    const web = realpathSync(options.webRoot), file = resolve(web, "." + decodeURIComponent(path), path.endsWith("/") ? "index.html" : "");
    if (relative(web, file).startsWith("..") || !existsSync(file) || !lstatSync(file).isFile() || relative(web, realpathSync(file)).startsWith("..")) return json({ detail: "Not Found" }, 404);
    return new Response(request.method === "HEAD" ? null : Bun.file(file), { headers: { "Content-Type": Bun.file(file).type, "Content-Length": String(Bun.file(file).size) } });
  }
  return {
    memory,
    async fetch(request: Request) {
      let response: Response;
      try {
        const url = new URL(request.url);
        if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) throw new HTTPFailure(403, "Host is not allowed");
        response = await route(request);
      } catch (error: any) {
        const status = error instanceof HTTPFailure ? error.status : error.code || /存储盘未连接|数据库事务已结束/.test(error.message) ? 409 : 400;
        response = claudePath(new URL(request.url).pathname) ? claudeError(status, masked(error))
          : json({ error: { message: masked(error) } }, status);
      }
      const headers = new Headers(response.headers);
      for (const [name, value] of Object.entries(security)) headers.set(name, value);
      if (new URL(request.url).pathname.startsWith("/api/")) headers.set("Cache-Control", "no-store");
      for (const name of ["connection", "transfer-encoding", "keep-alive"]) headers.delete(name);
      return new Response(response.body, { status: response.status, headers });
    },
    beginClose() { closing = true; },
    close() { closing = true; for (const store of memories.values()) store.close(); },
  };
}
