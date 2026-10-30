import { Database } from "bun:sqlite";
import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { canonical } from "./evaluation";

const POLICY = "inference_input_output_v1";
const MAX_TOTAL = 1e15;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SECRET = /^[0-9a-f]{64}$/;
const COOKIE = "enterprise_admin";
const sha = (value: string) => createHash("sha256").update(value).digest("hex");
const iso = (value: number) => new Date(value).toISOString();
class ApiError extends Error {
  constructor(message: string, readonly status = 400, readonly event?: string) { super(message); }
}
type Body = Record<string, unknown>;
type Company = { id: string; name: string; status: string; billing_policy: string; annual_tokens: number; purchased_tokens: number; used_tokens: number; expires_at: number; created_at: number };
type Device = { id: string; company_id: string; name: string; created_at: number; used_tokens: number; last_seen_at: number };

function fields(body: Body, required: string[]) {
  if (Object.keys(body).length !== required.length || required.some(key => !Object.hasOwn(body, key))) throw new ApiError("请求字段无效");
}
function textValue(value: unknown, max: number, allowEmpty = false): string {
  if (typeof value !== "string" || value.length > max || (!allowEmpty && !value.trim()) || /[\u0000-\u001f\u007f]/.test(value)) throw new ApiError("文字字段无效");
  return value.trim();
}
function integer(value: unknown, min = 1, max = MAX_TOTAL): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min || value > max) throw new ApiError("Token 数量无效");
  return value;
}
function uuid(value: unknown): string {
  if (typeof value !== "string" || !UUID.test(value)) throw new ApiError("请求标识无效");
  return value.toLowerCase();
}
function nextYear(start: number) {
  const date = new Date(start), month = date.getUTCMonth();
  date.setUTCFullYear(date.getUTCFullYear() + 1);
  if (date.getUTCMonth() !== month) date.setUTCDate(0);
  return date.getTime();
}
async function readBody(request: Request): Promise<Body> {
  if (!/^application\/json(?:\s*;.*)?$/i.test(request.headers.get("content-type") || "")) throw new ApiError("请求需要 JSON 格式", 415);
  const length = request.headers.get("content-length");
  if (length && (!/^\d+$/.test(length) || Number(length) > 16_384)) throw new ApiError("请求内容过大", 413);
  const reader = request.body?.getReader();
  let size = 0;
  const chunks: Uint8Array[] = [];
  if (reader) {
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > 16_384) { await reader.cancel(); throw new ApiError("请求内容过大", 413); }
        chunks.push(value);
      }
    } finally { reader.releaseLock(); }
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  try {
    const body = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error();
    return body;
  } catch { throw new ApiError("JSON 内容无效"); }
}

export function createEnterprise(db: Database, options: { adminKey: string; origin?: string; now?: () => number }) {
  const origin = options.origin || "https://ai.maihh.net";
  const originURL = new URL(origin);
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(originURL.hostname);
  if (originURL.origin !== origin || (!local && originURL.protocol !== "https:") || options.adminKey.length < 32 || options.adminKey.length > 1024) throw new Error("企业授权配置无效");
  const now = options.now || Date.now;
  const adminHash = Buffer.from(sha(options.adminKey), "hex");
  const cookie = (value: string, age: number) => `${COOKIE}=${value}; Path=/api/enterprise/admin; HttpOnly; SameSite=Strict; Max-Age=${age}${local ? "" : "; Secure"}`;
  const reply = (value: unknown, status = 200, extra: Record<string, string> = {}) => Response.json(value, { status, headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", ...extra } });
  db.transaction(() => {
    db.exec(`CREATE TABLE IF NOT EXISTS enterprise_companies (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('active','suspended')),
    billing_policy TEXT NOT NULL CHECK(billing_policy='inference_input_output_v1'), annual_tokens INTEGER NOT NULL,
    purchased_tokens INTEGER NOT NULL CHECK(purchased_tokens BETWEEN 0 AND 1000000000000000),
    used_tokens INTEGER NOT NULL CHECK(used_tokens BETWEEN 0 AND 1000000000000000),
    license_hash TEXT NOT NULL UNIQUE, expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS enterprise_devices (
    id TEXT PRIMARY KEY, company_id TEXT NOT NULL, name TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, created_at INTEGER NOT NULL, last_seen_at INTEGER NOT NULL, used_tokens INTEGER NOT NULL DEFAULT 0 CHECK(used_tokens BETWEEN 0 AND 1000000000000000)
  );
  CREATE INDEX IF NOT EXISTS enterprise_device_company ON enterprise_devices(company_id);
  CREATE TABLE IF NOT EXISTS enterprise_sessions (token_hash TEXT PRIMARY KEY, expires_at INTEGER NOT NULL);
  CREATE TABLE IF NOT EXISTS enterprise_requests (
    scope TEXT NOT NULL, request_id TEXT NOT NULL, body_hash TEXT NOT NULL, response TEXT NOT NULL, created_at INTEGER NOT NULL,
    PRIMARY KEY(scope,request_id)
  );
  CREATE TABLE IF NOT EXISTS enterprise_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT NOT NULL, device_id TEXT, kind TEXT NOT NULL,
    purchased_delta INTEGER NOT NULL, used_delta INTEGER NOT NULL,
    note TEXT NOT NULL, request_id TEXT, created_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS enterprise_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT NOT NULL, device_id TEXT NOT NULL,
    request_id TEXT NOT NULL, used_tokens INTEGER NOT NULL, delta INTEGER NOT NULL, created_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS enterprise_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT NOT NULL, device_id TEXT, kind TEXT NOT NULL, created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS enterprise_usage_company ON enterprise_usage(company_id,created_at DESC,id DESC);
  CREATE INDEX IF NOT EXISTS enterprise_ledger_company ON enterprise_ledger(company_id,created_at DESC,id DESC);
  CREATE INDEX IF NOT EXISTS enterprise_events_company ON enterprise_events(company_id,created_at DESC,id DESC);`);
  for (const table of ["enterprise_ledger", "enterprise_usage", "enterprise_events", "enterprise_requests"]) {
    db.exec(`CREATE TRIGGER IF NOT EXISTS ${table}_no_update BEFORE UPDATE ON ${table} BEGIN SELECT RAISE(ABORT,'append-only'); END;
      CREATE TRIGGER IF NOT EXISTS ${table}_no_delete BEFORE DELETE ON ${table} BEGIN SELECT RAISE(ABORT,'append-only'); END;`);
  }
  }).immediate();
  const limits = new Map<string, { count: number; reset: number }>();
  function rate(key: string, maximum: number) {
    const time = now();
    let bucket = limits.get(key);
    if (!bucket || bucket.reset <= time) { bucket = { count: 0, reset: time + 60_000 }; limits.set(key, bucket); }
    if (++bucket.count > maximum) throw new ApiError("请求过于频繁，请稍后重试", 429);
  }
  function company(id: string): Company {
    const row = db.query<Company, [string]>("SELECT * FROM enterprise_companies WHERE id=?").get(id);
    if (!row) throw new ApiError("企业不存在", 404);
    return row;
  }
  function summary(row: Company) {
    const count = db.query<{ total: number }, [string]>("SELECT COUNT(*) AS total FROM enterprise_devices WHERE company_id=?").get(row.id)!.total;
    const followup: string[] = [];
    if (row.expires_at <= now()) followup.push("license_expired");
    else if (row.expires_at - now() <= 30 * 86_400_000) followup.push("license_expiring");
    if (row.used_tokens > row.purchased_tokens) followup.push("quota_overage");
    else if (row.used_tokens === row.purchased_tokens) followup.push("quota_exhausted");
    return { id: row.id, name: row.name, status: row.status, billing_policy: row.billing_policy, annual_tokens: row.annual_tokens,
      purchased_tokens: row.purchased_tokens, used_tokens: row.used_tokens,
      available_tokens: row.purchased_tokens - row.used_tokens, remaining_tokens: Math.max(row.purchased_tokens - row.used_tokens, 0), overage_tokens: Math.max(row.used_tokens - row.purchased_tokens, 0), expires_at: iso(row.expires_at), created_at: iso(row.created_at), device_count: count,
      needs_followup: followup.length > 0, followup_reasons: followup };
  }
  function deviceView(device: Device) { return { device_id: device.id, device_name: device.name, created_at: iso(device.created_at), used_tokens: device.used_tokens, last_seen_at: iso(device.last_seen_at) }; }
  function ledger(companyId: string, kind: string, requestId: string | null, purchased = 0, used = 0, note = "", deviceId: string | null = null) {
    db.query("INSERT INTO enterprise_ledger (company_id,device_id,kind,purchased_delta,used_delta,note,request_id,created_at) VALUES (?,?,?,?,?,?,?,?)")
      .run(companyId, deviceId, kind, purchased, used, note, requestId, now());
  }
  function admin(request: Request) {
    const values = (request.headers.get("cookie") || "").split(";").map(value => value.trim()).filter(value => value.startsWith(`${COOKIE}=`));
    if (values.length !== 1) throw new ApiError("请先登录管理员", 401);
    const token = values[0].slice(COOKIE.length + 1);
    const row = SECRET.test(token) ? db.query<{ expires_at: number }, [string]>("SELECT expires_at FROM enterprise_sessions WHERE token_hash=?").get(sha(token)) : null;
    if (!row || row.expires_at <= now()) throw new ApiError("管理员登录已失效", 401);
    return token;
  }
  function deviceAuth(request: Request): Device {
    const token = /^Bearer ([0-9a-f]{64})$/.exec(request.headers.get("authorization") || "")?.[1];
    const row = token ? db.query<Device, [string]>("SELECT id,company_id,name,created_at,used_tokens,last_seen_at FROM enterprise_devices WHERE token_hash=?").get(sha(token)) : null;
    if (!row) { rate("bad-device", 100); throw new ApiError("设备凭据无效", 401); }
    return row;
  }
  function idempotent(scope: string, request: Body, path: string, operation: (id: string) => Body) {
    const requestId = uuid(request.request_id), bodyHash = sha(canonical({ path, body: request }));
    const previous = db.query<{ body_hash: string; response: string }, [string, string]>("SELECT body_hash,response FROM enterprise_requests WHERE scope=? AND request_id=?").get(scope, requestId);
    if (previous) {
      if (previous.body_hash !== bodyHash) throw new ApiError("请求标识已用于其他内容", 409, "idempotency_conflict");
      return JSON.parse(previous.response) as Body;
    }
    const response = operation(requestId);
    const stored = { ...response };
    if (Object.hasOwn(stored, "license_code")) { delete stored.license_code; stored.code_available = false; }
    db.query("INSERT INTO enterprise_requests VALUES (?,?,?,?,?)").run(scope, requestId, bodyHash, JSON.stringify(stored), now());
    return response;
  }
  const mutateAdmin = db.transaction((body: Body, path: string, companyId?: string, action?: string) => idempotent("admin", body, path, requestId => {
    if (!companyId) {
      const name = textValue(body.name, 200), tokens = integer(body.annual_tokens);
      if (body.billing_policy !== POLICY) throw new ApiError("计费口径无效");
      const id = crypto.randomUUID(), code = randomBytes(32).toString("hex"), time = now(), expiry = nextYear(time);
      db.query("INSERT INTO enterprise_companies VALUES (?,?, 'active',?,?,?,?,?,?,?)").run(id, name, POLICY, tokens, tokens, 0, sha(code), expiry, time);
      ledger(id, "initial", requestId, tokens);
      return { company: summary(company(id)), license_code: code, code_available: true };
    }
    const row = company(companyId), note = textValue(body.note, 500, true);
    if (action === "topups" || action === "renew") {
      const tokens = integer(action === "renew" ? body.annual_tokens : body.tokens);
      if (row.purchased_tokens + tokens > MAX_TOTAL) throw new ApiError("累计额度超过上限");
      const expiry = action === "renew" ? nextYear(Math.max(now(), row.expires_at ?? now())) : row.expires_at;
      if (action === "renew") db.query("UPDATE enterprise_companies SET purchased_tokens=purchased_tokens+?,annual_tokens=?,expires_at=? WHERE id=?").run(tokens, tokens, expiry, row.id);
      else db.query("UPDATE enterprise_companies SET purchased_tokens=purchased_tokens+? WHERE id=?").run(tokens, row.id);
      ledger(row.id, action === "renew" ? "renew" : "topup", requestId, tokens, 0, note);
    } else if (action === "status") {
      if (body.status !== "active" && body.status !== "suspended") throw new ApiError("授权状态无效");
      db.query("UPDATE enterprise_companies SET status=? WHERE id=?").run(body.status, row.id);
      ledger(row.id, `status_${body.status}`, requestId, 0, 0, note);
    } else if (action === "rotate") {
      const code = randomBytes(32).toString("hex");
      db.query("UPDATE enterprise_companies SET license_hash=? WHERE id=?").run(sha(code), row.id);
      ledger(row.id, "license_rotate", requestId, 0, 0, note);
      return { company: summary(company(row.id)), license_code: code, code_available: true };
    }
    return { company: summary(company(row.id)) };
  }));
  const activate = db.transaction((body: Body) => {
    const id = uuid(body.device_id), name = textValue(body.device_name, 120);
    if (typeof body.license_code !== "string" || !SECRET.test(body.license_code)) throw new ApiError("授权码无效", 401);
    const row = db.query<Company, [string]>("SELECT * FROM enterprise_companies WHERE license_hash=?").get(sha(body.license_code));
    if (!row) throw new ApiError("授权码无效", 401);
    if (row.status === "suspended") throw new ApiError("企业已暂停新增设备激活", 403);
    if (db.query("SELECT id FROM enterprise_devices WHERE id=?").get(id)) throw new ApiError("设备已激活，请使用原设备凭据", 409);
    const token = randomBytes(32).toString("hex"), time = now();
    db.query("INSERT INTO enterprise_devices (id,company_id,name,token_hash,created_at,last_seen_at) VALUES (?,?,?,?,?,?)").run(id, row.id, name, sha(token), time, time);
    ledger(row.id, "device_activate", null, 0, 0, name, id);
    return { company: summary(row), device: deviceView({ id, company_id: row.id, name, created_at: time, last_seen_at: time, used_tokens: 0 }), device_token: token, reporting_required: true, usage_allowed: true };
  });
  const mutateDevice = db.transaction((device: Device, body: Body, path: string) => idempotent(device.id, body, path, requestId => {
    const row = company(device.company_id);
    // Reload under the write lock: concurrent reports must compare against the latest counter.
    const current = db.query<Device, [string]>("SELECT id,company_id,name,created_at,used_tokens,last_seen_at FROM enterprise_devices WHERE id=?").get(device.id)!;
    const used = integer(body.used_tokens, 0);
    if (used < current.used_tokens) throw new ApiError("累计用量不可减少", 409, "usage_decrease");
    const delta = used - current.used_tokens;
    if (row.used_tokens + delta > MAX_TOTAL) throw new ApiError("企业累计用量超过安全上限", 409, "usage_total_exceeded");
    db.query("UPDATE enterprise_devices SET used_tokens=?,last_seen_at=? WHERE id=?").run(used, now(), device.id);
    db.query("UPDATE enterprise_companies SET used_tokens=used_tokens+? WHERE id=?").run(delta, row.id);
    db.query("INSERT INTO enterprise_usage (company_id,device_id,request_id,used_tokens,delta,created_at) VALUES (?,?,?,?,?,?)").run(row.id, device.id, requestId, used, delta, now());
    ledger(row.id, "usage", requestId, 0, delta, "", device.id);
    return { company: summary(company(row.id)), reporting_required: true, usage: { used_tokens: used, delta } };
  }));

  return { async handle(request: Request): Promise<Response | null> {
    const path = new URL(request.url).pathname;
    if (!path.startsWith("/api/enterprise/")) return null;
    let device: Device | undefined;
    try {
      const suppliedOrigin = request.headers.get("origin"), adminPath = path.startsWith("/api/enterprise/admin/");
      if ((suppliedOrigin !== null && suppliedOrigin !== origin) || (adminPath && request.method !== "GET" && suppliedOrigin !== origin)) throw new ApiError("请求来源不允许", 403);
      if (new URL(request.url).search) throw new ApiError("请求参数无效");
      if (path === "/api/enterprise/admin/login" && request.method === "POST") {
        rate("login", 20);
        const body = await readBody(request); fields(body, ["key"]);
        if (typeof body.key !== "string" || body.key.length > 1024 || !timingSafeEqual(Buffer.from(sha(body.key), "hex"), adminHash)) throw new ApiError("管理员密钥无效", 401);
        const token = randomBytes(32).toString("hex");
        db.transaction(() => {
          db.query("DELETE FROM enterprise_sessions WHERE expires_at<=?").run(now());
          db.query("INSERT INTO enterprise_sessions VALUES (?,?)").run(sha(token), now() + 8 * 60 * 60_000);
        }).immediate();
        return reply({ authenticated: true }, 200, { "Set-Cookie": cookie(token, 8 * 60 * 60) });
      }
      if (adminPath) {
        const token = admin(request);
        if (path === "/api/enterprise/admin/logout" && request.method === "POST") {
          const body = await readBody(request); fields(body, []);
          db.transaction(() => db.query("DELETE FROM enterprise_sessions WHERE token_hash=?").run(sha(token))).immediate();
          return reply({ authenticated: false }, 200, { "Set-Cookie": cookie("", 0) });
        }
        if (path === "/api/enterprise/admin/companies") {
          if (request.method === "GET") return reply({ companies: db.query<Company, []>("SELECT * FROM enterprise_companies ORDER BY created_at DESC,id").all().map(summary) });
          if (request.method === "POST") {
            const body = await readBody(request); fields(body, ["request_id", "name", "annual_tokens", "billing_policy"]);
            return reply(mutateAdmin.immediate(body, path), 201);
          }
        }
        if (path === "/api/enterprise/admin/followups" && request.method === "GET") {
          return reply({ companies: db.query<Company, []>("SELECT * FROM enterprise_companies ORDER BY expires_at ASC,id").all().map(summary).filter(row => row.needs_followup) });
        }
        const match = /^\/api\/enterprise\/admin\/companies\/([^/]+)(?:\/(topups|renew|status|rotate))?$/.exec(path);
        if (match) {
          const id = uuid(match[1]), action = match[2];
          if (request.method === "GET" && !action) {
            const row = company(id);
            const devices = db.query<Device, [string]>("SELECT id,company_id,name,created_at,used_tokens,last_seen_at FROM enterprise_devices WHERE company_id=? ORDER BY created_at DESC").all(id).map(deviceView);
            const recent = (table: string) => db.query<Record<string, unknown>, [string]>(`SELECT * FROM ${table} WHERE company_id=? ORDER BY created_at DESC,id DESC LIMIT 100`).all(id).map(record => ({ ...record, created_at: iso(record.created_at as number) }));
            const warnings: string[] = [];
            if (row.expires_at <= now()) warnings.push("年度授权已到期，请跟进续费；设备仍可使用");
            if (row.used_tokens > row.purchased_tokens) warnings.push("累计付费用量已超过购买额度，请跟进充值；设备仍可使用");
            if (row.status === "suspended") warnings.push("已暂停新增设备激活；既有设备仍可使用");
            return reply({ company: summary(row), devices, usage: recent("enterprise_usage"), ledger: recent("enterprise_ledger"), events: recent("enterprise_events"), warnings });
          }
          if (request.method === "POST" && action) {
            const body = await readBody(request);
            fields(body, action === "topups" ? ["request_id", "tokens", "note"] : action === "renew" ? ["request_id", "annual_tokens", "note"] : action === "status" ? ["request_id", "status", "note"] : ["request_id", "note"]);
            return reply(mutateAdmin.immediate(body, path, id, action));
          }
        }
        throw new ApiError("接口不存在或请求方法不允许", 404);
      }
      if (path === "/api/enterprise/activate" && request.method === "POST") {
        rate("activate", 60);
        const body = await readBody(request); fields(body, ["license_code", "device_id", "device_name"]);
        return reply(activate.immediate(body), 201);
      }
      if (["/api/enterprise/me", "/api/enterprise/usage"].includes(path)) {
        device = deviceAuth(request);
        if (path.endsWith("/me") && request.method === "GET") {
          const row = company(device.company_id), view = summary(row);
          return reply({ company: view, device: deviceView(device), total_devices: view.device_count, usage_allowed: true, reporting_required: true });
        }
        if (request.method === "POST" && !path.endsWith("/me")) {
          const body = await readBody(request);
          fields(body, ["request_id", "used_tokens"]);
          return reply(mutateDevice.immediate(device, body, path));
        }
      }
      throw new ApiError("接口不存在或请求方法不允许", 404);
    } catch (error) {
      if (device && error instanceof ApiError && error.event) {
        // This separate transaction survives rejection/rollback of the accounting transaction.
        try { db.transaction(() => db.query("INSERT INTO enterprise_events (company_id,device_id,kind,created_at) VALUES (?,?,?,?)").run(device!.company_id, device!.id, error.event!, now())).immediate(); }
        catch { return reply({ error: "异常记录保存失败，请稍后重试" }, 503); }
      }
      return reply({ error: error instanceof ApiError ? error.message : "企业授权服务暂时不可用" }, error instanceof ApiError ? error.status : 503);
    }
  } };
}
