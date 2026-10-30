import { Database } from "bun:sqlite";
import { createHash } from "node:crypto";
import { AdError, appManagementAuth } from "./app-ad";

const DAY = 86400000, RETENTION = 7 * DAY, MAX_BODY = 1024;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const date = (time: number) => new Date(time + 8 * 3600000).toISOString().slice(0, 10);
const hash = (value: string) => createHash("sha256").update(value.toLowerCase()).digest("hex");
function allowedOrigin(origin: string | null) {
  if (origin === "https://ai.maihh.net") return true;
  try {
    const url = new URL(origin || "");
    return url.origin === origin && ["http:", "https:"].includes(url.protocol)
      && ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname);
  } catch { return false; }
}
async function input(request: Request) {
  if (!/^application\/json(?:\s*;.*)?$/i.test(request.headers.get("content-type") || "")) throw new AdError("需要 JSON / JSON required", 415);
  const length = request.headers.get("content-length");
  if (length && (!/^\d+$/.test(length) || Number(length) > MAX_BODY)) throw new AdError("请求过大 / Request too large", 413);
  const reader = request.body?.getReader(), chunks: Uint8Array[] = [];
  let size = 0;
  if (reader) try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) { await reader.cancel(); throw new AdError("请求过大 / Request too large", 413); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  let value: Record<string, unknown>;
  try { value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks))); }
  catch { throw new AdError("JSON 无效 / Invalid JSON"); }
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join() !== "event_id,install_id,schema"
    || value.schema !== 1 || typeof value.install_id !== "string" || !UUID.test(value.install_id)
    || typeof value.event_id !== "string" || !UUID.test(value.event_id)) throw new AdError("启动记录无效 / Invalid start event");
  return { install: hash(value.install_id), event: hash(value.event_id) };
}

export function createAppStarts(db: Database, adminKey: string) {
  const authorize = appManagementAuth(adminKey);
  db.exec(`CREATE TABLE IF NOT EXISTS app_start_totals (id INTEGER PRIMARY KEY CHECK(id=1), starts INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS app_start_installs (install_hash TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS app_start_events (event_hash TEXT PRIMARY KEY, received_at INTEGER NOT NULL);
    CREATE INDEX IF NOT EXISTS app_start_event_time ON app_start_events(received_at);
    CREATE TABLE IF NOT EXISTS app_start_daily (day TEXT NOT NULL, install_hash TEXT NOT NULL, starts INTEGER NOT NULL, PRIMARY KEY(day,install_hash));`);
  const record = db.transaction((install: string, event: string, now: number) => {
    db.query("DELETE FROM app_start_events WHERE received_at<?").run(now - RETENTION);
    db.query("DELETE FROM app_start_daily WHERE day<?").run(date(now - 6 * DAY));
    const added = db.query("INSERT OR IGNORE INTO app_start_events VALUES (?,?)").run(event, now).changes;
    if (!added) return false;
    db.query("INSERT INTO app_start_totals VALUES (1,1) ON CONFLICT(id) DO UPDATE SET starts=starts+1").run();
    db.query("INSERT OR IGNORE INTO app_start_installs VALUES (?)").run(install);
    db.query("INSERT INTO app_start_daily VALUES (?,?,1) ON CONFLICT(day,install_hash) DO UPDATE SET starts=starts+1").run(date(now), install);
    return true;
  });
  const totals = (from: string, to: string) => db.query<{ starts: number; unique_installations: number }, [string, string]>(
    "SELECT COALESCE(SUM(starts),0) AS starts,COUNT(DISTINCT install_hash) AS unique_installations FROM app_start_daily WHERE day BETWEEN ? AND ?").get(from, to)!;
  return {
    async handle(request: Request): Promise<Response | null> {
      const path = new URL(request.url).pathname;
      if (path !== "/api/app-start" && path !== "/api/admin/app-starts") return null;
      const origin = request.headers.get("origin"), publicAccess = path === "/api/app-start";
      const cors = publicAccess && allowedOrigin(origin) ? { "Access-Control-Allow-Origin": origin!, "Vary": "Origin" } : {};
      const reply = (value: unknown, status = 200) => Response.json(value, { status, headers: {
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'", ...cors,
      } });
      try {
        if (publicAccess) {
          if (!allowedOrigin(origin)) throw new AdError("来源不允许 / Origin not allowed", 403);
          if (request.headers.has("authorization")) throw new AdError("公共接口不接受凭证 / Public endpoint does not accept credentials");
          if (request.method === "OPTIONS") {
            const headers = request.headers.get("access-control-request-headers") || "";
            if (request.headers.get("access-control-request-method") !== "POST" || headers.split(",").some(item => item.trim().toLowerCase() !== "content-type")) throw new AdError("预检无效 / Invalid preflight", 403);
            return new Response(null, { status: 204, headers: { ...cors, "Access-Control-Allow-Methods": "POST", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "600", "Cache-Control": "no-store" } });
          }
          if (request.method !== "POST") return reply({ error: "方法不允许 / Method not allowed" }, 405);
          const value = await input(request);
          const added = record(value.install, value.event, Date.now());
          return reply({ schema: 1, ok: true, duplicate: !added });
        }
        authorize(request);
        if (request.method !== "GET") return reply({ error: "方法不允许 / Method not allowed" }, 405);
        const now = Date.now(), today = date(now), from = date(now - 6 * DAY);
        const days = Array.from({ length: 7 }, (_, index) => {
          const day = date(now - (6 - index) * DAY); return { date: day, ...totals(day, day) };
        });
        return reply({ schema: 1, timezone: "Asia/Shanghai", generated_at: new Date(now).toISOString(),
          totals: { starts: db.query<{ starts: number }, []>("SELECT starts FROM app_start_totals WHERE id=1").get()?.starts || 0,
            unique_installations: db.query<{ count: number }, []>("SELECT COUNT(*) AS count FROM app_start_installs").get()!.count },
          today: { date: today, ...totals(today, today) },
          last_7_days: { from, to: today, ...totals(from, today), days }, deduplication_window_days: 7 });
      } catch (error) {
        if (error instanceof AdError) return reply({ error: error.message }, error.status);
        console.error("Application start statistics unavailable", error instanceof Error ? error.name : "unknown");
        return reply({ error: "启动统计暂时不可用 / Start statistics unavailable" }, 503);
      }
    },
  };
}
