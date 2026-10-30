import { Database } from "bun:sqlite";
import { expect, setSystemTime, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAppStarts } from "../website/app-starts";

const ORIGIN = "https://ai.maihh.net", KEY = "a".repeat(64);
const first = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", second = "bbbbbbbb-bbbb-4bbb-9bbb-bbbbbbbbbbbb";
const event = (index: number) => `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`;
function fixture(db = new Database(":memory:")) {
  const api = createAppStarts(db, KEY);
  return { db, api,
    post: (install = first, id = event(1), extra: Record<string, unknown> = {}, headers: Record<string, string> = {}) => api.handle(new Request(ORIGIN + "/api/app-start", {
      method: "POST", headers: { origin: "http://127.0.0.1:18766", "content-type": "application/json", ...headers }, body: JSON.stringify({ schema: 1, install_id: install, event_id: id, ...extra }),
    })) as Promise<Response>,
    stats: (headers: Record<string, string> = {}) => api.handle(new Request(ORIGIN + "/api/admin/app-starts", { headers: { authorization: `Bearer ${KEY}`, ...headers } })) as Promise<Response>,
  };
}

test("starts are durable, event-idempotent, anonymous, and use China calendar days", async () => {
  const root = mkdtempSync(join(tmpdir(), "luban-starts-")), path = join(root, "stats.sqlite");
  let db = new Database(path), f = fixture(db);
  setSystemTime(new Date("2026-10-14T15:59:59Z"));
  try {
    expect((await f.stats()).status).toBe(200);
    expect((await (await f.stats()).json()).totals).toEqual({ starts: 0, unique_installations: 0 });
    expect(await (await f.post()).json()).toEqual({ schema: 1, ok: true, duplicate: false });
    expect(await (await f.post(first.toUpperCase())).json()).toEqual({ schema: 1, ok: true, duplicate: true });
    setSystemTime(new Date("2026-10-14T16:00:00Z"));
    await Promise.all([f.post(first, event(2)), f.post(second, event(3)), f.post(second, event(3))]);
    const response = await f.stats(), stats = await response.json();
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(response.headers.get("access-control-allow-origin")).toBeNull();
    expect(stats.timezone).toBe("Asia/Shanghai");
    expect(stats.totals).toEqual({ starts: 3, unique_installations: 2 });
    expect(stats.today).toEqual({ date: "2026-10-15", starts: 2, unique_installations: 2 });
    expect(stats.last_7_days.starts).toBe(3);
    expect(stats.last_7_days.unique_installations).toBe(2);
    expect(stats.last_7_days.days).toHaveLength(7);
    expect(stats.last_7_days.days[5]).toEqual({ date: "2026-10-14", starts: 1, unique_installations: 1 });
    const stored = JSON.stringify([db.query("SELECT * FROM app_start_events").all(), db.query("SELECT * FROM app_start_installs").all(), db.query("SELECT * FROM app_start_daily").all()]);
    for (const raw of [first, second, event(1), "127.0.0.1"]) expect(stored).not.toContain(raw);
    expect(stored).toMatch(/[a-f0-9]{64}/);
    db.close(); db = new Database(path); f = fixture(db);
    expect((await (await f.stats()).json()).totals).toEqual(stats.totals);
    expect(await (await f.post()).json()).toEqual({ schema: 1, ok: true, duplicate: true });
  } finally { setSystemTime(); db.close(); rmSync(root, { recursive: true, force: true }); }
});

test("seven-day retention bounds event/daily rows without resetting lifetime totals", async () => {
  const f = fixture();
  try {
    setSystemTime(new Date("2026-10-01T00:00:00Z")); await f.post();
    setSystemTime(new Date("2026-10-07T00:00:00Z"));
    expect((await (await f.post()).json()).duplicate).toBe(true);
    setSystemTime(new Date("2026-10-09T00:00:00Z"));
    await f.post(second, event(2));
    expect(f.db.query("SELECT COUNT(*) AS n FROM app_start_events").get()).toEqual({ n: 1 });
    expect(f.db.query("SELECT COUNT(*) AS n FROM app_start_daily").get()).toEqual({ n: 1 });
    const stats = await (await f.stats()).json();
    expect(stats.totals).toEqual({ starts: 2, unique_installations: 2 });
    expect(stats.last_7_days.starts).toBe(1);
    expect(stats.deduplication_window_days).toBe(7);
    expect((await (await f.post()).json()).duplicate).toBe(false);
    expect((await (await f.stats()).json()).totals).toEqual({ starts: 3, unique_installations: 2 });
  } finally { setSystemTime(); f.db.close(); }
});

test("public endpoint validates exact tiny UUID schema and restricted CORS without credentials", async () => {
  const f = fixture();
  try {
    for (const origin of [ORIGIN, "http://127.0.0.1:18766", "http://localhost:8765", "http://[::1]:8765"]) {
      const response = await f.post(first, crypto.randomUUID(), {}, { origin });
      expect(response.status).toBe(200);
      expect(response.headers.get("access-control-allow-origin")).toBe(origin);
    }
    for (const origin of ["", "null", "http://ai.maihh.net", "https://evil.example", "http://127.0.0.1.evil.example", "http://localhost:8765/path"]) {
      const response = await f.post(first, event(9), {}, { origin });
      expect(response.status).toBe(403);
      expect(response.headers.has("access-control-allow-origin")).toBe(false);
    }
    for (const extra of [{ schema: 2 }, { user: "name" }, { install_id: first + "x" }, { event_id: "not-a-uuid" }, { event_id: "00000000-0000-0000-0000-000000000000" }]) {
      expect((await f.post(first, event(10), extra)).status).toBe(400);
    }
    expect((await f.post(first, event(11), {}, { authorization: `Bearer ${KEY}` })).status).toBe(400);
    expect((await f.post(first, event(11), {}, { "content-type": "text/plain" })).status).toBe(415);
    expect((await f.post(first, event(11), { extra: "x".repeat(1024) })).status).toBe(413);
    const streamed = new Request(ORIGIN + "/api/app-start", { method: "POST", headers: { origin: ORIGIN, "content-type": "application/json" }, body: new ReadableStream({ start(c) { c.enqueue(new Uint8Array(1025)); c.close(); } }) });
    expect((await f.api.handle(streamed))!.status).toBe(413);
    const preflight = await f.api.handle(new Request(ORIGIN + "/api/app-start", { method: "OPTIONS", headers: {
      origin: "http://127.0.0.1:18766", "access-control-request-method": "POST", "access-control-request-headers": "content-type",
    } }));
    expect(preflight!.status).toBe(204);
    expect(preflight!.headers.get("access-control-allow-headers")).toBe("Content-Type");
    expect(preflight!.headers.has("access-control-allow-credentials")).toBe(false);
    expect((await f.api.handle(new Request(ORIGIN + "/api/app-start", { headers: { origin: ORIGIN } })))!.status).toBe(405);
    expect((await (await f.stats()).json()).totals.starts).toBe(4);
  } finally { f.db.close(); }
});

test("statistics require the existing independent management key and reject browser cross-origin access", async () => {
  const f = fixture();
  try {
    for (const authorization of ["", "Bearer short", "Bearer " + "b".repeat(64)]) expect((await f.stats({ authorization })).status).toBe(401);
    expect((await f.stats({ origin: "https://evil.example" })).status).toBe(403);
    expect((await f.stats({ "sec-fetch-site": "cross-site" })).status).toBe(403);
    expect((await f.stats({ origin: ORIGIN })).status).toBe(200);
    const wrongMethod = await f.api.handle(new Request(ORIGIN + "/api/admin/app-starts", { method: "POST", headers: { authorization: `Bearer ${KEY}` } }));
    expect(wrongMethod!.status).toBe(405);
    expect(await f.api.handle(new Request(ORIGIN + "/api/unrelated"))).toBeNull();
  } finally { f.db.close(); }
});
