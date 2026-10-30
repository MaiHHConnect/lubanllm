import { expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createEnterprise } from "./enterprise";

const origin = "https://ai.maihh.net", key = "enterprise-admin-fixture-key-0123456789";
const policy = "inference_input_output_v1";
const id = () => crypto.randomUUID();
function fixture(path = ":memory:") {
  const db = new Database(path, { create: true });
  db.exec("PRAGMA busy_timeout=5000");
  let time = Date.UTC(2026, 9, 9);
  const app = createEnterprise(db, { adminKey: key, now: () => time });
  let cookie = "";
  async function request(route: string, method = "GET", body?: unknown, token?: string, suppliedOrigin: string | null = origin) {
    const headers = new Headers();
    if (suppliedOrigin !== null) headers.set("origin", suppliedOrigin);
    if (cookie) headers.set("cookie", cookie);
    if (token) headers.set("authorization", `Bearer ${token}`);
    if (body !== undefined) headers.set("content-type", "application/json");
    const response = await app.handle(new Request(`${origin}/api/enterprise${route}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) }));
    expect(response).not.toBeNull();
    expect(response!.headers.get("cache-control")).toBe("no-store");
    return { response: response!, data: await response!.json() as any };
  }
  async function login() {
    const result = await request("/admin/login", "POST", { key });
    expect(result.response.status).toBe(200);
    cookie = result.response.headers.get("set-cookie")!.split(";")[0];
    return result;
  }
  async function create(tokens = 100) {
    const body = { request_id: id(), name: "测试企业", annual_tokens: tokens, billing_policy: policy };
    const result = await request("/admin/companies", "POST", body);
    expect(result.response.status).toBe(201);
    return { ...result.data, body };
  }
  async function activate(code: string, deviceId = id()) {
    const result = await request("/activate", "POST", { license_code: code, device_id: deviceId, device_name: "测试设备" });
    expect(result.response.status).toBe(201);
    return result.data;
  }
  return { db, app, request, login, create, activate, advance: (amount: number) => { time += amount; } };
}

test("admin origin/session boundaries, strict JSON and logout", async () => {
  const f = fixture();
  try {
    expect((await f.request("/admin/companies")).response.status).toBe(401);
    expect((await f.request("/admin/login", "POST", { key }, undefined, null)).response.status).toBe(403);
    expect((await f.request("/admin/login", "POST", { key }, undefined, "https://evil.example")).response.status).toBe(403);
    expect((await f.request("/admin/login", "POST", { key: "bad" })).response.status).toBe(401);
    const login = await f.login();
    const cookie = login.response.headers.get("set-cookie")!;
    for (const flag of ["HttpOnly", "Secure", "SameSite=Strict", "Path=/api/enterprise/admin", "Max-Age=28800"]) expect(cookie).toContain(flag);
    expect((await f.request("/admin/companies", "POST", { request_id: id(), name: "bad", annual_tokens: 1, billing_policy: policy, extra: true })).response.status).toBe(400);
    expect((await f.request("/admin/companies", "POST", { request_id: id(), name: "x".repeat(17_000), annual_tokens: 1, billing_policy: policy })).response.status).toBe(413);
    expect((await f.request("/admin/logout", "POST", {})).response.status).toBe(200);
    expect((await f.request("/admin/companies")).response.status).toBe(401);
    await f.login(); f.advance(8 * 60 * 60_000);
    expect((await f.request("/admin/companies")).response.status).toBe(401);
    expect(await f.app.handle(new Request(`${origin}/other`))).toBeNull();
  } finally { f.db.close(); }
});

test("localhost fixture cookie and auth enumeration rate limits", async () => {
  const db = new Database(":memory:");
  const app = createEnterprise(db, { adminKey: key, origin: "http://localhost:8123" });
  try {
    const response = await app.handle(new Request("http://localhost:8123/api/enterprise/admin/login", { method: "POST", headers: { origin: "http://localhost:8123", "content-type": "application/json" }, body: JSON.stringify({ key }) }));
    expect(response!.status).toBe(200); expect(response!.headers.get("set-cookie")).not.toContain("Secure");
    for (let i = 0; i < 20; i++) {
      const invalid = await app.handle(new Request("http://localhost:8123/api/enterprise/admin/login", { method: "POST", headers: { origin: "http://localhost:8123", "content-type": "application/json" }, body: JSON.stringify({ key: "x" }) }));
      expect(invalid!.status).toBe(i === 19 ? 429 : 401);
    }
  } finally { db.close(); }
});

test("one-time codes, canonical idempotency, ledger, rotation and renewal", async () => {
  const f = fixture();
  try {
    await f.login(); const c = await f.create();
    const retry = await f.request("/admin/companies", "POST", { billing_policy: policy, annual_tokens: 100, name: "测试企业", request_id: c.body.request_id });
    expect(retry.data.code_available).toBe(false); expect(retry.data.license_code).toBeUndefined();
    expect((await f.request("/admin/companies", "POST", { ...c.body, annual_tokens: 101 })).response.status).toBe(409);
    const activated = await f.activate(c.license_code);
    const base = `/admin/companies/${c.company.id}`;
    const topup = { request_id: id(), tokens: 50, note: "补充" };
    expect((await f.request(`${base}/topups`, "POST", topup)).data.company.purchased_tokens).toBe(150);
    expect((await f.request(`${base}/topups`, "POST", topup)).data.company.purchased_tokens).toBe(150);
    const renewed = await f.request(`${base}/renew`, "POST", { request_id: id(), annual_tokens: 200, note: "续年" });
    expect(renewed.data.company.purchased_tokens).toBe(350);
    expect(renewed.data.company.expires_at).toBe("2028-10-09T00:00:00.000Z");
    const rotate = { request_id: id(), note: "更换" };
    const rotated = await f.request(`${base}/rotate`, "POST", rotate);
    expect(rotated.data.license_code).not.toBe(c.license_code);
    expect((await f.request(`${base}/rotate`, "POST", rotate)).data.license_code).toBeUndefined();
    expect((await f.request("/activate", "POST", { license_code: c.license_code, device_id: id(), device_name: "旧码" })).response.status).toBe(401);
    expect((await f.request("/me", "GET", undefined, activated.device_token)).response.status).toBe(200);
    await f.activate(rotated.data.license_code);
    const detail = (await f.request(base)).data;
    expect(detail.devices.length).toBe(2);
    expect(detail.ledger.reduce((sum: number, entry: any) => sum + entry.purchased_delta, 0)).toBe(350);
    expect(detail.devices[0].token_hash).toBeUndefined(); expect(detail.company.license_hash).toBeUndefined();
    expect(() => f.db.query("UPDATE enterprise_ledger SET note='changed'").run()).toThrow("append-only");
  } finally { f.db.close(); }
});

test("soft quota permits overage and expired usage/activation; suspension only stops new activations", async () => {
  const f = fixture();
  try {
    await f.login(); const c = await f.create(10), d = await f.activate(c.license_code);
    const used = await f.request("/usage", "POST", { request_id: id(), used_tokens: 2_000_000 }, d.device_token);
    expect(used.response.status).toBe(200);
    expect(used.data.company.available_tokens).toBe(-1_999_990); expect(used.data.company.remaining_tokens).toBe(0); expect(used.data.company.overage_tokens).toBe(1_999_990);
    f.advance(400 * 86_400_000); await f.activate(c.license_code);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 2_000_001 }, d.device_token)).response.status).toBe(200);
    const base = `/admin/companies/${c.company.id}`;
    await f.login();
    await f.request(`${base}/status`, "POST", { request_id: id(), status: "suspended", note: "暂停新增" });
    expect((await f.request("/activate", "POST", { license_code: c.license_code, device_id: id(), device_name: "新设备" })).response.status).toBe(403);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 2_000_002 }, d.device_token)).response.status).toBe(200);
    expect((await f.request("/me", "GET", undefined, d.device_token)).data.usage_allowed).toBe(true);
    expect((await f.request(base)).data.warnings.length).toBe(3);
  } finally { f.db.close(); }
});

test("activation cannot rotate existing device credentials or cross-company identity", async () => {
  const f = fixture();
  try {
    await f.login(); const a = await f.create(), b = await f.create();
    const deviceId = id(), d = await f.activate(a.license_code, deviceId);
    for (const code of [a.license_code, b.license_code]) expect((await f.request("/activate", "POST", { license_code: code, device_id: deviceId, device_name: "同设备" })).response.status).toBe(409);
    const body = { request_id: id(), used_tokens: 100, device_id: id() };
    expect((await f.request("/usage", "POST", body, d.device_token)).response.status).toBe(400);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 100 }, "0".repeat(64))).response.status).toBe(401);
    expect((await f.request("/me", "GET", undefined, d.device_token, null)).data.company.id).toBe(a.company.id);
  } finally { f.db.close(); }
});

test("cumulative usage delta, retries and rejected monotonic writes preserve ledger and events", async () => {
  const f = fixture();
  try {
    await f.login(); const c = await f.create(), a = await f.activate(c.license_code), b = await f.activate(c.license_code);
    const first = { request_id: id(), used_tokens: 20 };
    expect((await f.request("/usage", "POST", first, a.device_token)).data.usage.delta).toBe(20);
    expect((await f.request("/usage", "POST", first, a.device_token)).data.usage.delta).toBe(20);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 25 }, a.device_token)).data.usage.delta).toBe(5);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 25 }, a.device_token)).data.usage.delta).toBe(0);
    expect((await f.request("/usage", "POST", { ...first, used_tokens: 30 }, a.device_token)).response.status).toBe(409);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 24 }, a.device_token)).response.status).toBe(409);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 10 }, b.device_token)).response.status).toBe(200);
    const detail = (await f.request(`/admin/companies/${c.company.id}`)).data;
    expect(detail.company.used_tokens).toBe(35); expect(detail.usage.length).toBe(4);
    expect(detail.devices.reduce((sum: number, d: any) => sum + d.used_tokens, 0)).toBe(35);
    expect(detail.ledger.reduce((sum: number, d: any) => sum + d.used_delta, 0)).toBe(35);
    expect(detail.events.map((e: any) => e.kind).sort()).toEqual(["idempotency_conflict", "usage_decrease"]);
  } finally { f.db.close(); }
});

test("multiple database connections serialize device lifetime increments and safe company cap", async () => {
  const root = mkdtempSync(join(tmpdir(), "luban-enterprise-concurrency-"));
  const path = join(root, "enterprise.sqlite"), a = fixture(path), b = fixture(path);
  try {
    await a.login(); const c = await a.create(), d1 = await a.activate(c.license_code), d2 = await a.activate(c.license_code);
    const reports = await Promise.all([
      a.request("/usage", "POST", { request_id: id(), used_tokens: 6e14 }, d1.device_token),
      b.request("/usage", "POST", { request_id: id(), used_tokens: 4e14 }, d2.device_token),
    ]);
    expect(reports.map(r => r.response.status)).toEqual([200, 200]);
    const fail = await b.request("/usage", "POST", { request_id: id(), used_tokens: 4e14 + 1 }, d2.device_token);
    expect(fail.response.status).toBe(409);
    const detail = (await a.request(`/admin/companies/${c.company.id}`)).data;
    expect(detail.company.used_tokens).toBe(1e15); expect(detail.devices.reduce((sum: number, d: any) => sum + d.used_tokens, 0)).toBe(1e15);
    expect(detail.ledger.reduce((sum: number, d: any) => sum + d.used_delta, 0)).toBe(1e15);
    expect(detail.events[0].kind).toBe("usage_total_exceeded");
    expect((await a.request("/usage", "POST", { request_id: id(), used_tokens: 1e15 + 1 }, d1.device_token)).response.status).toBe(400);
    expect((await a.request("/usage", "POST", { request_id: id(), used_tokens: 0.5 }, d1.device_token)).response.status).toBe(400);
    expect((await a.request(`/admin/companies/${c.company.id}/topups`, "POST", { request_id: id(), tokens: 1e15, note: "越界" })).response.status).toBe(400);
    expect((await a.request(`/admin/companies/${c.company.id}`)).data.company.purchased_tokens).toBe(100);
  } finally { a.db.close(); b.db.close(); rmSync(root, { recursive: true, force: true }); }
});

test("persistent restore excludes plaintext secrets and retains idempotency", async () => {
  const root = mkdtempSync(join(tmpdir(), "luban-enterprise-restore-")), path = join(root, "db.sqlite");
  const first = fixture(path);
  let secret = "", token = "", usage: any, companyId = "";
  try {
    await first.login(); const c = await first.create(); secret = c.license_code; companyId = c.company.id;
    token = (await first.activate(secret)).device_token; usage = { request_id: id(), used_tokens: 999 };
    await first.request("/usage", "POST", usage, token);
  } finally { first.db.close(); }
  const content = readFileSync(path);
  for (const value of [key, secret, token]) expect(content.includes(Buffer.from(value))).toBe(false);
  const restored = fixture(path);
  try {
    expect((await restored.request("/me", "GET", undefined, token)).data.company.used_tokens).toBe(999);
    expect((await restored.request("/usage", "POST", usage, token)).response.status).toBe(200);
    await restored.login(); const detail = (await restored.request(`/admin/companies/${companyId}`)).data;
    expect(detail.usage.length).toBe(1); expect(detail.company.used_tokens).toBe(999);
    expect(detail.company.reserved_tokens).toBeUndefined();
  } finally { restored.db.close(); rmSync(root, { recursive: true, force: true }); }
});

test("no anonymous counters or team-size collection endpoints exist", async () => {
  const f = fixture();
  try {
    await f.login(); const c = await f.create();
    expect((await f.request("/admin/companies", "POST", { ...c.body, request_id: id(), team_size: 49 })).response.status).toBe(400);
    expect((await f.request(`/admin/companies/${c.company.id}/team`, "POST", { request_id: id(), team_size: 50, note: "人数" })).response.status).toBe(404);
    expect((await f.request("/usage", "POST", { request_id: id(), used_tokens: 5000 })).response.status).toBe(401);
    const d = await f.activate(c.license_code);
    expect(d.reporting_required).toBe(true); expect(d.usage_allowed).toBe(true);
    const me = (await f.request("/me", "GET", undefined, d.device_token)).data;
    expect(me.reporting_required).toBe(true); expect(me.usage_allowed).toBe(true);
    expect(me.company.team_size).toBeUndefined(); expect(me.company.plan).toBeUndefined(); expect(me.company.billable_tokens).toBeUndefined();
    expect((await f.request(`/admin/companies/${c.company.id}`)).data.usage).toEqual([]);
  } finally { f.db.close(); }
});

test("independent processes contend for the exact safe integer total without lost updates", async () => {
  const root = mkdtempSync(join(tmpdir(), "luban-enterprise-process-race-")), path = join(root, "db.sqlite"), f = fixture(path);
  try {
    await f.login(); const c = await f.create(), d1 = await f.activate(c.license_code), d2 = await f.activate(c.license_code);
    async function report(token: string) {
      const source = `import {Database} from "bun:sqlite";
        import {createEnterprise} from ${JSON.stringify(join(import.meta.dir, "enterprise.ts"))};
        const db=new Database(${JSON.stringify(path)}); db.exec("PRAGMA busy_timeout=5000");
        const app=createEnterprise(db,{adminKey:${JSON.stringify(key)}});
        const response=await app.handle(new Request(${JSON.stringify(`${origin}/api/enterprise/usage`)}, {
          method:"POST",headers:{"content-type":"application/json",authorization:${JSON.stringify(`Bearer ${token}`)}},
          body:JSON.stringify({request_id:crypto.randomUUID(),used_tokens:600000000000000})
        })); console.log(JSON.stringify({status:response.status,body:await response.json()})); db.close();`;
      const process = Bun.spawn([Bun.argv[0], "-e", source], { stdout: "pipe", stderr: "pipe" });
      const output = await new Response(process.stdout).text(), error = await new Response(process.stderr).text();
      expect(await process.exited).toBe(0); expect(error).toBe("");
      return JSON.parse(output);
    }
    const results = await Promise.all([report(d1.device_token), report(d2.device_token)]);
    expect(results.map(result => result.status).sort()).toEqual([200, 409]);
    const detail = (await f.request(`/admin/companies/${c.company.id}`)).data;
    expect(detail.company.used_tokens).toBe(6e14);
    expect(detail.devices.reduce((sum: number, d: any) => sum + d.used_tokens, 0)).toBe(6e14);
    expect(detail.usage.length).toBe(1); expect(detail.events[0].kind).toBe("usage_total_exceeded");
  } finally { f.db.close(); rmSync(root, { recursive: true, force: true }); }
});
