import { Database } from "bun:sqlite";
import { test, expect } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createAppAd } from "../website/app-ad";

const key = "a".repeat(64);
const base = "https://ai.maihh.net";
const image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
const draft = { schema: 1, title: { zh: "鲁班零号", en: "Luban Zero" }, subtitle: { zh: "共同成长", en: "Grow together" }, cta: { zh: "了解更多", en: "Learn more" }, href: base + "/?lang=zh" };
function fixture() {
  const db = new Database(":memory:");
  const api = createAppAd(db, key);
  return { db, api, request: (method = "GET", value?: unknown, headers: Record<string, string> = {}, path = "/api/admin/app-ad") => api.handle(new Request(base + path, { method, headers: { authorization: `Bearer ${key}`, ...(value === undefined ? {} : { "content-type": "application/json" }), ...headers }, ...(value === undefined ? {} : { body: JSON.stringify(value) }) })) as Promise<Response> };
}

test("公开广告默认含双语和安全图片；管理接口不泄露凭证", async () => {
  const { db, request } = fixture();
  try {
    const response = await request("GET", undefined, { authorization: "" }, "/api/app-ad");
    expect(response.status).toBe(200);
    expect(response.headers.get("access-control-allow-origin")).toBe("*");
    expect(response.headers.get("cache-control")).toBe("no-store");
    const ad = await response.json();
    expect(ad.title).toEqual(draft.title);
    expect(ad.image).toBe("/app-ad/luban-intro-v1.png");
    expect(JSON.stringify(ad)).not.toContain(key);
    for (const authorization of ["", `Bearer ${"b".repeat(64)}`, "Bearer short", key]) {
      expect((await request("GET", undefined, { authorization })).status).toBe(401);
      expect((await request("PUT", draft, { authorization })).status).toBe(401);
    }
    expect((await request("PUT", draft, { origin: "https://evil.example" })).status).toBe(403);
    expect((await request("PUT", draft, { "sec-fetch-site": "cross-site" })).status).toBe(403);
    expect((await request("GET")).headers.get("access-control-allow-origin")).toBeNull();
    expect((await request("POST", draft)).status).toBe(405);
  } finally { db.close(); }
});

test("带凭证更新、PNG 公共读取和重开数据库持久化", async () => {
  const root = mkdtempSync(join(tmpdir(), "luban-ad-api-")), file = join(root, "ad.sqlite");
  let db = new Database(file);
  try {
    let api = createAppAd(db, key);
    const response = await api.handle(new Request(base + "/api/admin/app-ad", { method: "PUT", headers: { authorization: `Bearer ${key}`, "content-type": "application/json", origin: base }, body: JSON.stringify({ ...draft, png_base64: image }) }));
    expect(response!.status).toBe(200);
    const ad = await response!.json();
    expect(ad.image).toMatch(/^\/api\/app-ad\/image\/[0-9a-f]{64}\.png$/);
    db.close(); db = new Database(file); api = createAppAd(db, key);
    expect(await (await api.handle(new Request(base + "/api/app-ad")))!.json()).toEqual(ad);
    const png = await api.handle(new Request(base + ad.image));
    expect(png!.status).toBe(200);
    expect(png!.headers.get("content-type")).toBe("image/png");
    expect(png!.headers.get("access-control-allow-origin")).toBe("*");
    expect(png!.headers.get("cache-control")).toContain("immutable");
    expect(Buffer.from(await png!.arrayBuffer()).toString("base64")).toBe(image);
    const head = await api.handle(new Request(base + ad.image, { method: "HEAD" }));
    expect(await head!.text()).toBe("");
    expect(head!.headers.get("content-length")).toBe(String(Buffer.from(image, "base64").length));
    expect((await api.handle(new Request(base + "/api/app-ad/image/" + "0".repeat(64) + ".png")))!.status).toBe(404);
    expect((await api.handle(new Request(base + "/api/app-ad/image/secret.txt")))!.status).toBe(404);
    const preserve = await api.handle(new Request(base + "/api/admin/app-ad", { method: "PUT", headers: { authorization: `Bearer ${key}`, "content-type": "application/json" }, body: JSON.stringify({ ...draft, title: { zh: "新广告", en: "New advertisement" } }) }));
    expect((await preserve!.json()).image).toBe(ad.image);
  } finally { db.close(); rmSync(root, { recursive: true, force: true }); }
});

test("畸形文案、外链、任意图片路径和 PNG 数据拒绝且旧配置保持", async () => {
  const { db, request } = fixture();
  try {
    for (const changes of [
      { schema: 2 }, { unknown: "value" }, { title: { zh: "只中文" } }, { title: { zh: "", en: "empty" } },
      { subtitle: { zh: "x\u0000", en: "bad" } }, { cta: { zh: "x".repeat(41), en: "long" } },
      { href: "http://ai.maihh.net" }, { href: "https://ai.maihh.net.evil.example/" }, { href: "javascript:alert(1)" }, { href: "https://user:pass@ai.maihh.net/" },
      { image: "file:///etc/passwd" }, { image: "https://evil.example/banner.png" }, { image: "/api/app-ad/image/" + "0".repeat(64) + ".png" },
      { png_base64: "<svg/>" }, { png_base64: Buffer.from("<svg/>").toString("base64") }, { png_base64: image.slice(0, -4) },
      { png_base64: image, image: "/app-ad/luban-intro-v1.png" },
    ]) expect((await request("PUT", { ...draft, ...changes })).status).toBe(400);
    expect((await request("GET")).status).toBe(200);
    expect((await (await request("GET")).json()).image).toBe("/app-ad/luban-intro-v1.png");
    expect((await request("PUT", draft, { "content-type": "text/plain" })).status).toBe(415);
    expect((await request("PUT", { ...draft, png_base64: "a".repeat(3 * 1024 * 1024) })).status).toBe(413);
    expect((await request("PUT", { ...draft, png_base64: Buffer.alloc(2 * 1024 * 1024 + 1).toString("base64") })).status).toBe(413);
  } finally { db.close(); }
});

test("无 Content-Length 的流式超限请求拒绝", async () => {
  const { db, api } = fixture();
  try {
    const request = new Request(base + "/api/admin/app-ad", { method: "PUT", headers: { authorization: `Bearer ${key}`, "content-type": "application/json" }, body: new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(3 * 1024 * 1024 + 1)); controller.close(); } }) });
    expect((await api.handle(request))!.status).toBe(413);
    expect((await api.handle(new Request(base + "/api/app-ad")))!.status).toBe(200);
  } finally { db.close(); }
});
