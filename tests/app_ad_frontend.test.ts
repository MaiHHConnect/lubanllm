import { expect, setSystemTime, test } from "bun:test";
import { initializeAppAd, refreshAppAd, validatedAd } from "../frontend/app-ad";

const config = { schema: 1, title: { zh: "鲁班零号", en: "Luban Zero" },
  subtitle: { zh: "新的广告内容", en: "Updated advertisement" }, cta: { zh: "了解鲁班", en: "Discover" },
  href: "https://ai.maihh.net/?from=app", image: "/api/app-ad/image/" + "a".repeat(64) + ".png" };
test("remote ad rejects injected links, images, malformed text, and incomplete translations", () => {
  expect(validatedAd(config).href).toBe(config.href);
  for (const href of ["javascript:alert(1)", "https://evil.test/", "http://ai.maihh.net/", "https://user@ai.maihh.net/"])
    expect(() => validatedAd({ ...config, href })).toThrow();
  for (const image of ["https://evil.test/a.png", "/api/app-ad/image/../private", "/app-ad/evil.svg", "//evil.test/a.png"])
    expect(() => validatedAd({ ...config, image })).toThrow();
  for (const title of [{ zh: "", en: "A" }, { zh: "中文" }, { zh: "<img>\n", en: "A" }])
    expect(() => validatedAd({ ...config, title })).toThrow();
});

function popupFixture(storage = new Map<string, string>(), storageFails = false) {
  const globals = globalThis as any;
  const old = { fetch: globals.fetch, document: globals.document, Image: globals.Image,
    localStorage: globals.localStorage, setInterval: globals.setInterval, clearInterval: globals.clearInterval };
  const elements = new Map<string, any>();
  const get = (id: string) => {
    if (!elements.has(id)) elements.set(id, Object.assign(new EventTarget(), {
      hidden: true, textContent: "bundled", href: "https://ai.maihh.net/", src: "bundled.png", setAttribute() {},
    }));
    return elements.get(id);
  };
  const intervals = new Map<number, () => void>();
  let requests = 0, sequence = 0;
  globals.document = { hidden: false, documentElement: { lang: "zh-CN" }, getElementById: get };
  globals.localStorage = {
    getItem(key: string) { if (storageFails) throw Error("unavailable"); return storage.get(key) ?? null; },
    setItem(key: string, value: string) { if (storageFails) throw Error("unavailable"); storage.set(key, value); },
  };
  globals.Image = class { async decode() {} };
  globals.fetch = async () => { requests++; return Response.json(config); };
  globals.setInterval = (callback: () => void, ms: number) => {
    expect(ms).toBe(300000); const id = ++sequence; intervals.set(id, callback); return id;
  };
  globals.clearInterval = (id: number) => { intervals.delete(id); };
  let dispose = () => {};
  return {
    storage, intervals, get, get requests() { return requests; },
    start() { dispose = initializeAppAd(); return dispose; },
    close() { get("app-ad-close").dispatchEvent(new Event("click")); },
    restore() { dispose(); Object.assign(globals, old); },
  };
}

test("popup shows once per local calendar day, persists on display, and stays closed after refresh or reinitialization", async () => {
  const previousTZ = process.env.TZ;
  process.env.TZ = "Asia/Shanghai";
  setSystemTime(new Date("2026-10-14T16:05:00Z"));
  const storage = new Map<string, string>();
  let fixture = popupFixture(storage);
  try {
    fixture.start();
    expect(fixture.get("app-ad-popup").hidden).toBe(false);
    // This instant is October 15 locally, but still October 14 in UTC.
    expect([...storage.values()]).toEqual(["2026-10-15"]);
    expect(fixture.requests).toBe(1);
    expect(fixture.intervals.size).toBe(1);
    (globalThis.document as any).hidden = true;
    [...fixture.intervals.values()][0]();
    expect(fixture.requests).toBe(1);
    (globalThis.document as any).hidden = false;
    [...fixture.intervals.values()][0]();
    expect(fixture.requests).toBe(2);
    fixture.close();
    expect(fixture.get("app-ad-popup").hidden).toBe(true);
    expect(fixture.intervals.size).toBe(0);
    fixture.start();
    expect(fixture.get("app-ad-popup").hidden).toBe(true);
    expect(fixture.requests).toBe(2);
    await Bun.sleep(0);
    expect(fixture.get("app-ad-title").textContent).toBe("鲁班零号");
    expect(fixture.get("app-ad-popup").hidden).toBe(true);
    fixture.restore();
    fixture = popupFixture(storage);
    fixture.start();
    expect(fixture.get("app-ad-popup").hidden).toBe(true);
    expect(fixture.requests).toBe(0);
    expect(fixture.intervals.size).toBe(0);
    fixture.restore();
    setSystemTime(new Date("2026-10-15T16:00:00Z"));
    fixture = popupFixture(storage);
    fixture.start();
    expect(fixture.get("app-ad-popup").hidden).toBe(false);
    expect([...storage.values()]).toEqual(["2026-10-16"]);
    expect(fixture.requests).toBe(1);
    await Bun.sleep(0);
  } finally {
    fixture.restore(); setSystemTime();
    if (previousTZ === undefined) delete process.env.TZ; else process.env.TZ = previousTZ;
  }
});

test("unavailable storage still allows only one popup per page and disposal clears its timer", async () => {
  const fixture = popupFixture(new Map(), true);
  try {
    const dispose = fixture.start();
    expect(fixture.get("app-ad-popup").hidden).toBe(false);
    expect(fixture.requests).toBe(1);
    fixture.close(); fixture.start();
    expect(fixture.get("app-ad-popup").hidden).toBe(true);
    expect(fixture.requests).toBe(1);
    expect(fixture.intervals.size).toBe(0);
    dispose(); dispose();
    await Bun.sleep(0);
  } finally { fixture.restore(); }
  const openFixture = popupFixture();
  try {
    const dispose = openFixture.start();
    expect(openFixture.intervals.size).toBe(1);
    dispose();
    expect(openFixture.intervals.size).toBe(0);
    await Bun.sleep(0);
  } finally { openFixture.restore(); }
});
test("public refresh sends no credential, uses text content, and preserves a working ad on failure", async () => {
  const globals = globalThis as any;
  const old = { fetch: globals.fetch, document: globals.document, Image: globals.Image };
  const elements = new Map<string, any>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, { textContent: "bundled", src: "bundled.png", href: "https://ai.maihh.net/", setAttribute() {} }); return elements.get(id); };
  let mode = "valid", requests: RequestInit[] = [];
  try {
    globals.document = { documentElement: { lang: "zh-CN" }, getElementById: get };
    globals.Image = class { src = ""; async decode() { if (mode === "image-fails") throw Error(); } };
    globals.fetch = async (_: string, options: RequestInit) => {
      requests.push(options);
      return mode === "offline" ? new Response("unavailable", { status: 503 })
        : mode === "oversized" ? new Response("x".repeat(16385)) : Response.json(config);
    };
    await refreshAppAd();
    expect(get("app-ad-title").textContent).toBe("鲁班零号");
    expect(get("app-ad-subtitle").textContent).toBe("新的广告内容");
    expect(get("app-ad").href).toBe(config.href);
    expect(requests[0].credentials).toBe("omit");
    expect(requests[0].headers).toBeUndefined();
    expect(requests[0].redirect).toBe("error");
    for (const failure of ["offline", "oversized", "image-fails"]) {
      mode = failure; get("app-ad-title").textContent = "keep previous";
      await refreshAppAd();
      expect(get("app-ad-title").textContent).toBe("keep previous");
    }
  } finally { Object.assign(globals, old); }
});
