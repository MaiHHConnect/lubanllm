import { t, locale } from "../frontend/i18n";
import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { COUNTRY_CODES, countryLabel } from "../frontend/countries";

const source = readFileSync(new URL("../frontend/app.ts", import.meta.url), "utf8");
const html = readFileSync(new URL("../frontend/index.html", import.meta.url), "utf8");
// Exercise the real ranking controller in a minimal DOM, without browser/network access.
const rankingCode = new Bun.Transpiler({ loader: "ts" }).transformSync(
  source.slice(source.indexOf("function scoreText"), source.indexOf('\nfor (const id of ["evolution-mode')),
);
class Element {
  textContent = "";
  value = "";
  hidden = false;
  disabled = false;
  children: Element[] = [];
  attributes: Record<string, string> = {};
  listeners: Record<string, (event: { preventDefault(): void }) => void> = {};
  append(child: Element) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  setAttribute(key: string, value: string) { this.attributes[key] = value; }
  removeAttribute(key: string) { delete this.attributes[key]; }
  reportValidity() { return true; }
  addEventListener(type: string, action: (event: { preventDefault(): void }) => void) { this.listeners[type] = action; }
}
function controller(total = 101) {
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  const calls: { path: string; data?: any; method?: string }[] = [];
  const profile = { username: "tester", country_code: "CN", evolution_count: 1, count_error: null, published_profile: null };
  const api = async (path: string, data?: any, method?: string) => {
    calls.push({ path, data, method });
    if (method === "PUT") return { profile: { ...profile, ...data } };
    if (path.endsWith("/start")) return { state: "queued" };
    const page = Number(new URL(path, "http://localhost").searchParams.get("page"));
    const entries = Array.from({ length: Math.max(0, Math.min(100, total - (page - 1) * 100)) }, (_, i) => ({
      username: `row-${i}`, country_code: "US", evolution_count: 0, rank: (page - 1) * 100 + i + 1,
      growth: { domain: "mathematics", eligible: true, net_delta: 0, question_count: 24, retention: 100 },
    }));
    return { profile, entries, total, page, page_size: 100, error: null };
  };
  const run = new Function("$", "input", "select", "button", "text", "count", "api", "COUNTRY_CODES", "countryLabel", "document", "t", "locale", `
    let rankingBusy=false, rankingLoaded=false, rankingCountryEdited=false, rankingPage=1, rankingTotal=0, rankingCountError=null, rankingSort="growth", rankingDomain="mathematics", growthState, benchmarkState, benchmarkRefreshPending=false;
    ${rankingCode}
    return {loadRanking, rankingOperation};
  `);
  const actions = run(get, get, get, get, (id: string, text: string) => { get(id).textContent = text; },
    (n: number) => String(n), api, COUNTRY_CODES, countryLabel, { createElement: () => new Element() }, t, locale);
  return { get, calls, ...actions };
}
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

test("all official ISO regions have flags/names and API whitelist stays aligned", () => {
  expect(COUNTRY_CODES.length).toBe(249);
  expect(new Set(COUNTRY_CODES).size).toBe(249);
  const python = readFileSync(new URL("../src/luban/leaderboard.py", import.meta.url), "utf8");
  const apiCodes = python.match(/COUNTRY_CODES = frozenset\("([A-Z ]+)"\.split\(\)\)/)![1].split(" ");
  expect(apiCodes).toEqual(COUNTRY_CODES);
  expect(countryLabel("CN").startsWith("🇨🇳 中国")).toBe(true);
  expect(countryLabel("US")).toBe("🇺🇸 美国");
  expect(countryLabel(null)).toBe("未选择");
  expect(countryLabel("XX")).toBe("未选择");
});

test("ranking is lazy, fetches only the selected page, and shows pagination after 100", async () => {
  const app = controller();
  expect(app.calls).toEqual([]);
  expect(source.slice(source.indexOf("async function initialize()"))).not.toContain("loadRanking");
  await app.loadRanking();
  expect(app.calls.map(call => call.path)).toEqual(["/api/leaderboard?page=1&sort=growth&domain=mathematics"]);
  expect(app.get("leaderboard-entries").children.length).toBe(100);
  expect(app.get("leaderboard-pagination").hidden).toBe(false);
  expect(app.get("leaderboard-previous").disabled).toBe(true);
  expect(app.get("leaderboard-next").disabled).toBe(false);
  expect(app.get("leaderboard-country").value).toBe("CN");
  app.get("leaderboard-next").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.calls.map(call => call.path)).toEqual(["/api/leaderboard?page=1&sort=growth&domain=mathematics", "/api/leaderboard?page=2&sort=growth&domain=mathematics"]);
  expect(app.get("leaderboard-entries").children.length).toBe(1);
  expect(app.get("leaderboard-page").textContent).toBe("2 / 2");
  expect(app.get("leaderboard-next").disabled).toBe(true);
  expect(app.get("leaderboard-entries").children[0].children[2].textContent).toBe("🇺🇸 美国");
  app.get("leaderboard-previous").listeners.click({ preventDefault() {} });
  await settle();
  expect(app.calls.at(-1)!.path).toBe("/api/leaderboard?page=1&sort=growth&domain=mathematics");
});

test("100 or fewer rows hide paging and the UI has no withdrawal action", async () => {
  for (const total of [0, 1, 100]) {
    const app = controller(total);
    await app.loadRanking();
    expect(app.get("leaderboard-pagination").hidden).toBe(true);
  }
  expect(html).not.toContain("leaderboard-unpublish");
  expect(rankingCode).not.toContain("leaderboard-unpublish");
  expect(html).toContain('maxlength="32" required autocomplete="nickname" placeholder="1–16 个字符"');
});

test("manual region selection is saved before queuing the local evaluation", async () => {
  const app = controller();
  await app.loadRanking(2);
  app.get("leaderboard-username").value = "new";
  app.get("leaderboard-country").value = "GB";
  app.get("leaderboard-form").listeners.submit({ preventDefault() {} });
  await settle();
  expect(app.calls.at(-2)).toEqual({ path: "/api/leaderboard/profile", data: { username: "new", country_code: "GB" }, method: "PUT" });
  expect(app.calls.at(-1)!.path).toBe("/api/growth/start");
  expect(app.get("leaderboard-page").textContent).toBe("2 / 2");
  expect(app.get("leaderboard-publish").disabled).toBe(true);
});

test("page emptied by a smaller ranking returns to first page without prefetching", async () => {
  const app = controller(1);
  await app.loadRanking(2);
  expect(app.calls.map(call => call.path)).toEqual(["/api/leaderboard?page=2&sort=growth&domain=mathematics", "/api/leaderboard?page=1&sort=growth&domain=mathematics"]);
  expect(app.get("leaderboard-entries").children.length).toBe(1);
});
