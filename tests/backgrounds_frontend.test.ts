import { expect, test } from "bun:test";
import { initializeBackgrounds } from "../frontend/backgrounds";

class Element extends EventTarget {
  type = ""; className = ""; dataset: Record<string, string> = {}; textContent = "";
  src = ""; alt = ""; loading = ""; hidden = false; disabled = false; open = false;
  children: Element[] = []; attributes = new Map<string, string>(); replacements = 0;
  classes = new Set<string>();
  classList = { toggle: (name: string, enabled: boolean) => enabled ? this.classes.add(name) : this.classes.delete(name) };
  append(...children: Element[]) { this.children.push(...children); }
  replaceChildren(...children: Element[]) { this.children = children; this.replacements++; }
  getAttribute(name: string) { return this.attributes.get(name) ?? null; }
  setAttribute(name: string, value: string) { this.attributes.set(name, value); if (name === "src") this.src = value; }
  showModal() { this.open = true; }
  close() { this.open = false; }
  click() { this.dispatchEvent(new Event("click")); }
}
function fixture(saved?: string, unavailable = false) {
  const globals = globalThis as any, previous = { document: globals.document, localStorage: globals.localStorage };
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  const values = new Map<string, string>(); if (saved !== undefined) values.set("luban.home-background", saved);
  globals.document = { getElementById: get, createElement: () => new Element() };
  globals.localStorage = {
    getItem(key: string) { if (unavailable) throw Error("unavailable"); return values.get(key) ?? null; },
    setItem(key: string, value: string) { if (unavailable) throw Error("unavailable"); values.set(key, value); },
  };
  const controller = initializeBackgrounds();
  return { ...controller, get, values, cards: get("background-grid").children,
    restore() { Object.assign(globals, previous); } };
}

test("only received images appear; unlocks require actual safe evolution counts and locked clicks cannot select", () => {
  const app = fixture();
  try {
    const [welcome, study, relax, inspiration, sunset] = ["welcome", "study", "relax", "inspiration", "sunset"].map(id => app.cards.find(card => card.dataset.backgroundId === id)!);
    expect(app.cards.length).toBe(44);
    expect(app.get("background-progress").textContent).toBe("进化次数待确认 · 已解锁 10 / 44 张背景");
    expect(welcome.disabled).toBe(false); expect(study.disabled).toBe(true);
    expect(relax.disabled).toBe(true);
    expect(inspiration.disabled).toBe(true);
    expect(sunset.disabled).toBe(true);
    expect(inspiration.children[0].children[1].textContent).toBe("未解锁\n5次进化");
    expect(study.children[0].children[1].textContent).toBe("未解锁\n1次进化");
    expect(study.classes.has("is-locked")).toBe(true);
    for (const count of [null, undefined, -1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, "1" as any, { tokens: 999999 } as any, 0]) {
      app.update(count); study.click(); relax.click();
      expect(study.disabled).toBe(true);
      expect(relax.disabled).toBe(true);
      expect(app.get("home-background").src).toBe("/assets/brand/home-free-memories.webp");
      expect(app.values.size).toBe(0);
    }
    app.update(1);
    expect(study.disabled).toBe(false); expect(study.children[0].children[1].hidden).toBe(true);
    expect(relax.disabled).toBe(false); expect(relax.children[0].children[1].hidden).toBe(true);
    expect(inspiration.disabled).toBe(true);
    inspiration.click(); expect(app.values.size).toBe(0);
    expect(sunset.disabled).toBe(true);
    expect(app.get("background-progress").textContent).toBe("已完成 1 次进化 · 已解锁 12 / 44 张背景");
    study.click();
    expect(app.get("home-background").src).toBe("/assets/brand/home-study.webp");
    expect(study.getAttribute("aria-pressed")).toBe("true");
    expect(app.values.get("luban.home-background")).toBe("study");
    relax.click();
    expect(app.get("home-background").src).toBe("/assets/brand/home-relax.webp");
    expect(relax.getAttribute("aria-pressed")).toBe("true");
    expect(study.getAttribute("aria-pressed")).toBe("false");
    expect(app.values.get("luban.home-background")).toBe("relax");
    app.update(4); expect(inspiration.disabled).toBe(true);
    app.update(5); expect(inspiration.disabled).toBe(false);
    expect(sunset.disabled).toBe(false);
    expect(app.get("background-progress").textContent).toBe("已完成 5 次进化 · 已解锁 14 / 44 张背景");
    inspiration.click();
    expect(app.get("home-background").src).toBe("/assets/brand/home-inspiration.webp");
    sunset.click();
    expect(app.get("home-background").src).toBe("/assets/brand/home-sunset.webp");
    expect(app.values.get("luban.home-background")).toBe("sunset");
    app.update(180); app.update(180);
    expect(app.get("background-grid").replacements).toBe(1);
    expect(app.get("background-grid").children.find(card => card.dataset.backgroundId === "study")).toBe(study);
    app.get("open-backgrounds").click(); expect(app.get("background-dialog").open).toBe(true);
    app.get("close-backgrounds").click(); expect(app.get("background-dialog").open).toBe(false);
  } finally { app.restore(); }
});

test("stored preference waits for current evidence, relocks on lower or missing evidence, and returns without erasing preference", () => {
  for (const [id, image, minimum] of [["study", "home-study.webp", 1], ["relax", "home-relax.webp", 1],
    ["inspiration", "home-inspiration.webp", 5], ["sunset", "home-sunset.webp", 5]] as const) {
    const app = fixture(id);
    try {
    expect(app.get("home-background").src).toBe("/assets/brand/home-free-memories.webp");
    app.update(5); expect(app.get("home-background").src).toBe(`/assets/brand/${image}`);
    app.update(0); expect(app.get("home-background").src).toBe("/assets/brand/home-free-memories.webp");
    expect(app.values.get("luban.home-background")).toBe(id);
    app.update(1); app.update(undefined);
    expect(app.cards.find(card => card.dataset.backgroundId === "study")!.disabled).toBe(true);
    expect(app.cards.find(card => card.dataset.backgroundId === "relax")!.disabled).toBe(true);
    expect(app.cards.find(card => card.dataset.backgroundId === "inspiration")!.disabled).toBe(true);
    expect(app.cards.find(card => card.dataset.backgroundId === "sunset")!.disabled).toBe(true);
    expect(app.get("background-progress").textContent).toContain("进化次数待确认");
    app.update(minimum); expect(app.get("home-background").src).toBe(`/assets/brand/${image}`);
    app.cards[0].click(); app.update(100);
    expect(app.get("home-background").src).toBe("/assets/brand/home-free-memories.webp");
    } finally { app.restore(); }
  }
});

test("storage failure does not block selection and untrusted saved URLs or markup never become images or names", () => {
  const unavailable = fixture(undefined, true);
  try {
    unavailable.update(1); unavailable.cards.find(card => card.dataset.backgroundId === "study")!.click();
    expect(unavailable.get("home-background").src).toBe("/assets/brand/home-study.webp");
  } finally { unavailable.restore(); }
  for (const saved of ["https://evil.test/image.png", "<img src=x onerror=alert(1)>", "missing"]) {
    const app = fixture(saved);
    try {
      app.update(100);
      expect(app.get("home-background").src).toBe("/assets/brand/home-free-memories.webp");
      expect(app.cards[0].children[1].textContent).toBe("时光记忆");
      expect(app.cards.every(card => /^\/assets\/brand\/home-[a-z-]+\.webp$/.test(card.children[0].children[0].src))).toBe(true);
    } finally { app.restore(); }
  }
});

// Milestones are product requirements, independent of token totals or stored selection.
test("all received backgrounds follow the requested evolution milestones", () => {
  const app = fixture();
  try {
    const expected = [0,0,0,0,0,0,0,0,0,0,1,1,5,5,10,10,10,20,20,30,30,50,50,70,70,100,100,100,140,140,180,180,180,180,180,180,180,180,180,180,180,180,180,200];
    expect(app.cards.length).toBe(expected.length);
    for (const [id, minimum] of [["milestone-ten", 10], ["milestone-hundred", 100], ["milestone-two-hundred", 200]] as const) {
      const card = app.cards.find(card => card.dataset.backgroundId === id)!;
      expect(card).toBeDefined();
      app.update(minimum - 1); card.click(); expect(card.disabled).toBe(true);
      expect(app.values.get("luban.home-background")).not.toBe(id);
      app.update(minimum); card.click(); expect(card.disabled).toBe(false);
      expect(app.values.get("luban.home-background")).toBe(id);
      app.update(null); expect(app.get("home-background").src).toBe("/assets/brand/home-free-memories.webp");
    }
    for (const [index, minimum] of expected.entries()) {
      if (minimum > 0) {
        app.update(minimum - 1); app.cards[index].click();
        expect(app.cards[index].disabled).toBe(true);
        expect(app.cards[index].children[0].children[1].textContent).toBe(`未解锁\n${minimum}次进化`);
      }
      app.update(minimum); expect(app.cards[index].disabled).toBe(false);
      app.cards[index].click();
      expect(app.get("home-background").src).toBe(app.cards[index].children[0].children[0].src);
    }
    expect(app.get("background-progress").textContent).toBe("已完成 200 次进化 · 已解锁 44 / 44 张背景");
  } finally { app.restore(); }
});


test("the free backgrounds remain free with zero or unknown evolution evidence and persist across restarts", () => {
  const ids = ["free-memories", "free-stickers", "free-summit", "free-nap", "free-fishing", "free-marbles", "free-cards", "free-warehouse", "free-faces"];
  for (const id of ids) {
    const app = fixture(id);
    try {
      const card = app.cards.find(card => card.dataset.backgroundId === id)!;
      expect(app.cards.filter(card => card.dataset.backgroundId.startsWith("free-")).map(card => card.dataset.backgroundId)).toEqual(ids);
      for (const count of [null, 0, undefined, -1, NaN]) {
        app.update(count);
        expect(card.disabled).toBe(false);
        expect(card.children[0].children[1].hidden).toBe(true);
        expect(app.get("home-background").src).toBe(`/assets/brand/home-${id}.webp`);
        app.get("open-backgrounds").click();
        card.click();
        expect(app.get("background-dialog").open).toBe(false);
        expect(card.getAttribute("aria-pressed")).toBe("true");
        expect(app.values.get("luban.home-background")).toBe(id);
      }
    } finally { app.restore(); }
  }
});
