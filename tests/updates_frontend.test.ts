import { expect, test } from "bun:test";
import { initializeUpdates } from "../frontend/updates";
import { componentDefinitions, nextVersions, publicVersions } from "../scripts/component-versions";

class Element {
  textContent = ""; className = ""; hidden = false; disabled = false; children: Element[] = [];
  append(...children: Element[]) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  addEventListener() {}
}

test("update page renders product component versions and operation learning with its shared training engine version", async () => {
  const previous = { document: globalThis.document, window: globalThis.window, fetch: globalThis.fetch };
  const elements = new Map<string, Element>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id)!; };
  const registry = nextVersions(null, Object.fromEntries(componentDefinitions.map(c => [c.id, "a".repeat(64)])), { version: "0.3.0", build: 1, notes: ["fixture"] }).registry;
  const manifest = publicVersions(registry, {}), requests: string[] = [];
  Object.assign(globalThis, {
    document: { getElementById: get, querySelectorAll: () => [], createElement: () => new Element() },
    window: { addEventListener() {}, webkit: { messageHandlers: { lubanUpdate: { postMessage() {} } } } },
    fetch: async (url: string) => { requests.push(url); return { ok: true, json: async () => manifest }; },
  });
  try {
    await initializeUpdates();
    const rows = get("component-versions").children.map(row => row.children.map(child => child.textContent));
    expect(rows).toContainEqual(["图片理解", "1.0.0"]);
    expect(rows).toContainEqual(["声音识别", "1.0.0"]);
    expect(rows).toContainEqual(["训练引擎", "1.0.0"]);
    expect(rows).toContainEqual(["深度学习", "1.0.0"]);
    expect(rows.filter(([name]) => name === "深度学习")).toHaveLength(1);
    expect(requests).toEqual(["/component-versions.json"]);
  } finally { Object.assign(globalThis, previous); }
});
