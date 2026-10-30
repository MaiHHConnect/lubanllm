import { expect, test } from "bun:test";
import { copyEvolutionImage, createEvolutionShareObserver } from "../frontend/evolution-share";

const active = (id = "live") => ({ job: { id, kind: "evolution", state: "running" } });
const complete = (id = "live", promoted = true, publication_state = "published") => ({ job: {
  id, kind: "evolution", state: "complete", result: { id, promoted, publication_state },
} });
test("only newly observed, independently accepted and published evolution celebrates once", () => {
  let celebrations = 0, persisted: string[] = [];
  const observe = createEvolutionShareObserver(() => celebrations++, [], ids => { persisted = ids; });
  observe(complete("historic"));
  expect(celebrations).toBe(0);
  observe(active()); observe(complete()); observe(complete()); observe(active()); observe(complete());
  expect(celebrations).toBe(1);
  expect(persisted).toContain("live");
  const reopened = createEvolutionShareObserver(() => celebrations++, persisted);
  reopened(active()); reopened(complete());
  expect(celebrations).toBe(1);
  observe(active("next")); observe(complete("next"));
  expect(celebrations).toBe(2);
});
test("failed, unadopted, uncommitted, unrelated or mismatched jobs do not celebrate", () => {
  let celebrations = 0;
  for (const end of [complete("a", false), complete("a", true, "validated_candidate"), complete("other"),
    { job: { ...complete("a").job, state: "error" } },
    { job: { ...complete("a").job, kind: "learning" } },
    { job: { ...complete("a").job, result: { ...complete("a").job.result, id: "mismatch" } } },
    { job: { kind: "evolution", state: "complete", result: { promoted: true, publication_state: "published" } } },
  ]) {
    const observe = createEvolutionShareObserver(() => celebrations++);
    observe(active("a")); observe(end);
  }
  expect(celebrations).toBe(0);
  const interrupted = createEvolutionShareObserver(() => celebrations++);
  interrupted(active()); interrupted({}); interrupted(complete());
  expect(celebrations).toBe(0);
});
test("native image copy waits for its matching successful acknowledgement", async () => {
  const globals = globalThis as any, old = globals.window;
  const target = new EventTarget();
  let sent: any;
  try {
    globals.window = Object.assign(target, { webkit: { messageHandlers: { lubanShare: { postMessage(value: unknown) { sent = value; } } } } });
    let finished = false;
    const copy = copyEvolutionImage(Promise.resolve(new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }))).then(() => { finished = true; });
    await Bun.sleep(1);
    expect(sent.action).toBe("copy-image"); expect(sent.png).toBe("iVBORw=="); expect(finished).toBe(false);
    target.dispatchEvent(new CustomEvent("luban-share-copied", { detail: { id: "other", success: true } }));
    await Bun.sleep(1); expect(finished).toBe(false);
    target.dispatchEvent(new CustomEvent("luban-share-copied", { detail: { id: sent.id, success: true } }));
    await copy; expect(finished).toBe(true);
    const rejected = copyEvolutionImage(Promise.resolve(new Blob(["png"], { type: "image/png" })));
    await Bun.sleep(1);
    target.dispatchEvent(new CustomEvent("luban-share-copied", { detail: { id: sent.id, success: false } }));
    await expect(rejected).rejects.toThrow("图片未能复制");
  } finally { globals.window = old; }
});
test("browser image copy writes a PNG ClipboardItem and surfaces permission errors", async () => {
  const globals = globalThis as any;
  const old = { window: globals.window, navigator: globals.navigator, ClipboardItem: globals.ClipboardItem };
  try {
    globals.window = {};
    globals.ClipboardItem = class { constructor(readonly value: any) {} };
    const png = new Blob(["png"], { type: "image/png" }); let contents: Blob | undefined;
    globals.navigator = { clipboard: { async write(items: any[]) { contents = await items[0].value["image/png"]; } } };
    await copyEvolutionImage(Promise.resolve(png)); expect(contents).toBe(png);
    globals.navigator.clipboard.write = async () => { throw Error("permission denied"); };
    await expect(copyEvolutionImage(Promise.resolve(png))).rejects.toThrow("permission denied");
  } finally { Object.assign(globals, old); }
});
