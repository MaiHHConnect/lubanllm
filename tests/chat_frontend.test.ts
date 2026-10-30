import { test, expect } from "bun:test";
import { attachmentPrompt, conversationContext, conversationHasMedia, deleteConversation, conversationTitle, displayMessageMeta, enterSends, fileKind, modelMessages, newConversation, type Attachment } from "../frontend/chat-storage";
const file = (kind: Attachment["kind"], content: string, name = "附件.txt"): Attachment => ({ id: "file", name, kind, content, size: content.length });

test("saved average speeds retain their original meaning after decode metric changes", () => {
  const old = { role: "assistant" as const, content: "旧回答", meta: "输出 61 tokens · 生成速度 10.0 词元/秒" };
  expect(displayMessageMeta(old)).toBe("输出 61 tokens · 生成速度（端到端） 10.0 token/秒");
  expect(old.meta).toContain("生成速度 10.0 词元/秒");
  expect(displayMessageMeta({ ...old, speedMetric: "decode", meta: "生成速度 20.0 token/秒" })).toBe("生成速度 20.0 token/秒");
  expect(displayMessageMeta({ ...old, meta: "Generation speed 10.0 token/s" })).toBe("Generation speed (end-to-end) 10.0 token/s");
});

test("Enter sends, Shift+Enter and IME selection stay in the editor", () => {
  const event = { key: "Enter", shiftKey: false, isComposing: false, keyCode: 13 };
  expect(enterSends(event)).toBe(true);
  expect(enterSends({ ...event, shiftKey: true })).toBe(false);
  expect(enterSends({ ...event, isComposing: true })).toBe(false);
  expect(enterSends({ ...event, keyCode: 229 })).toBe(false);
});
test("file names and MIME recognize media so it cannot become teacher text", () => {
  expect(fileKind({ name: "voice.M4A", type: "" })).toBe("audio");
  expect(fileKind({ name: "file", type: "audio/wav" })).toBe("audio");
  expect(fileKind({ name: "paste.png", type: "" })).toBe("image");
  expect(fileKind({ name: "notes.md", type: "text/plain" })).toBe("text");
  const prompt = attachmentPrompt("总结", [file("text", "真实文件正文", "目录/笔记.md")]);
  expect(prompt).toContain("真实文件正文"); expect(prompt).toContain("目录/笔记.md");
  for (const kind of ["audio", "image"] as const) expect(() => attachmentPrompt("总结", [file(kind, "旧附件")])).toThrow("仅支持文字");
});
test("old media histories stay intact and cannot enter follow-up or compaction", () => {
  for (const kind of ["image", "audio"] as const) {
    const conversation = newConversation(), attachment = file(kind, "历史原始内容", "旧附件");
    conversation.messages = [{ role: "user", content: "旧问题", context: "历史转写或说明", attachments: [attachment] }, { role: "assistant", content: "旧回答" }, { role: "user", content: "继续" }];
    const before = structuredClone(conversation);
    expect(conversationHasMedia(conversation)).toBe(true);
    expect(() => modelMessages(conversation.messages)).toThrow("新建纯文字会话");
    expect(() => conversationContext(conversation)).toThrow("新建纯文字会话");
    conversation.compaction = { through: 2, summary: "旧媒体摘要" };
    expect(() => conversationContext(conversation)).toThrow("新建纯文字会话");
    expect(conversation.messages).toEqual(before.messages);
    const draft = newConversation(); draft.attachments = [attachment];
    expect(() => conversationContext(draft)).toThrow("新建纯文字会话");
    expect(draft.attachments).toEqual([attachment]);
  }
  expect(modelMessages([{ role: "assistant", content: "", meta: "失败" }])).toEqual([]);
});
test("old media titles preserve names while text document bodies remain usable", () => {
  const conversation = newConversation();
  conversation.messages = [{ role: "user", content: "", attachments: [file("image", "data:image/png;base64,aGVsbG8=", "参考.png")] }];
  expect(conversationTitle(conversation)).toBe("参考.png");
  expect(() => modelMessages(conversation.messages)).toThrow("历史图片或音频");
  expect(attachmentPrompt("", [file("text", "123")])).toContain("123");
  const context = attachmentPrompt("读正文", [file("text", "文档实际正文", "资料.docx")]);
  expect(modelMessages([{ role: "user", content: "读正文", context, attachments: [file("text", "文档实际正文")] }])[0].content).toContain("文档实际正文");
});

test("compaction checkpoints preserve full history and map repeated summaries to original indices", () => {
  const conversation = newConversation();
  conversation.messages = [
    { role: "user", content: "原始文件正文" }, { role: "assistant", content: "旧回答" },
    { role: "user", content: "继续" }, { role: "assistant", content: "", error: true },
    { role: "user", content: "后续问题" }, { role: "assistant", content: "后续回答" },
  ];
  conversation.compaction = { summary: "交接摘要", through: 2 };
  const context = conversationContext(conversation);
  expect(context.indices).toEqual([1, 2, 4, 5]);
  expect(context.messages[0].content).toContain("交接摘要");
  expect(context.messages[1].content).toBe("继续");
  expect(conversation.messages[0].content).toBe("原始文件正文");
  conversation.compaction = { summary: "第二轮摘要", through: context.indices[2] + 1 };
  expect(conversationContext(conversation).indices).toEqual([4, 5]);
  expect(conversationContext(conversation).messages[1].content).toBe("后续回答");
});

test("system constraints survive repeated compaction outside the generated summary", () => {
 const conversation = newConversation();
 conversation.messages = [{ role: "system", content: "必须保留的原系统约束" }, { role: "user", content: "旧问题" }, { role: "assistant", content: "旧回答" }, { role: "user", content: "新问题" }];
 conversation.compaction = { summary: "可损失的摘要", through: 3 };
 const context = conversationContext(conversation);
 expect(context.messages[0]).toEqual({ role: "system", content: "必须保留的原系统约束" });
 expect(context.messages[1].role).toBe("user");
 expect(context.indices).toEqual([0, 2, 3]);
});


test("local conversation deletion waits for transaction commit and reports aborts", async () => {
  const deleted: string[] = [];
  const transaction: any = { objectStore(name: string) { expect(name).toBe("conversations"); return { delete(id: string) { deleted.push(id); } }; } };
  const db = { transaction(name: string, mode: string) { expect(name).toBe("conversations"); expect(mode).toBe("readwrite"); return transaction; } } as unknown as IDBDatabase;
  let committed = false;
  const pending = deleteConversation(db, "local-draft").then(() => { committed = true; });
  expect(deleted).toEqual(["local-draft"]); await Promise.resolve(); expect(committed).toBe(false);
  transaction.oncomplete(); await pending; expect(committed).toBe(true);
  const failed = deleteConversation(db, "saved-chat"); transaction.onabort();
  await expect(failed).rejects.toThrow("会话未能删除");
});

test("context menu deletes local and server chats, preserves failed deletions, and blocks overlapping actions", async () => {
  const { initializeChat } = await import("../frontend/chat");
  const globals = globalThis as any;
  const previous = { document: globals.document, window: globals.window, Node: globals.Node, indexedDB: globals.indexedDB };
  let focused: ElementFixture | undefined;
  class ElementFixture extends EventTarget {
    id = ""; className = ""; textContent = ""; title = ""; value = ""; hidden = false; disabled = false;
    style: Record<string, string> = {}; attributes = new Map<string, string>(); children: ElementFixture[] = [];
    parentElement?: ElementFixture; offsetWidth = 150; offsetHeight = 40; scrollHeight = 0; scrollTop = 0;
    constructor(readonly tagName = "div") { super(); }
    setAttribute(key: string, value: string) { this.attributes.set(key, value); }
    append(...items: ElementFixture[]) { for (const item of items) { item.parentElement = this; this.children.push(item); } }
    replaceChildren(...items: ElementFixture[]) { this.children = []; this.append(...items); }
    contains(item: ElementFixture): boolean { return item === this || this.children.some(child => child.contains(item)); }
    querySelectorAll(selector: string): ElementFixture[] { return this.children.flatMap(child => [...(child.tagName === selector ? [child] : []), ...child.querySelectorAll(selector)]); }
    focus() { focused = this; }
    getBoundingClientRect() { return { left: 12, bottom: 80 }; }
  }
  const nodes = new Map<string, ElementFixture>();
  const get = (id: string) => {
    if (!nodes.has(id)) { const element = new ElementFixture(); element.id = id; nodes.set(id, element); }
    return nodes.get(id)!;
  };
  const body = new ElementFixture("body");
  const server = { ...newConversation(), serverId: "server-one", title: "Saved", draft: "Saved", updated: 20 };
  const local = { ...newConversation(), title: "Draft", draft: "Draft", updated: 10 };
  const records = new Map([[server.id, server], [local.id, local]]);
  const db = { transaction() {
    const transaction: any = { objectStore() { return {
      getAll() { const request: any = { result: [...records.values()] }; queueMicrotask(() => request.onsuccess()); return request; },
      put(value: typeof server) { records.set(value.id, structuredClone(value)); queueMicrotask(() => transaction.oncomplete()); },
      delete(id: string) { records.delete(id); queueMicrotask(() => transaction.oncomplete()); },
    }; } }; return transaction;
  } };
  const errors: unknown[] = []; const deletions: string[] = [];
  let settleDelete: (value?: any) => void = () => {}, rejectDelete: (error: Error) => void = () => {};
  globals.Node = ElementFixture;
  globals.document = Object.assign(new EventTarget(), { body, getElementById: get, createElement: (tag: string) => new ElementFixture(tag) });
  globals.window = Object.assign(new EventTarget(), { innerWidth: 640, innerHeight: 480 });
  globals.indexedDB = { open() { const request: any = { result: db }; queueMicrotask(() => request.onsuccess()); return request; } };
  const fire = (target: ElementFixture, type: string, fields = {}) => target.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), fields));
  try {
    const chat = initializeChat({ api: (async (path: string, _data?: unknown, method?: string) => {
      if (method === "DELETE") { deletions.push(path); return new Promise((resolve, reject) => { settleDelete = resolve; rejectDelete = reject; }); }
      return { conversations: [] };
    }) as any, failure: error => errors.push(error), refresh: async () => {}, ready: () => true });
    await chat.load();
    const menu = body.children[0], remove = menu.children[0];
    let item = get("chat-conversations").children[0];
    fire(item, "keydown", { key: "F10", shiftKey: true });
    expect(menu.hidden).toBe(false); expect(focused).toBe(remove);
    fire(menu, "keydown", { key: "Escape" }); expect(menu.hidden).toBe(true); expect(focused).toBe(item);
    fire(item, "contextmenu", { clientX: 630, clientY: 470 });
    expect(menu.style.left).toBe("482px"); expect(menu.style.top).toBe("432px");
    fire(remove, "click"); await Bun.sleep(0);
    expect(deletions).toEqual(["/api/conversations/server-one"]);
    expect(chat.isBusy()).toBe(true); expect(get("chat-conversations").children).toHaveLength(2);
    expect(get("chat-input").disabled).toBe(true);
    fire(get("chat-conversations").children[0], "contextmenu", { clientX: 12, clientY: 30 }); expect(menu.hidden).toBe(true);
    rejectDelete(Error("service unavailable")); await Bun.sleep(0);
    expect(errors).toHaveLength(1); expect(records.has(server.id)).toBe(true); expect(chat.isBusy()).toBe(false);
    item = get("chat-conversations").children[0]; fire(item, "contextmenu", { clientX: 12, clientY: 30 }); fire(remove, "click");
    await Bun.sleep(0); settleDelete({ ok: true }); await Bun.sleep(0);
    expect(records.has(server.id)).toBe(false); expect(get("chat-conversations").children).toHaveLength(1);
    expect(get("chat-title").textContent).toBe("Draft");
    fire(get("chat-conversations").children[0], "contextmenu", { clientX: 12, clientY: 30 }); fire(remove, "click"); await Bun.sleep(0);
    expect(records.size).toBe(0); expect(deletions).toHaveLength(2); expect(get("chat-title").textContent).toBe("新会话");
    globals.document.dispatchEvent(new Event("visibilitychange")); await Bun.sleep(0);
    expect(records.size).toBe(0);
  } finally { Object.assign(globals, previous); }
});
