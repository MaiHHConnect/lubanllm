import { t, locale } from "./i18n";
import { attachmentPrompt, conversationDatabase, conversationContext, deleteConversation, conversationHasMedia, conversationTitle, displayMessageMeta, enterSends, fileKind, newConversation, readConversations, saveConversation, type Attachment, type ChatMessage, type Conversation } from "./chat-storage";

type ChatEvent = { type: string; summary?: string; prefix_count?: number; input_tokens?: number; after_tokens?: number; limit?: number; compact_only?: boolean; request_id?: string; conversation_id?: string; route?: string; source?: string; action?: string; confidence?: number; content?: string; reasoning_content?: string; message?: string; usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; elapsed_seconds?: number; tokens_per_second?: number | null; decode_tokens_per_second?: number | null; estimated?: boolean; finish_reason?: string; requested_thinking?: string; effective_thinking?: string; reference_used?: boolean } };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const button = (id: string) => $<HTMLButtonElement>(id);
const field = () => $<HTMLTextAreaElement>("chat-input");
const choose = (id: string) => $<HTMLSelectElement>(id);
const count = (value?: number) => typeof value === "number" ? value.toLocaleString(locale()) : "—";

export function initializeChat(options: { api: <T = Record<string, unknown>>(path: string, data?: unknown, method?: string) => Promise<T>; failure: (error: unknown) => void; refresh: () => Promise<unknown>; ready: () => boolean; teacherFirst?: () => boolean }) {
  let current = newConversation();
  let conversations: Conversation[] = [];
  let db: IDBDatabase | undefined;
  let busy = false, adding = false, requestId = "", stopping = false, loaded = false;
  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  let queue = Promise.resolve();
  let storageError = false, deleting = false;
  let menuConversation: Conversation | undefined, menuOrigin: HTMLButtonElement | undefined;
  const conversationMenu = document.createElement("div");
  conversationMenu.id = "chat-conversation-menu"; conversationMenu.className = "chat-conversation-menu";
  conversationMenu.hidden = true; conversationMenu.setAttribute("role", "menu");
  const deleteButton = document.createElement("button"); deleteButton.type = "button";
  deleteButton.textContent = t("删除会话"); deleteButton.setAttribute("role", "menuitem");
  conversationMenu.append(deleteButton); document.body.append(conversationMenu);
  function closeConversationMenu(restoreFocus = false) {
    conversationMenu.hidden = true; menuOrigin?.setAttribute("aria-expanded", "false");
    if (restoreFocus) menuOrigin?.focus();
    menuConversation = undefined; menuOrigin = undefined;
  }
  function openConversationMenu(conversation: Conversation, origin: HTMLButtonElement, x: number, y: number) {
    if (!loaded || busy || adding) return;
    closeConversationMenu(); menuConversation = conversation; menuOrigin = origin;
    origin.setAttribute("aria-expanded", "true"); conversationMenu.hidden = false;
    conversationMenu.style.left = Math.max(8, Math.min(x, window.innerWidth - conversationMenu.offsetWidth - 8)) + "px";
    conversationMenu.style.top = Math.max(8, Math.min(y, window.innerHeight - conversationMenu.offsetHeight - 8)) + "px";
    deleteButton.focus();
  }
  async function removeConversation(conversation: Conversation) {
    if (!loaded || busy || adding || !db) return;
    adding = true; closeConversationMenu(); renderControls();
    try {
      await persist(); deleting = true; await queue;
      if (conversation.serverId) await options.api("/api/conversations/" + encodeURIComponent(conversation.serverId), undefined, "DELETE");
      await deleteConversation(db, conversation.id);
      conversations = conversations.filter(item => item.id !== conversation.id);
      if (current.id === conversation.id) current = [...conversations].sort((a, b) => b.updated - a.updated)[0] || newConversation();
      renderConversation(); $("chat-save-state").textContent = t("会话已删除");
    } finally { deleting = false; adding = false; renderControls(); field().focus(); }
  }
  deleteButton.addEventListener("click", () => { if (menuConversation) void removeConversation(menuConversation).catch(options.failure); });
  conversationMenu.addEventListener("keydown", event => {
    if (event.key === "Escape") { event.preventDefault(); closeConversationMenu(true); }
    else if (event.key === "Tab") closeConversationMenu(true);
    else if (["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) { event.preventDefault(); deleteButton.focus(); }
  });
  document.addEventListener("pointerdown", event => {
    if (event.target instanceof Node && !conversationMenu.contains(event.target)) closeConversationMenu();
  });
  document.addEventListener("scroll", () => closeConversationMenu(), true);
  window.addEventListener("resize", () => closeConversationMenu());
  const notify = (message: string) => { $("attachment-state").textContent = message; $("attachment-state").hidden = !message; };

  function renderControls() {
    const teacherFirst = !!options.teacherFirst?.();
    choose("chat-route").value = teacherFirst ? "teacher" : current.route;
    button("chat-send").disabled = !loaded || busy || adding || !options.ready() || (!field().value.trim() && !current.attachments.length) || conversationHasMedia(current);
    button("chat-compact").disabled = teacherFirst || !loaded || busy || adding || !options.ready() || current.messages.length < 6 || conversationHasMedia(current);
    button("chat-stop").hidden = !busy;
    button("chat-stop").disabled = !busy || stopping;
    field().disabled = busy || adding;
    for (const id of ["chat-new", "chat-add", "chat-pick-file", "chat-pick-folder"]) button(id).disabled = !loaded || busy || adding;
    choose("chat-route").disabled = teacherFirst || busy; choose("chat-thinking").disabled = busy;
    $("chat-teacher-hint").hidden = choose("chat-route").value !== "teacher";
    $("chat-form").setAttribute("aria-busy", String(busy || adding));
    $("chat-messages").setAttribute("aria-busy", String(busy));
    for (const item of $("chat-conversations").querySelectorAll<HTMLButtonElement>("button")) item.disabled = busy || adding;
    for (const item of $("chat-attachments").querySelectorAll<HTMLButtonElement>("button")) item.disabled = busy || adding;
    if (!busy) $("chat-state").textContent = adding ? t("正在读取附件") : options.ready() ? t("可以发送") : t("等待模型就绪");
  }
  function renderList() {
    closeConversationMenu();
    $("chat-conversations").replaceChildren();
    for (const conversation of [...conversations].sort((a, b) => b.updated - a.updated)) {
      const item = document.createElement("button"); item.type = "button"; item.className = "chat-conversation";
      item.textContent = conversation.title; item.title = conversation.title;
      item.setAttribute("aria-current", conversation.id === current.id ? "true" : "false");
      item.setAttribute("aria-haspopup", "menu"); item.setAttribute("aria-expanded", "false");
      item.addEventListener("contextmenu", event => { event.preventDefault(); openConversationMenu(conversation, item, event.clientX, event.clientY); });
      item.addEventListener("keydown", event => {
        if (event.key === "ContextMenu" || event.key === "F10" && event.shiftKey) {
          event.preventDefault(); const bounds = item.getBoundingClientRect(); openConversationMenu(conversation, item, bounds.left, bounds.bottom);
        }
      });
      item.addEventListener("click", () => { void switchConversation(conversation).catch(options.failure); });
      $("chat-conversations").append(item);
    }
    renderControls();
  }
  function rememberDraft() {
    current.draft = field().value; current.route = choose("chat-route").value; current.thinking = choose("chat-thinking").value;
  }
  async function persist() {
    if (deleting) return;
    clearTimeout(saveTimer); saveTimer = undefined; rememberDraft();
    if (!db) throw new Error(t("会话存储未就绪，暂时不能发送。"));
    current.title = conversationTitle(current); current.updated = Date.now();
    if (!current.messages.length && !current.draft.trim() && !current.attachments.length) return;
    const snapshot = structuredClone(current);
    const existing = conversations.findIndex(value => value.id === current.id);
    if (existing < 0) conversations.push(current); else conversations[existing] = current;
    renderList(); $("chat-title").textContent = current.title;
    const saving = queue.then(() => saveConversation(db!, snapshot));
    queue = saving.catch(() => {});
    try { await saving; storageError = false; $("chat-save-state").textContent = t("已保存到本机"); }
    catch (error) { storageError = true; $("chat-save-state").textContent = t("保存失败，内容仍在当前窗口"); throw error; }
  }
  function scheduleSave() {
    rememberDraft(); clearTimeout(saveTimer);
    $("chat-save-state").textContent = t("正在保存…");
    saveTimer = setTimeout(() => { saveTimer = undefined; void persist().catch(options.failure); }, 350);
  }
  function attachmentElement(file: Attachment, removable: boolean) {
    const chip = document.createElement("div"); chip.className = "chat-attachment";
    if (file.kind === "image") {
      const image = document.createElement("img"); image.src = file.content; image.alt = file.name; chip.append(image);
    }
    const name = document.createElement("span"); name.textContent = (file.kind === "audio" ? t("历史音频：") : "") + file.name; name.title = file.name; chip.append(name);
    if (removable) {
      const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×"; remove.setAttribute("aria-label", t`移除 ${file.name}`);
      remove.addEventListener("click", () => { current.attachments = current.attachments.filter(item => item.id !== file.id); renderAttachments(); scheduleSave(); }); chip.append(remove);
    }
    return chip;
  }
  function renderAttachments() {
    $("chat-attachments").replaceChildren(...current.attachments.map(file => attachmentElement(file, true)));
    if (conversationHasMedia(current)) notify(t("此会话含历史图片或音频，已保留原记录；请新建纯文字会话。"));
    else notify("");
    renderControls();
  }
  function messageElement(message: ChatMessage) {
    const row = document.createElement("article"); row.className = "chat-message"; row.dataset.role = message.role;
    const title = document.createElement("strong"); title.textContent = message.role === "user" ? t("你") : message.role === "system" ? t("系统") : t("鲁班零号");
    const content = document.createElement("p"); content.className = "chat-content"; content.textContent = message.content;
    row.append(title, content);
    if (message.attachments?.length) {
      const files = document.createElement("div"); files.className = "chat-attachments";
      files.append(...message.attachments.map(file => attachmentElement(file, false))); row.append(files);
    }
    const reasoning = document.createElement("details"); reasoning.hidden = !message.reasoning;
    const summary = document.createElement("summary"); summary.textContent = t("思考过程");
    const reasoningText = document.createElement("p"); reasoningText.className = "chat-reasoning"; reasoningText.textContent = message.reasoning || "";
    reasoning.append(summary, reasoningText); row.append(reasoning);
    const meta = document.createElement("p"); meta.className = "chat-meta"; meta.textContent = displayMessageMeta(message); row.append(meta);
    if (message.error) { row.dataset.error = "true"; meta.setAttribute("role", "alert"); }
    return { row, content, meta, reasoning, reasoningText };
  }
  function scrollChat() { const list = $("chat-messages"); list.scrollTop = list.scrollHeight; }
  function renderContext() {
    const usage = current.contextUsage;
    $("chat-context").textContent = usage ? t`上下文 ${count(usage.input_tokens)} / ${count(usage.limit)} 词元 · 75% 自动压缩${current.compaction ? t(" · 已压缩，原记录保留") : ""}` : t("上下文达到实际容量的 75% 时自动压缩");
  }
  function renderConversation() {
    renderContext();
    $("chat-title").textContent = current.title;
    $("chat-messages").replaceChildren(...current.messages.map(message => messageElement(message).row));
    if (!current.messages.length) {
      const empty = document.createElement("div"); empty.className = "chat-empty";
      const heading = document.createElement("h3"); heading.textContent = t("和你的模型聊一聊");
      const hint = document.createElement("p"); hint.textContent = t("输入消息，或用 ＋ 添加附件。"); empty.append(heading, hint); $("chat-messages").append(empty);
    }
    field().value = current.draft; choose("chat-route").value = current.route; choose("chat-thinking").value = current.thinking;
    renderAttachments(); renderList(); scrollChat();
  }
  async function switchConversation(conversation?: Conversation) {
    if (busy || adding) return;
    adding = true; renderControls();
    try { await persist(); current = conversation || newConversation(); renderConversation(); }
    finally { adding = false; renderControls(); field().focus(); }
  }
  function setMenu(open: boolean) {
    $("chat-attachment-menu").hidden = !open; button("chat-add").setAttribute("aria-expanded", String(open));
  }
  async function readFile(file: File): Promise<Attachment> {
    if (!file.size || file.size > 8 * 1024 * 1024) throw new Error(t("文件需要为非空且不超过 8 MiB。"));
    const kind = fileKind(file), name = file.webkitRelativePath || file.name;
    if (kind !== "text") throw new Error(t("当前对话仅支持文字，不接收图片或音频；请选择文本文件或文档。"));
    let content: string;
    if (/\.(pdf|docx)$/i.test(name)) {
      const response = await fetch("/api/chat/attachments/extract?name=" + encodeURIComponent(file.name), { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/octet-stream" }, body: file });
      const result = await response.json();
      if (!response.ok) throw new Error(t(String(result.error?.message || result.detail || t("当前服务无法读取此文档。"))));
      if (typeof result.text !== "string" || !result.text.trim()) throw new Error(t("文档未提取到文本。")); content = result.text;
    } else {
      const raw = await file.arrayBuffer();
      const bytes = new Uint8Array(raw);
      try { content = new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
      catch { throw new Error(t("当前不支持这种二进制文件，请使用文本或文档。")); }
      if (/[\x00-\x08\x0e-\x1f]/.test(content)) throw new Error(t("文件包含二进制内容，无法作为文本读取。"));
      if (!content.trim()) throw new Error(t("文件没有可读取的文本。"));
    }
    if (content.length > 262144) throw new Error(t("文件文本超过 26 万字符，请拆分后添加。"));
    return { id: crypto.randomUUID(), name, size: file.size, kind, content };
  }
  async function addFiles(files: File[]) {
    if (!files.length || busy || adding || !loaded) return;
    if (current.attachments.length + files.length > 32 || current.attachments.reduce((sum, file) => sum + file.size, 0) + files.reduce((sum, file) => sum + file.size, 0) > 16 * 1024 * 1024) {
      notify(t("每次最多 32 个附件，共 16 MiB；请选取较小的文件夹或文件。")); return;
    }
    adding = true; setMenu(false); renderControls();
    const errors: string[] = [];
    try {
      for (const file of files) {
        try { current.attachments.push(await readFile(file)); }
        catch (error) { errors.push(`${file.webkitRelativePath || file.name}：${error instanceof Error ? t(error.message) : t("读取失败")}`); }
      }
      renderAttachments();
      await persist();
      if (errors.length) notify(errors.join("\n"));
    } catch (error) { options.failure(error); }
    finally { adding = false; renderControls(); }
  }
  async function stop() {
    if (!busy || stopping) return;
    stopping = true; $("chat-state").textContent = t("正在停止"); renderControls();
    try { if (requestId) await options.api("/api/stop", { request_id: requestId }); }
    catch (error) { stopping = false; renderControls(); options.failure(error); }
  }
  async function send(compactOnly = false) {
    if (button(compactOnly ? "chat-compact" : "chat-send").disabled || busy || adding) return;
    const prompt = field().value.trim();
    const user: ChatMessage = { role: "user", content: prompt, context: attachmentPrompt(prompt, current.attachments), attachments: structuredClone(current.attachments) };
    const prepared = conversationContext(current, compactOnly ? current.messages : [...current.messages, user]);
    const messages = prepared.messages;
    if (messages.length > 100 || JSON.stringify(messages).length > 7 * 1024 * 1024) { notify(t("对话超过当前接口的长度限制，请新建会话或缩小附件。")); return; }
    busy = true; stopping = false; requestId = ""; renderControls();
    const answer: ChatMessage = { role: "assistant", content: "", meta: t("正在请求…") };
    if (!compactOnly) { current.messages.push(user, answer); field().value = ""; current.draft = ""; current.attachments = []; }
    renderConversation(); const target = compactOnly ? messageElement(answer).row : $("chat-messages").lastElementChild as HTMLElement;
    const content = target.querySelector<HTMLElement>(".chat-content")!, meta = target.querySelector<HTMLElement>(".chat-meta")!;
    const reasoning = target.querySelector<HTMLDetailsElement>("details")!, reasoningText = target.querySelector<HTMLElement>(".chat-reasoning")!;
    let done = false, decision = "";
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    try {
      // Save the user message before calling the model, including failures and cancellation.
      await persist(); $("chat-state").textContent = t("等待回答");
      const response = await fetch("/api/chat", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages, route: current.route, thinking: current.thinking, conversation_id: current.serverId, auto_compact: true, compact_only: compactOnly }) });
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(t(String(body.error?.message || body.error || body.detail || t`请求失败（${response.status}）`))); }
      if (!response.body) throw new Error(t("当前窗口不支持流式回答"));
      reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
      const receive = async (block: string) => {
        const data = block.split(/\r?\n/).filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
        if (!data) return;
        const event = JSON.parse(data) as ChatEvent;
        if (event.type === "error") throw new Error(event.message ? t(event.message) : t("回答失败"));
        if (event.type === "compaction" && event.summary && event.prefix_count && event.prefix_count <= prepared.indices.length) {
          current.compaction = { summary: event.summary, through: prepared.indices[event.prefix_count - 1] + 1 };
          current.serverId = undefined;
          current.contextUsage = { input_tokens: event.after_tokens || 0, limit: event.limit || 0 };
          renderContext(); await persist(); notify(t("上下文已压缩，完整原会话仍保存在本机。"));
        } else if (event.type === "context" && typeof event.input_tokens === "number" && event.limit) {
          current.contextUsage = { input_tokens: event.input_tokens, limit: event.limit }; renderContext();
        } else if (event.type === "meta") {
          requestId = event.request_id || "";
          if (event.source === "context_check") {
            $("chat-state").textContent = t("正在检查或压缩上下文");
            if (stopping && requestId) await options.api("/api/stop", { request_id: requestId });
            return;
          }
          const sources: Record<string, string> = { local: t("本地模型"), teacher: t("API 教师") };
          const actions: Record<string, string> = { local_core: t("本地核心"), neural_expert: t("神经专家"), teacher: t("API 教师") };
          decision = t`来源：${sources[event.route || ""] || event.route || t("未知")}${event.action ? ` · ${actions[event.action] || event.action}` : ""}${typeof event.confidence === "number" ? t` · 置信度 ${(event.confidence * 100).toFixed(1)}%` : ""}`;
          answer.meta = decision; $("chat-state").textContent = stopping ? t("正在停止") : t("正在回答");
          if (stopping && requestId) await options.api("/api/stop", { request_id: requestId });
        } else if (event.type === "delta") {
          if (typeof event.content === "string") answer.content += event.content;
          if (typeof event.reasoning_content === "string") answer.reasoning = (answer.reasoning || "") + event.reasoning_content;
          scheduleSave();
        } else if (event.type === "done") {
          done = true; if (event.compact_only) { notify(current.compaction ? t("上下文已压缩，原记录保留。") : t("暂无可压缩的较早对话。最近消息会完整保留。")); return; }
          if (event.conversation_id) current.serverId = event.conversation_id;
          const usage = event.usage;
          if (current.contextUsage && usage?.total_tokens) { current.contextUsage.input_tokens = usage.total_tokens; renderContext(); }
          const ending = stopping ? t("已停止，回答可能不完整") : usage?.finish_reason === "length" ? t("达到本次输出预算，回答可能不完整") : usage?.finish_reason && usage.finish_reason !== "stop" ? t`结束：${usage.finish_reason}` : t("回答完成");
          const thinking = usage?.requested_thinking && usage.effective_thinking && usage.requested_thinking !== usage.effective_thinking ? t(" · 本次未启用思考") : "";
          const rate = usage?.decode_tokens_per_second;
          const speed = typeof rate === "number" && Number.isFinite(rate) && rate > 0 ? t`${rate.toFixed(1)} token/秒${usage?.estimated ? t("（估算）") : ""}` : "—";
          answer.meta = t`${decision}${decision ? " · " : ""}${ending}${thinking}${usage?.reference_used ? t(" · 本次使用知识参考") : ""}${usage ? t` · 输入 ${count(usage.prompt_tokens)} / 输出 ${count(usage.completion_tokens)} tokens${usage.estimated ? t("（估算）") : ""}` : ""} · 生成速度 ${speed}`;
          answer.speedMetric = "decode";
        }
        content.textContent = answer.content; meta.textContent = answer.meta || ""; reasoningText.textContent = answer.reasoning || ""; reasoning.hidden = !answer.reasoning;
        scrollChat();
      };
      while (true) {
        const chunk = await reader.read(); buffer += decoder.decode(chunk.value, { stream: !chunk.done });
        let separator: RegExpExecArray | null;
        while ((separator = /\r?\n\r?\n/.exec(buffer))) { const block = buffer.slice(0, separator.index); buffer = buffer.slice(separator.index + separator[0].length); await receive(block); }
        if (chunk.done) { if (buffer.trim()) await receive(buffer); break; }
      }
      if (!done) throw new Error(t("回答连接提前结束，可以继续发送或重试。"));
      if (!compactOnly && !answer.content && !answer.reasoning) answer.meta += t(" · 本次返回空回答");
    } catch (error) {
      if (compactOnly) notify(error instanceof Error ? t(error.message) : t("压缩失败，原会话保留。"));
      answer.error = true; answer.meta = `${decision}${decision ? " · " : ""}${error instanceof Error ? t(error.message) : t("回答失败")}`;
      target.dataset.error = "true"; meta.setAttribute("role", "alert");
    } finally {
      if (reader) { await reader.cancel().catch(() => {}); reader.releaseLock(); }
      meta.textContent = answer.meta || "";
      await persist().catch(options.failure);
      busy = false; requestId = ""; stopping = false; renderControls(); scrollChat();
      if (!$("view-chat").hidden) field().focus();
      void options.refresh().catch(options.failure);
    }
  }
  button("chat-compact").addEventListener("click", () => { void send(true).catch(options.failure); });
  button("chat-new").addEventListener("click", () => { void switchConversation().catch(options.failure); });
  button("chat-add").addEventListener("click", () => setMenu(Boolean($("chat-attachment-menu").hidden)));
  for (const kind of ["file", "folder"]) {
    button("chat-pick-" + kind).addEventListener("click", () => { setMenu(false); $("chat-" + kind).click(); });
    $("chat-" + kind).addEventListener("change", event => { const input = event.target as HTMLInputElement; void addFiles(Array.from(input.files || [])); input.value = ""; });
  }
  document.addEventListener("click", event => { if (event.target instanceof Node && !$("chat-add").parentElement!.contains(event.target)) setMenu(false); });
  $("chat-form").addEventListener("keydown", event => { if (event.key === "Escape") { setMenu(false); button("chat-add").focus(); } });
  window.addEventListener("luban-chat-files", event => {
    if (!(event instanceof CustomEvent) || !Array.isArray(event.detail)) return;
    try {
      const files = event.detail.map((item: { name: string; type: string; base64: string }) => {
        const bytes = Uint8Array.from(atob(item.base64), character => character.charCodeAt(0));
        return new File([bytes], item.name, { type: item.type });
      });
      void addFiles(files);
    } catch { notify(t("剪贴板附件读取失败，请用 ＋ 选择文件。")); }
  });
  window.addEventListener("luban-chat-paste-error", event => { if (event instanceof CustomEvent) notify(event.detail.message ? t(event.detail.message) : t("剪贴板文件读取失败。")); });
  field().addEventListener("paste", event => {
    const files = Array.from(event.clipboardData?.files || []);
    if (files.length) { event.preventDefault(); void addFiles(files); }
  });
  field().addEventListener("input", () => { renderControls(); scheduleSave(); });
  field().addEventListener("keydown", event => { if (enterSends(event)) { event.preventDefault(); void send(); } });
  $("chat-form").addEventListener("submit", event => { event.preventDefault(); void send().catch(options.failure); });
  button("chat-stop").addEventListener("click", () => { void stop(); });
  for (const id of ["chat-route", "chat-thinking"]) $(id).addEventListener("change", () => { renderAttachments(); scheduleSave(); });
  // Preserve drafts on tab changes; streaming saves are independent of navigation.
  document.addEventListener("visibilitychange", () => { if (loaded) void persist().catch(options.failure); });
  window.addEventListener("beforeunload", event => { if (busy || adding || saveTimer || storageError) { if (loaded) void persist().catch(options.failure); event.preventDefault(); event.returnValue = ""; } });
  async function load() {
    db = await conversationDatabase(); conversations = await readConversations(db);
    try {
      const existing = await options.api<{ conversations: { id: string; title: string; updated: number }[] }>("/api/conversations");
      for (const item of existing.conversations) if (!conversations.some(value => value.serverId === item.id)) {
        const saved = await options.api<{ messages: ChatMessage[] }>("/api/conversations/" + encodeURIComponent(item.id));
        const conversation = { ...newConversation(), serverId: item.id, title: item.title, updated: item.updated * 1000, messages: saved.messages };
        conversations.push(conversation); await saveConversation(db, conversation);
      }
    } catch { /* Browser drafts remain available even while the model data scope is unavailable. */ }
    current = conversations.sort((a, b) => b.updated - a.updated)[0] || newConversation(); loaded = true;
    renderConversation(); $("chat-save-state").textContent = t("会话保存在本机");
  }
  async function prepareLanguageChange() {
    if (busy || adding) throw new Error(t("请等待当前对话完成后切换语言"));
    if (loaded) await persist();
    await queue;
  }
  return { load, renderControls, isBusy: () => busy || adding, prepareLanguageChange };
}
