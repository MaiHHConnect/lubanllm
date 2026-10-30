import { t } from "./i18n";
type EvolutionRuntime = { job?: { id?: string; kind?: string; state: string; result?: { id?: string; promoted?: boolean; publication_state?: string } } };
const POSTER = "/assets/share/evolution-success.png";
const SHARE_TEXT = t("我的鲁班零号 AI 完成了一轮进化，通过能力验证，新的学习成果已保留！和我一起培养你的专属 AI：https://ai.maihh.net/");
const COPIED = t("模型又强大啦，快去分享好友！");
const KEY = "luban-evolution-share-seen";

// Only a live observed job may cause a celebration. Restored history is never replayed.
export function createEvolutionShareObserver(onSuccess: () => void, prior: string[] = [], persist: (ids: string[]) => void = () => {}) {
  const seen = new Set(prior.filter(id => typeof id === "string").slice(-50));
  let active: string | undefined;
  return (runtime: EvolutionRuntime) => {
    const job = runtime.job;
    if (job?.kind !== "evolution" || typeof job.id !== "string" || !job.id || job.id.length > 200) { active = undefined; return; }
    if (["running", "queued", "paused"].includes(job.state)) { active = seen.has(job.id) ? undefined : job.id; return; }
    const eligible = active === job.id && !seen.has(job.id) && job.state === "complete"
      && job.result?.id === job.id && job.result.promoted === true && job.result.publication_state === "published";
    active = undefined;
    if (!["complete", "error", "rejected", "cancelled"].includes(job.state)) return;
    seen.add(job.id);
    while (seen.size > 50) seen.delete(seen.values().next().value!);
    persist([...seen]);
    if (eligible) onSuccess();
  };
}

async function posterBlob(): Promise<Blob> {
  const response = await fetch(POSTER, { credentials: "same-origin", signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw Error(t("分享图片读取失败，请稍后重试"));
  const blob = await response.blob();
  if (blob.type !== "image/png" || blob.size === 0 || blob.size > 3 * 1024 * 1024) throw Error(t("分享图片格式或大小不正确"));
  return blob;
}

export async function copyEvolutionImage(blob: Promise<Blob>) {
  const bridge = (window as Window & { webkit?: { messageHandlers?: { lubanShare?: { postMessage: (value: unknown) => void } } } }).webkit?.messageHandlers?.lubanShare;
  if (!bridge) {
    if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") throw Error(t("当前窗口不支持复制图片，请在鲁班零号 Mac 应用中分享"));
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    return;
  }
  const bytes = new Uint8Array(await (await blob).arrayBuffer());
  if (!bytes.length || bytes.length > 3 * 1024 * 1024) throw Error(t("分享图片大小不正确"));
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  const id = crypto.randomUUID();
  await new Promise<void>((resolve, reject) => {
    const finish = (error?: Error) => { clearTimeout(timer); window.removeEventListener("luban-share-copied", receive); error ? reject(error) : resolve(); };
    const receive = (event: Event) => {
      const result = (event as CustomEvent).detail;
      if (result?.id === id) finish(result.success === true ? undefined : Error(t("图片未能复制，请重试")));
    };
    const timer = setTimeout(() => finish(Error(t("复制图片未收到确认，请重试"))), 10000);
    window.addEventListener("luban-share-copied", receive);
    try { bridge.postMessage({ action: "copy-image", id, png: btoa(binary) }); }
    catch { finish(Error(t("图片未能复制，请重试"))); }
  });
}

export function initializeEvolutionShare(actions: { toast: (message: string) => void; failure: (error: unknown) => void }) {
  const get = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
  const dialog = get<HTMLDialogElement>("evolution-share-dialog");
  let image: Promise<Blob> | undefined;
  let prior: string[] = [];
  try { const saved: unknown = JSON.parse(sessionStorage.getItem(KEY) || "[]"); if (Array.isArray(saved)) prior = saved; } catch { /* private mode: use in-memory deduplication */ }
  const observeRuntime = createEvolutionShareObserver(() => {
    if (!dialog.open) dialog.showModal();
    image = posterBlob();
    void image.catch(() => { image = undefined; });
  }, prior, ids => { try { sessionStorage.setItem(KEY, JSON.stringify(ids)); } catch { /* same live session remains deduplicated */ } });
  get("evolution-share-close").addEventListener("click", () => dialog.close());
  const copy = (id: string, action: () => Promise<void>) => {
    const button = get<HTMLButtonElement>(id);
    button.addEventListener("click", async () => {
      if (button.disabled) return;
      button.disabled = true;
      try { await action(); actions.toast(COPIED); }
      catch (error) { actions.failure(error); }
      finally { button.disabled = false; }
    });
  };
  copy("evolution-share-copy-text", async () => {
    if (!navigator.clipboard?.writeText) throw Error(t("当前窗口不支持复制文本，请在鲁班零号 Mac 应用中分享"));
    await navigator.clipboard.writeText(SHARE_TEXT);
  });
  copy("evolution-share-copy-image", () => copyEvolutionImage(image || posterBlob()));
  return { observeRuntime };
}
