import { language } from "./i18n";
const ORIGIN = "https://ai.maihh.net";
const MAX_BODY = 16 * 1024;
const LAST_SHOWN_DATE = "luban.app-ad.last-shown-date";
const initializedPopups = new WeakMap<HTMLElement, () => void>();
type Labels = { zh: string; en: string };
export function validatedAd(value: unknown) {
  if (!value || typeof value !== "object") throw Error("Invalid advertisement");
  const ad = value as Record<string, unknown>;
  const labels = (field: string, max: number): Labels => {
    const item = ad[field] as Labels | undefined;
    if (!item || [item.zh, item.en].some(text => typeof text !== "string" || !text.trim()
      || [...text].length > max || /[\u0000-\u001f\u007f-\u009f]/.test(text))) throw Error("Invalid advertisement copy");
    return item;
  };
  const href = new URL(String(ad.href));
  if (ad.schema !== 1 || href.origin !== ORIGIN || href.username || href.password) throw Error("Invalid advertisement target");
  if (typeof ad.image !== "string" || !/^\/(?:app-ad\/luban-intro-v1\.png|api\/app-ad\/image\/[a-f0-9]{64}\.png)$/.test(ad.image))
    throw Error("Invalid advertisement image");
  return { title: labels("title", 60), subtitle: labels("subtitle", 160), cta: labels("cta", 40),
    href: href.href, image: ORIGIN + ad.image };
}
export async function refreshAppAd() {
  try {
    const response = await fetch(ORIGIN + "/api/app-ad", { credentials: "omit", redirect: "error",
      cache: "no-store", referrerPolicy: "no-referrer", signal: AbortSignal.timeout(5000) });
    if (!response.ok || !response.body) return;
    const reader = response.body.getReader(), parts: Uint8Array[] = [];
    let size = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > MAX_BODY) { await reader.cancel(); return; }
        parts.push(value);
      }
    } finally { reader.releaseLock(); }
    const bytes = new Uint8Array(size); let offset = 0;
    for (const part of parts) { bytes.set(part, offset); offset += part.length; }
    const ad = validatedAd(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
    const anchor = document.getElementById("app-ad") as HTMLAnchorElement;
    const image = document.getElementById("app-ad-image") as HTMLImageElement;
    const selectedLanguage = language();
    // Load a complete image before changing the visible copy; offline clients keep the bundled ad.
    const probe = new Image(); probe.referrerPolicy = "no-referrer"; probe.src = ad.image;
    await probe.decode();
    anchor.href = ad.href; anchor.setAttribute("aria-label", `${ad.title[selectedLanguage]}: ${ad.cta[selectedLanguage]}`);
    document.getElementById("app-ad-title")!.textContent = ad.title[selectedLanguage];
    document.getElementById("app-ad-subtitle")!.textContent = ad.subtitle[selectedLanguage];
    document.getElementById("app-ad-cta")!.textContent = ad.cta[selectedLanguage] + " ↗";
    image.referrerPolicy = "no-referrer"; image.src = ad.image;
  } catch { /* Ads never block local model use; keep the bundled introduction. */ }
}
export function initializeAppAd() {
  const popup = document.getElementById("app-ad-popup");
  const close = document.getElementById("app-ad-close");
  if (!popup || !close) return () => {};
  const initialized = initializedPopups.get(popup);
  if (initialized) return initialized;
  let timer: ReturnType<typeof setInterval> | undefined;
  const stopRefresh = () => { if (timer !== undefined) clearInterval(timer); timer = undefined; };
  const dismiss = () => { popup.hidden = true; stopRefresh(); };
  const dispose = () => { stopRefresh(); close.removeEventListener("click", dismiss); };
  initializedPopups.set(popup, dispose);
  close.addEventListener("click", dismiss);
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  let alreadyShown = false;
  try {
    alreadyShown = localStorage.getItem(LAST_SHOWN_DATE) === today;
    if (!alreadyShown) localStorage.setItem(LAST_SHOWN_DATE, today);
  } catch { /* Storage can be unavailable; this page still shows only once. */ }
  popup.hidden = alreadyShown;
  if (!alreadyShown) {
    void refreshAppAd();
    timer = setInterval(() => {
      if (popup.hidden) stopRefresh();
      else if (!document.hidden) void refreshAppAd();
    }, 5 * 60 * 1000);
  }
  return dispose;
}
