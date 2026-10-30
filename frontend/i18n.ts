import { english } from "./locales/en";

export type Language = "zh" | "en";
const STORAGE_KEY = "luban.language";
export const LEARNING_SOURCE_RECHECK_NOTICE = "专家学习来源已变化，正在重新核对资料。";
export function resolveLanguage(saved: unknown, preferred: string | undefined): Language {
  return saved === "zh" || saved === "en" ? saved : /^zh(?:-|$)/i.test(preferred || "") ? "zh" : "en";
}
// Node tests and non-browser consumers retain the original Chinese source messages.
let current: Language = "zh";
if (typeof window !== "undefined") {
  let saved: string | null = null;
  try { saved = window.localStorage.getItem(STORAGE_KEY); } catch { /* System preference still works without storage. */ }
  current = resolveLanguage(saved, navigator.language);
}
export const language = () => current;
export const locale = () => current === "zh" ? "zh-CN" : "en-US";

export function t(source: string | TemplateStringsArray, ...values: unknown[]): string {
  const key = typeof source === "string" ? source : source.reduce((result, part, index) => result + (index ? `{${index - 1}}` : "") + part, "");
  const translated = current === "en" ? english[key] || key : key;
  return values.length ? translated.replace(/\{(\d+)\}/g, (match, index) => Number(index) < values.length ? String(values[Number(index)] ?? "") : match) : translated;
}

export function localizeDocument(root: Document) {
  root.documentElement.lang = current === "zh" ? "zh-CN" : "en";
  const walker = root.createTreeWalker(root.documentElement, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.parentElement?.closest("script, style, textarea, [data-no-translate]")) continue;
    const value = node.nodeValue || "", trimmed = value.trim();
    if (trimmed) node.nodeValue = value.replace(trimmed, t(trimmed));
  }
  for (const element of root.querySelectorAll("[title], [placeholder], [aria-label], [alt]")) {
    if (element.closest("[data-no-translate]")) continue;
    for (const attribute of ["title", "placeholder", "aria-label", "alt"]) {
      const value = element.getAttribute(attribute);
      if (value) element.setAttribute(attribute, t(value));
    }
  }
}

export function initializeLanguage(beforeChange: () => Promise<void>, failure: (error: unknown) => void) {
  const select = document.getElementById("interface-language") as HTMLSelectElement;
  select.value = current;
  select.addEventListener("change", () => {
    const next = select.value;
    if (next !== "zh" && next !== "en" || next === current) return;
    select.disabled = true;
    void beforeChange().then(() => {
      try { window.localStorage.setItem(STORAGE_KEY, next); }
      catch { throw new Error(t("无法保存语言设置，请检查本机存储后重试")); }
      // Explicit language changes reload only the UI; the local model service keeps running.
      window.location.reload();
    }).catch(error => { select.value = current; select.disabled = false; failure(error); });
  });
}
