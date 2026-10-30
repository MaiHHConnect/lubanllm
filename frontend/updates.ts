import { t } from "./i18n";
type UpdateState = { current_version: string; current_build: number; state: "idle" | "checking" | "downloading" | "ready" | "installing" | "error"; version?: string; build?: number; notes?: string[]; error?: string };
declare global { interface Window { webkit?: { messageHandlers?: { lubanUpdate?: { postMessage: (message: { action: string }) => void } } } } }
const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const bridge = () => window.webkit?.messageHandlers?.lubanUpdate;
async function loadComponentVersions() {
  const target = element("component-versions");
  try {
    const response = await fetch("/component-versions.json", { credentials: "same-origin", cache: "no-store" });
    if (!response.ok) throw new Error();
    const manifest = await response.json() as { schema: number; components: { id: string; name: string; version: string }[]; pending_components?: { name: string; status: string }[] };
    if (manifest.schema !== 1 || !Array.isArray(manifest.components) || !manifest.components.length
      || manifest.components.some(c => typeof c.name !== "string" || !/^\d+\.\d+\.\d+$/.test(c.version))) throw new Error();
    target.replaceChildren();
    for (const component of manifest.components) {
      const row = document.createElement("div"); row.className = "version";
      const name = document.createElement("span"), version = document.createElement("strong");
      name.textContent = t(component.name); version.textContent = component.version;
      row.append(name, version); target.append(row);
    }
    for (const component of manifest.pending_components || []) {
      if (typeof component.name !== "string" || typeof component.status !== "string") throw new Error();
      const row = document.createElement("div"); row.className = "version";
      const name = document.createElement("span"), status = document.createElement("strong");
      name.textContent = t(component.name); status.textContent = manifest.components.find(item => item.id === "training")?.version || "—";
      status.title = t(component.status);
      row.append(name, status); target.append(row);
    }
  } catch { target.textContent = t("无法读取引擎版本，请重新构建应用。"); }
}
let update: UpdateState | undefined;
let manualCheck = false;
function send(action: string) { bridge()?.postMessage({ action }); }
function showNotes() {
  if (update?.state !== "ready") return;
  element("update-target").textContent = `${update.current_version} → ${update.version}`;
  const notes = element("update-notes"); notes.replaceChildren();
  for (const note of update.notes || []) { const item = document.createElement("li"); item.textContent = note; notes.append(item); }
  element("update-dialog-error").hidden = true;
  const dialog = element<HTMLDialogElement>("update-dialog"); if (!dialog.open) dialog.showModal();
}
function render(next: UpdateState) {
  update = next;
  element("client-version").textContent = t`当前版本 ${next.current_version}（构建 ${next.current_build}）`;
  const labels = { idle: t("已是最新版本。运行期间会自动检查更新。"), checking: t("正在检查更新…"), downloading: t("正在静默下载更新，不影响当前工作。"), ready: t("更新已下载并校验，重启即可更新。"), installing: t("正在安装，应用即将重启…"), error: next.error ? t(next.error) : t("检查更新失败，稍后会自动重试。") };
  element("update-state").textContent = labels[next.state];
  element("update-state").hidden = !manualCheck && !["checking", "downloading", "installing"].includes(next.state) || next.state === "idle";
  if (!["checking", "downloading"].includes(next.state)) manualCheck = false;
  element<HTMLButtonElement>("check-update").disabled = ["checking", "downloading", "installing"].includes(next.state);
  element("view-update").hidden = next.state !== "ready";
  element("update-notice").hidden = next.state !== "ready";
  element("update-notice-text").textContent = t("有新更新");
  element<HTMLButtonElement>("install-update").disabled = next.state !== "ready";
  element<HTMLButtonElement>("close-update").disabled = next.state === "installing";
  if (element<HTMLDialogElement>("update-dialog").open && next.error) { element("update-dialog-error").textContent = t(next.error); element("update-dialog-error").hidden = false; }
}
export async function initializeUpdates() {
  for (const target of document.querySelectorAll<HTMLButtonElement>("[data-setting]")) target.addEventListener("click", () => {
    for (const page of document.querySelectorAll<HTMLElement>(".settings-page")) page.hidden = page.id !== `settings-${target.dataset.setting}`;
    for (const item of document.querySelectorAll<HTMLButtonElement>("[data-setting]")) { const active = item === target; item.classList.toggle("active", active); if (active) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current"); }
    if (target.dataset.setting === "update") send("status");
  });
  window.addEventListener("luban-update", event => render((event as CustomEvent<UpdateState>).detail));
  element("check-update").addEventListener("click", () => { manualCheck = true; send("check"); });
  for (const id of ["view-update", "notice-update"]) element(id).addEventListener("click", showNotes);
  element("close-update").addEventListener("click", () => element<HTMLDialogElement>("update-dialog").close());
  element("update-dialog").addEventListener("cancel", event => { if (update?.state === "installing") event.preventDefault(); });
  element("install-update").addEventListener("click", () => {
    if (update?.state !== "ready") return;
    element<HTMLButtonElement>("install-update").disabled = true;
    element("update-dialog-error").hidden = true;
    send("install");
  });
  await loadComponentVersions();
  if (bridge()) { send("status"); return; }
  const response = await fetch("/api/version", { credentials: "same-origin" });
  if (!response.ok) throw new Error(t("无法读取客户端版本"));
  const metadata = await response.json() as { version: string; build: number };
  element("client-version").textContent = t`当前版本 ${metadata.version}（构建 ${metadata.build}）`;
  element("check-update").title = t("请在鲁班零号 Mac 应用中检查和安装更新。");
  element("update-state").hidden = true;
  element<HTMLButtonElement>("check-update").disabled = true;
}
