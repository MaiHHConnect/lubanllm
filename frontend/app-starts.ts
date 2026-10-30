const INSTALL_KEY = "luban.anonymous-install";
let reported = false;

// A random installation ID counts devices, not people. No model or personal content is sent.
export async function reportAppStart() {
  if (reported) return;
  reported = true;
  try {
    const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    let install = localStorage.getItem(INSTALL_KEY);
    if (!install || !uuid.test(install)) {
      install = crypto.randomUUID();
      localStorage.setItem(INSTALL_KEY, install);
    }
    await fetch("https://ai.maihh.net/api/app-start", { method: "POST", credentials: "omit", redirect: "error",
      referrerPolicy: "no-referrer", signal: AbortSignal.timeout(5000), headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schema: 1, install_id: install, event_id: crypto.randomUUID() }) });
  } catch { /* Optional usage counts never block startup or retry in the background. */ }
}
