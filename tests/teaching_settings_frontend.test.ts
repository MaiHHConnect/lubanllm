import { expect, test } from "bun:test";
import { t } from "../frontend/i18n";
import { english } from "../frontend/locales/en";

// Exercise the production form load/save functions without starting the app or network.
async function form(saved: () => any = () => settings()) {
  const source = await Bun.file(new URL("../frontend/app.ts", import.meta.url)).text();
  const functions = source.slice(source.indexOf("function applySettings("), source.indexOf("async function saveTeacher("));
  const compiled = new Bun.Transpiler({ loader: "ts" }).transformSync(functions);
  const fields = new Map<string, any>();
  const input = (id: string) => { if (!fields.has(id)) fields.set(id, { value: "", checked: false }); return fields.get(id); };
  const requests: string[] = [];
  const actions = new Function("input", "select", "text", "t", "chat", "api", `
    let teacherFirst=false, teacherConfigured=false, settingsEdited=false, automaticLearningSaveBusy=false;
    ${compiled}
    return { applySettings, refreshSavedSettings, teacherPayload, edit:()=>settingsEdited=true, isTeacherFirst:()=>teacherFirst, isTeacherConfigured:()=>teacherConfigured };
  `)(input, input, () => {}, (value: string | string[]) => typeof value === "string" ? value : value.join(""), { renderControls() {} },
    async (path: string) => { requests.push(path); return saved(); });
  return { ...actions, input, requests };
}
const settings = (modalities?: string[]) => ({ model_path: "/offline", storage_root: "/offline", local_thinking: "off",
  context_mode: "auto", temperature: 0.6, teacher: { answer_mode: "teacher_first", name: "Offline",
    base_url: "https://teacher.invalid", model: "offline", key_configured: true, thinking: "default",
    send_reasoning_effort: false, session_call_limit: 20, ...(modalities === undefined ? {} : { teaching_modalities: modalities }) } });

test("automatic learning defaults on and restores the saved switch without replacing teacher drafts", async () => {
  const ui = await form();
  ui.applySettings(settings());
  expect(ui.input("learning-automatic").checked).toBe(true);
  expect(ui.input("learning-automatic").disabled).toBe(false);
  ui.input("teacher-model").value = "draft"; ui.edit();
  ui.applySettings({ ...settings(), automatic: { learning_enabled: false } }, true);
  expect(ui.input("learning-automatic").checked).toBe(false);
  expect(ui.input("teacher-model").value).toBe("draft");
});

test("teacher form loads legacy text default and saves canonical checked scope including empty", async () => {
  const ui = await form(); ui.applySettings(settings());
  expect(ui.isTeacherFirst()).toBe(true);
  expect(["text", "image", "video", "audio"].map(kind => ui.input(`teacher-teach-${kind}`).checked)).toEqual([true, false, false, false]);
  expect(ui.teacherPayload().teaching_modalities).toEqual(["text"]);
  for (const kind of ["text", "image", "video", "audio"]) ui.input(`teacher-teach-${kind}`).checked = kind !== "text";
  expect(ui.teacherPayload().teaching_modalities).toEqual(["image", "video", "audio"]);
  ui.applySettings(settings([])); expect(ui.teacherPayload().teaching_modalities).toEqual([]);
  expect(ui.isTeacherFirst()).toBe(false);
});

test("text teacher-first requires text scope, mode, configured key and endpoint/model", async () => {
  const ui = await form();
  for (const modalities of [[], ["image"], ["audio", "video"]]) {
    ui.applySettings(settings(modalities)); expect(ui.isTeacherFirst()).toBe(false);
  }
  ui.applySettings(settings(["image", "text"])); expect(ui.isTeacherFirst()).toBe(true);
  for (const teacher of [{ answer_mode: "local_first" }, { key_configured: false }, { base_url: "" }, { model: "" }]) {
    const value = settings(["text"]); Object.assign(value.teacher, teacher);
    ui.applySettings(value); expect(ui.isTeacherFirst()).toBe(false);
  }
});

test("four labeled checkboxes share the existing form and disclose frame-only video teaching", async () => {
  const html = await Bun.file(new URL("../frontend/index.html", import.meta.url)).text();
  const teacher = html.slice(html.indexOf('<form id="teacher-form"'), html.indexOf('</form>', html.indexOf('<form id="teacher-form"')));
  for (const kind of ["text", "image", "video", "audio"]) {
    expect(teacher).toContain(`id="teacher-teach-${kind}" type="checkbox"`);
  }
  expect(teacher).toContain("<legend>教什么</legend>");
  expect(teacher).not.toContain("视频处理尚未接通"); expect(teacher).toContain("每段视频最多4帧按时间共同标注，1次请求，不含音轨或全部画面"); expect(teacher).toContain("仅发送已选择且明确提交的资料");
});

test("answer mode uses three selectable cards and persists automatic selection", async () => {
  const html = await Bun.file(new URL("../frontend/index.html", import.meta.url)).text();
  const teacher = html.slice(html.indexOf('<form id="teacher-form"'), html.indexOf('</form>', html.indexOf('<form id="teacher-form"')));
  for (const mode of ["local_first", "teacher_first", "auto"]) {
    expect(teacher).toContain(`name="teacher-answer-mode" type="radio" value="${mode}"`);
  }
  const ui = await form();
  const value = settings(); value.teacher.answer_mode = "auto";
  ui.applySettings(value);
  expect(ui.input("teacher-answer-auto").checked).toBe(true);
  expect(ui.teacherPayload().answer_mode).toBe("auto");
});


test("learning navigation requires saved teacher endpoint, key, model and teaching scope", async () => {
  const ui = await form(); ui.applySettings(settings()); expect(ui.isTeacherConfigured()).toBe(true);
  for (const teacher of [{ key_configured: false }, { base_url: " " }, { model: "" }, { teaching_modalities: [] }]) {
    const value = settings(); Object.assign(value.teacher, teacher); ui.applySettings(value); expect(ui.isTeacherConfigured()).toBe(false);
  }
});

test("entering learning reloads saved teacher availability and preserves unsaved form edits", async () => {
  const ui = await form(() => settings(["text"]));
  ui.applySettings(settings([])); expect(ui.isTeacherConfigured()).toBe(false);
  await ui.refreshSavedSettings(); expect(ui.isTeacherConfigured()).toBe(true);
  expect(ui.teacherPayload().teaching_modalities).toEqual(["text"]);
  ui.input("teacher-model").value = "unsaved-model"; ui.input("teacher-key").value = "unsaved-key"; ui.edit();
  ui.applySettings(settings([]), true);
  expect(ui.isTeacherConfigured()).toBe(false);
  await ui.refreshSavedSettings(); expect(ui.isTeacherConfigured()).toBe(true);
  expect(ui.input("teacher-model").value).toBe("unsaved-model"); expect(ui.input("teacher-key").value).toBe("unsaved-key");
  expect(ui.requests).toEqual(["/api/settings", "/api/settings"]);
});

test("context capacity distinguishes zero-memory admission from unloaded and positive windows without changing settings", async () => {
  const source = await Bun.file(new URL("../frontend/app.ts", import.meta.url)).text();
  const statement = source.slice(source.indexOf("  const context = runtime.context;"), source.indexOf("  home?.renderRuntime"));
  let rendered = "";
  const render = new Function("runtime", "text", "t", statement);
  const show = (current_limit: number | null) => {
    const context = Object.freeze({ current_limit, mode: "auto", configured_limit: 262144, model_limit: 262144, estimated: true });
    render({ context }, (id: string, value: string) => { expect(id).toBe("context-summary"); rendered = value; }, t);
    expect(context.configured_limit).toBe(262144); return rendered;
  };
  const zero = "当前内存不足，暂无法分配上下文；容量设置已保留。";
  expect(show(0)).toBe(zero); expect(english[zero]).toBe("Not enough memory to allocate context right now. Your capacity setting is retained.");
  expect(show(null)).toContain("模型加载后");
  expect(show(4608)).toContain("当前4.5K");
  expect(show(32768)).toContain("当前32.0K");
});
