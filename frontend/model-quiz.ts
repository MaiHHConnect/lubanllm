import { t } from "./i18n";
type Result = { question_count: number; correct: number; original_count: number; new_count: number; previously_wrong_count: number; newly_correct: number | null; regressed: number | null; teacher_calls: number; private: boolean };
type Quiz = { state: string; completed?: number; total?: number; result?: Result | null; error?: string | null };
const active = (state?: string) => state === "queued" || state === "running";
const number = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;

export function quizResultText(result: Result): string {
  if (!number(result.question_count) || result.question_count < 1 || result.question_count > 10 || !number(result.correct)
    || result.correct > result.question_count || !number(result.original_count) || !number(result.new_count)
    || result.original_count + result.new_count !== result.question_count || !number(result.previously_wrong_count)
    || result.previously_wrong_count > result.original_count || result.teacher_calls !== 0 || result.private !== true)
    throw new Error(t("小测结果不完整，请重新读取"));
  const lines = [t`本次测试 ${result.question_count} 题，独立答对 ${result.correct} 题。`,
    t`教材原题 ${result.original_count} 题 · 同类新题 ${result.new_count} 题`];
  if (result.previously_wrong_count === 0) lines.push(t("暂无以前独立答错的原题记录，本次为教材小测。"));
  else lines.push(t`其中 ${result.previously_wrong_count} 道原题有此前独立答错的记录。`);
  if (result.newly_correct === null || result.regressed === null) lines.push(t("暂无同题集、不同模型版本的前次成绩，暂不比较进步。"));
  else {
    if (!number(result.newly_correct) || !number(result.regressed) || result.newly_correct > result.question_count || result.regressed > result.question_count)
      throw new Error(t("小测对照结果无效"));
    lines.push(t`相比前次同题小测，${result.newly_correct} 题由错变对，${result.regressed} 题由对变错。`);
  }
  return lines.join("\n");
}

export function initializeModelQuiz(options: { api: <T = Record<string, unknown>>(path: string, data?: unknown) => Promise<T>; ready: () => boolean; refresh: () => Promise<unknown> }) {
  const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
  const button = (id: string) => element<HTMLButtonElement>(id);
  const dialog = element<HTMLDialogElement>("model-quiz-dialog");
  let current: Quiz = { state: "idle" }, requesting = false, timer: ReturnType<typeof setTimeout> | undefined;
  function renderControls() {
    const busy = active(current.state);
    button("model-quiz-open").textContent = busy ? t("小测进行中…") : t("考考我的 AI");
    button("model-quiz-start").disabled = requesting || busy || !options.ready();
    button("model-quiz-stop").hidden = !busy;
    button("model-quiz-stop").disabled = requesting;
    const progress = element<HTMLProgressElement>("model-quiz-progress");
    progress.hidden = !busy;
    progress.max = number(current.total) && current.total > 0 ? current.total : 1;
    if (number(current.completed)) progress.value = current.completed;
    else progress.removeAttribute("value");
  }
  function render() {
    const state = element("model-quiz-state");
    state.setAttribute("role", current.error ? "alert" : "status");
    state.textContent = (current.error ? t(current.error) : "") || (active(current.state) ? t`正在独立答题：${number(current.completed) ? current.completed : 0} / ${number(current.total) ? current.total : "—"} 题`
      : current.state === "waiting_materials" ? t("暂无可自动核对答案的教材。先教 AI 算术、字母大写或 JSON 提取，再来小测。")
      : current.state === "cancelled" ? t("小测已停止，本次未保存成绩。")
      : current.state === "complete" ? t("小测完成") : current.state === "idle" && current.result ? t("上次小测（本机保存）")
      : t("全程由本地模型独立答题，关闭教师与参考资料，结果仅保存在本机。"));
    const result = element("model-quiz-result");
    result.hidden = !current.result;
    result.textContent = current.result ? quizResultText(current.result) : "";
    renderControls();
  }
  function schedule() {
    clearTimeout(timer);
    if (active(current.state)) timer = setTimeout(() => { void refresh(); }, 1000);
  }
  async function refresh() {
    try { current = await options.api<Quiz>("/api/model-quiz"); render(); }
    catch (error) { current = { state: "error", error: error instanceof Error ? t(error.message) : t("读取小测失败") }; render(); }
    schedule();
  }
  async function action(path: string) {
    if (requesting) return;
    requesting = true; renderControls();
    try { await options.api(path, {}); await refresh(); await options.refresh(); }
    catch (error) { current = { state: "error", error: error instanceof Error ? t(error.message) : t("小测未能开始") }; render(); }
    finally { requesting = false; renderControls(); }
  }
  button("model-quiz-open").addEventListener("click", () => { dialog.showModal(); void refresh(); });
  button("model-quiz-close").addEventListener("click", () => dialog.close());
  button("model-quiz-start").addEventListener("click", () => {
    if (!active(current.state) && options.ready()) void action("/api/model-quiz/start");
  });
  button("model-quiz-stop").addEventListener("click", () => { if (active(current.state)) void action("/api/model-quiz/cancel"); });
  render();
  return { renderControls };
}
