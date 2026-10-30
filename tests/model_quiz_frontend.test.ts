import { expect, test } from "bun:test";
import { initializeModelQuiz, quizResultText } from "../frontend/model-quiz";

const result = { question_count: 6, correct: 4, original_count: 1, new_count: 5,
  previously_wrong_count: 0, newly_correct: null, regressed: null, teacher_calls: 0, private: true };

test("quiz reports actual questions and keeps unknown progress separate from capabilities", () => {
  const text = quizResultText(result);
  expect(text).toContain("本次测试 6 题，独立答对 4 题");
  expect(text).toContain("暂无以前独立答错");
  expect(text).toContain("暂不比较进步");
  const comparison = quizResultText({ ...result, previously_wrong_count: 1, newly_correct: 2, regressed: 1 });
  expect(comparison).toContain("2 题由错变对，1 题由对变错");
  expect(comparison).not.toContain("掌握 2 项能力");
  for (const invalid of [{ correct: 7 }, { teacher_calls: 1 }, { private: false }, { question_count: 10 }, { newly_correct: NaN, regressed: 0 }])
    expect(() => quizResultText({ ...result, ...invalid })).toThrow();
});

test("quiz opens its own local workflow and stops without calling teacher or leaderboard", async () => {
  const old = (globalThis as any).document;
  const elements = new Map<string, any>();
  const element = (id: string) => {
    if (!elements.has(id)) elements.set(id, { hidden: false, disabled: false, textContent: "", events: {} as Record<string, () => void>,
      addEventListener(name: string, fn: () => void) { this.events[name] = fn; },
      setAttribute() {}, removeAttribute() {}, showModal() { this.hidden = false; }, close() { this.hidden = true; } });
    return elements.get(id);
  };
  let state: any = { state: "idle" }, ready = true;
  const paths: string[] = [];
  const settle = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
  try {
    (globalThis as any).document = { getElementById: element };
    const quiz = initializeModelQuiz({ ready: () => ready, refresh: async () => {}, api: async <T>(path: string) => {
      paths.push(path);
      if (path.endsWith("/start")) state = { state: "running", completed: 1, total: 6 };
      if (path.endsWith("/cancel")) state = { state: "cancelled" };
      return structuredClone(state) as T;
    } });
    element("model-quiz-open").events.click(); await settle();
    expect(paths).toEqual(["/api/model-quiz"]);
    ready = false; quiz.renderControls();
    expect(element("model-quiz-start").disabled).toBe(true);
    element("model-quiz-start").events.click(); await settle();
    expect(paths).toHaveLength(1);
    ready = true; quiz.renderControls();
    element("model-quiz-start").events.click(); await settle();
    expect(element("model-quiz-state").textContent).toContain("1 / 6 题");
    expect(element("model-quiz-stop").hidden).toBe(false);
    element("model-quiz-stop").events.click(); await settle();
    expect(element("model-quiz-state").textContent).toContain("本次未保存成绩");
    expect(paths.every(path => path.startsWith("/api/model-quiz"))).toBe(true);
    state = { state: "complete", result };
    element("model-quiz-open").events.click(); await settle();
    expect(element("model-quiz-result").textContent).toContain("独立答对 4 题");
    element("model-quiz-close").events.click();
    expect(element("model-quiz-dialog").hidden).toBe(true);
  } finally { (globalThis as any).document = old; }
});
