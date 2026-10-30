/** The existing finite contracts; never execute generated code or revise scoring. */
import { pythonStrip } from "./memory";
import { parseJSON } from "./wire";
import { encoded } from "./canonical-json";
import unicode from "./unicode-15.json";
import goals from "./goal-topics.json";

const finite = new Set(["integer", "json", "python_upper", "translation", "normalized_exact", "one_of", "goal_topics"]);
const validators = new Set([...finite, "nonempty"]);
const folding = new Map(unicode.folding.map(([code, mapped]) => [code as number, String.fromCodePoint(...mapped as number[])]));
const sentences: Record<string, string[]> = goals.sentences;
const required: Record<string, string[]> = goals.required;
const length = (value: string) => Array.from(value).length;
const object = (value: any): value is Record<string, any> => value !== null && typeof value === "object" && [Object.prototype, null].includes(Object.getPrototypeOf(value));
const need = (condition: unknown, message: string) => { if (!condition) throw new Error(message); };
const integer = (value: any) => typeof value === "bigint" || typeof value === "number" && Number.isSafeInteger(value);

export const pythonCasefold = (value: string) => Array.from(value, char => folding.get(char.codePointAt(0)!) ?? char).join("");
export function pythonNFKC(value: string) {
  let run = "", result = "";
  for (const char of value) {
    const code = char.codePointAt(0)!; let low = 0, high = unicode.assignedRanges.length - 1, assigned = false;
    while (low <= high) {
      const middle = (low + high) >>> 1, [start, end] = unicode.assignedRanges[middle]!;
      if (code < start!) high = middle - 1; else if (code > end!) low = middle + 1; else { assigned = true; break; }
    }
    // In Python15 unassigned characters have combining class zero and no
    // decomposition. They form boundaries, even if this platform knows newer data.
    if (assigned) run += char; else { result += run.normalize("NFKC") + char; run = ""; }
  }
  return result + run.normalize("NFKC");
}

// Typed canonical values retain Python json.dumps distinctions, including -0.0,
// while equivalent float spellings (1e0/1.0) compare as the same IEEE-754 value.
// Escape strings before using snapshot encoded so JSON's valid lone surrogates
// remain comparable rather than crossing the UTF-8 persistence boundary.
function jsonValue(value: any, parents = new Set<any>()): any {
  if (value === null) return ["null"];
  if (typeof value === "boolean") return ["bool", value];
  if (typeof value === "string") return ["string", JSON.stringify(value)];
  if (typeof value === "bigint") { const decimal = String(value); need(decimal.replace(/^-/, "").length <= 4300, "JSON 检验 expected 无效"); return ["int", decimal]; }
  if (typeof value === "number" || value instanceof Number) {
    const number = Number(value); need(Number.isFinite(number), "JSON 检验 expected 无效");
    if (typeof value === "number" && Number.isSafeInteger(number)) return ["int", String(number)];
    return ["float", Object.is(number, -0) ? "-0" : String(number)];
  }
  need(Array.isArray(value) || object(value), "JSON 检验 expected 无效");
  need(!parents.has(value), "JSON 检验 expected 无效"); parents.add(value);
  try {
    return Array.isArray(value) ? ["array", value.map(item => jsonValue(item, parents))]
      : ["object", Object.keys(value).sort().map(key => [JSON.stringify(key), jsonValue(value[key], parents)])];
  } finally { parents.delete(value); }
}

export function validateGenerationCheck(input: any, defaultExpected: any = null): Record<string, any> {
  const check = input === null || input === undefined ? { type: "exact", expected: defaultExpected } : input;
  need(object(check) && typeof check.type === "string" && ["exact", "contains", "semantic"].includes(check.type), "生成检验类型无效");
  const result = { ...check }, maximum = Object.hasOwn(check, "max_new_tokens") ? check.max_new_tokens : 192, forbidden = Object.hasOwn(check, "forbidden") ? check.forbidden : [];
  need(integer(maximum) && maximum >= 1 && maximum <= 512, "核心生成检验预算须在 1–512 token 内");
  need(Array.isArray(forbidden) && forbidden.length <= 32 && forbidden.every(item => typeof item === "string" && pythonStrip(item) && length(item) <= 512), "核心生成禁用片段无效");
  result.max_new_tokens = maximum; result.forbidden = [...forbidden];
  if (check.type === "exact" || check.type === "contains") {
    need(typeof check.expected === "string" && pythonStrip(check.expected), "生成检验 expected 必须为非空字符串");
    result.expected = pythonStrip(check.expected); return result;
  }
  const validator = check.validator, expected = check.expected;
  need(typeof validator === "string" && validators.has(validator), "语义检验 validator 无效");
  if (validator === "goal_topics") need(typeof expected === "string" && Object.hasOwn(required, expected), "目标说明检验需要预先冻结的有效 topic");
  if (validator === "normalized_exact") need(typeof expected === "string" && pythonStrip(expected), "规范化精确检验需要非空字符串 expected");
  if (validator === "integer") need(integer(expected), "整数检验需要整数 expected");
  if (validator === "json") {
    // Python serializes None even before it checks that expected was provided.
    jsonValue(Object.hasOwn(check, "expected") ? expected : null);
    need(Object.hasOwn(check, "expected"), "JSON 检验需要 expected");
  }
  if (validator === "translation" || validator === "one_of") {
    need(Array.isArray(check.accepted) && check.accepted.length >= 1 && check.accepted.length <= 16
      && check.accepted.every((item: any) => typeof item === "string" && pythonStrip(item) && length(item) <= 256),
    validator === "translation" ? "翻译检验需要明确的 accepted 译文" : "候选精确检验需要明确的 accepted 文本");
    result.accepted = [...check.accepted];
  }
  return result;
}
export function isFiniteGenerationCheck(input: any) {
  const check = validateGenerationCheck(input); return check.type === "exact" || check.type === "semantic" && finite.has(check.validator);
}

export function evaluateGoalTopics(answer: any, topic: any, complete: any): boolean {
  need(typeof topic === "string" && Object.hasOwn(required, topic) && typeof answer === "string" && typeof complete === "boolean", "invalid frozen goal contract input");
  answer = pythonStrip(answer);
  if (!complete || length(answer) < 1 || length(answer) > 512 || !answer.endsWith("。")) return false;
  const parts: string[] = answer.slice(0, -1).split("。");
  if (parts.length < 1 || parts.length > 8 || new Set(parts).size !== parts.length || parts.some(part => !Object.hasOwn(sentences, part))) return false;
  const concepts = new Set(parts.flatMap(part => sentences[part]!)); return required[topic]!.every(concept => concepts.has(concept));
}

const unfence = (answer: string) => /^```(?:python|py|json)?[ \t]*\n([\s\S]*?)\n```$/i.exec(answer)?.[1] ?? answer;
function uniqueJSON(text: string) {
  const value = parseJSON(text), stack: { object: boolean; key: boolean; keys: Set<string> }[] = [];
  // Reuse native-checkpoint's framing/key scan after wire has validated grammar.
  for (let index = 0; index < text.length; index++) {
    const char = text[index], top = stack.at(-1);
    if (char === '"') {
      const start = index++;
      while (text[index] !== '"') { if (text[index] === "\\") index++; index++; }
      if (top?.object && top.key) {
        const key = JSON.parse(text.slice(start, index + 1)); need(!top.keys.has(key), "duplicate JSON key"); top.keys.add(key); top.key = false;
      }
    } else if (char === "{" || char === "[") stack.push({ object: char === "{", key: true, keys: new Set() });
    else if (char === "}" || char === "]") stack.pop();
    else if (char === "," && top?.object) top.key = true;
    else if (char === "-" || char !== undefined && /[0-9]/.test(char)) {
      const token = /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(index))![0];
      if (!/[.eE]/.test(token)) need(token.replace(/^-/, "").length <= 4300, "integer conversion limit");
      index += token.length - 1;
    }
  }
  return value;
}
function jsonCorrect(answer: string, expected: any) {
  try { return encoded(jsonValue(uniqueJSON(unfence(answer)))) === encoded(jsonValue(expected)); } catch { return false; }
}
function integerCorrect(answer: string, expected: number | bigint) {
  let sign = "", digits = "";
  for (const char of answer) {
    if (!digits && !sign && (char === "+" || char === "-")) { sign = char; continue; }
    const code = char.codePointAt(0)!, zero = unicode.decimalZeros.find(start => code >= start && code < start + 10);
    if (zero === undefined) return false; digits += String(code - zero);
  }
  // CPython 3.12's default decimal conversion limit counts even leading zeros.
  return digits.length > 0 && digits.length <= 4300 && BigInt(sign + digits) === BigInt(expected);
}

export function evaluateGeneration(inputAnswer: any, inputCheck: any, complete: any) {
  const check = validateGenerationCheck(inputCheck);
  need(typeof inputAnswer === "string" && typeof complete === "boolean", "生成文本与结束标记无效");
  const answer = pythonStrip(inputAnswer), kind = check.type, expected = check.expected, validator = check.validator;
  let correct: boolean;
  if (kind === "exact") correct = answer === expected;
  else if (kind === "contains") correct = answer.includes(expected);
  else if (validator === "integer") correct = integerCorrect(answer, expected);
  else if (validator === "json") correct = jsonCorrect(answer, expected);
  else if (validator === "goal_topics") correct = evaluateGoalTopics(answer, expected, complete);
  else if (validator === "normalized_exact") correct = pythonNFKC(answer) === pythonNFKC(pythonStrip(expected));
  else if (validator === "python_upper") correct = require("./python-upper").pythonUpper(answer);
  else if (validator === "one_of") correct = check.accepted.includes(answer);
  else if (validator === "translation") {
    const normalized = pythonCasefold(answer.replace(/^[ \t\r\n"'“”‘’。.!！]+|[ \t\r\n"'“”‘’。.!！]+$/g, ""));
    correct = check.accepted.some((value: string) => normalized === pythonCasefold(pythonStrip(value)));
  } else correct = !!answer;
  const normalized = pythonCasefold(pythonNFKC(answer));
  const found = check.forbidden.filter((fragment: string) => normalized.includes(pythonCasefold(pythonNFKC(fragment))));
  return { passed: complete && correct && !found.length, correct, complete, forbidden_found: found, check: kind,
    validator: validator ?? null, expected: Object.hasOwn(check, "accepted") ? check.accepted : expected ?? null };
}
