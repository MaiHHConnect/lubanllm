/** Existing Python-style persisted JSON, shared by snapshots and validators. */
const object = (value: any): value is Record<string, any> => value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof Number);
function need(condition: unknown, message: string): asserts condition { if (!condition) throw new Error(message); }
const unicodeOrder = (a: string, b: string) => {
  const left = Array.from(a, x => x.codePointAt(0)!), right = Array.from(b, x => x.codePointAt(0)!);
  for (let index = 0; index < Math.min(left.length, right.length); index++) if (left[index] !== right[index]) return left[index]! - right[index]!;
  return left.length - right.length;
};
function quoted(value: string) {
  need(!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(value), "快照文本包含无效Unicode"); return JSON.stringify(value);
}
// Same normalization as native-core-resume.pythonFloat; JSON source spellings
// such as 1e0 must not change Python content bindings after parsing.
function pythonFloatSource(value: number) {
  need(Number.isFinite(value), "快照值无效");
  if (Object.is(value, -0)) return "-0.0";
  if (value === 0) return "0.0";
  const [digits, power] = value.toExponential().split("e"), exponent = Number(power);
  return exponent < -4 || exponent >= 16
    ? digits + "e" + (exponent < 0 ? "-" : "+") + String(Math.abs(exponent)).padStart(2, "0")
    : String(value) + (Number.isInteger(value) ? ".0" : "");
}
export function encoded(value: any, compact = false): string {
  if (Array.isArray(value)) return "[" + value.map(item => encoded(item, compact)).join(compact ? "," : ", ") + "]";
  if (object(value)) return "{" + Object.keys(value).sort(unicodeOrder).map(key => quoted(key) + (compact ? ":" : ": ") + encoded(value[key], compact)).join(compact ? "," : ", ") + "}";
  if (value instanceof Number || typeof value === "number" && !Number.isInteger(value)) return pythonFloatSource(Number(value));
  if (typeof value === "bigint") return String(value);
  if (typeof value === "string") return quoted(value);
  need(value !== undefined && (typeof value !== "number" || Number.isFinite(value)), "快照值无效");
  return JSON.stringify(value);
}
