/** Private JSON transport preserves SQLite integers, Python floats and binary values. */
const floatSources = new WeakMap<Number, string>();
export function floatValue(value: number, source = String(value)): Number {
  if (!Number.isFinite(value)) throw new TypeError("数值需要是有限数");
  const result = new Number(value);
  floatSources.set(result, source);
  return result;
}
export function floatSource(value: Number): string {
  const source = floatSources.get(value);
  return source ?? (Number.isInteger(Number(value)) ? Number(value) + ".0" : String(value));
}
export function parseJSON(text: string): any {
  return JSON.parse(text, (_key: string, value: any, context?: { source?: string }) => {
    if (typeof value !== "number") return value;
    if (!Number.isFinite(value)) throw new TypeError("数值需要是有限数");
    const source = context?.source ?? String(value);
    if (/^-?\d+$/.test(source)) {
      const precise = BigInt(source);
      return precise <= BigInt(Number.MAX_SAFE_INTEGER) && precise >= BigInt(Number.MIN_SAFE_INTEGER) ? Number(precise) : precise;
    }
    return floatValue(value, source);
  });
}
export function encodeWire(value: any): any {
  if (typeof value === "bigint") return { $luban_int: String(value) };
  if (value instanceof Number) return { $luban_float: floatSource(value) };
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("数值需要是有限数");
    return Number.isSafeInteger(value) ? value : { $luban_float: String(value) };
  }
  if (value instanceof Uint8Array) return { $luban_bytes: Buffer.from(value).toString("base64") };
  if (Array.isArray(value)) return value.map(encodeWire);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, encodeWire(item)]));
  if (value === undefined) return null;
  return value;
}
export function decodeWire(value: any): any {
  if (value instanceof Number) return value;
  if (Array.isArray(value)) return value.map(decodeWire);
  if (value && typeof value === "object") {
    if (Object.keys(value).length === 1) {
      if (typeof value.$luban_int === "string" && /^-?\d+$/.test(value.$luban_int)) {
        const precise = BigInt(value.$luban_int);
        return precise <= BigInt(Number.MAX_SAFE_INTEGER) && precise >= BigInt(Number.MIN_SAFE_INTEGER) ? Number(precise) : precise;
      }
      if (typeof value.$luban_float === "string" && /^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value.$luban_float)) return floatValue(Number(value.$luban_float), value.$luban_float);
      if (typeof value.$luban_bytes === "string") {
        const decoded = Buffer.from(value.$luban_bytes, "base64");
        if (decoded.toString("base64") !== value.$luban_bytes) throw new TypeError("二进制传输格式无效");
        return new Uint8Array(decoded);
      }
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, decodeWire(item)]));
  }
  return value;
}
/** Public/persisted JSON uses numbers, rather than private transport tags. */
export function stringifyJSON(value: any): string {
  if (typeof value === "bigint") return String(value);
  if (value instanceof Number) return floatSource(value);
  if (Array.isArray(value)) return "[" + value.map(stringifyJSON).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.entries(value).filter(([, item]) => item !== undefined).map(([key, item]) => JSON.stringify(key) + ":" + stringifyJSON(item)).join(",") + "}";
  if (typeof value === "number" && !Number.isFinite(value)) throw new TypeError("数值需要是有限数");
  return JSON.stringify(value) ?? "null";
}
