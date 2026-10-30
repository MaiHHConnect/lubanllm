import { randomBytes } from "node:crypto";
import { chmodSync, closeSync, constants, existsSync, fchmodSync, fsyncSync, lstatSync, mkdirSync, openSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { isIP } from "node:net";
import { basename, dirname, isAbsolute, join, normalize } from "node:path";
import { AGREEMENT_VERSION, agreementStatus } from "./agreement.ts";
import { parseJSON, stringifyJSON } from "./wire.ts";

export const LEGACY_MODEL_PATH = "/Volumes/移动数据库/lubanllm/models/Qwen3-1.7B";
export const TEACHING_MODALITIES = ["text", "image", "video", "audio"] as const;
export const DEFAULTS = {
  model_path: "/Volumes/移动数据库/lubanllm/models/luban0",
  speech_model_path: "/Volumes/移动数据库/lubanllm/models/Qwen3-ASR-0.6B-luban0-v1",
  storage_root: "/Volumes/移动数据库/lubanllm", local_thinking: "off",
  max_new_tokens: 0, context_tokens: 262144, context_mode: "auto", temperature: 0.6, native_inference_precision: "bf16",
  native_q4_profile: "", native_mlx_model_path: "",
  native_skip_layer_mask: 0, native_exit_after: 24, native_layer_paging: false,
  agreement: { version: "", accepted_at: null as string | null },
  automatic: { mode: "manual", scheduled_time: "02:00", learning_enabled: true },
  teacher: { answer_mode: "local_first", name: "", base_url: "", model: "", thinking: "default", send_reasoning_effort: false,
    timeout_seconds: 120, session_call_limit: 0, teaching_modalities: ["text"] },
};
export type SettingsData = typeof DEFAULTS & Record<string, unknown>;
export type Credentials = Record<string, string>;
const PRESETS = [65536, 131072, 262144];
const object = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v);
const copy = <T>(v: T): T => structuredClone(v);
const token = () => randomBytes(32).toString("base64url");
const truthy = (v: unknown) => Array.isArray(v) ? v.length > 0 : object(v) ? Object.keys(v).length > 0 : Boolean(v);
function teachingModalities(value: unknown): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== "string" || !(TEACHING_MODALITIES as readonly string[]).includes(item))
      || new Set(value).size !== value.length) throw new Error("教师教学范围需要不重复的文字、图片、视频或音频列表");
  return TEACHING_MODALITIES.filter(item => value.includes(item));
}
function decimalDigits(value: string): string {
  return value.replace(/\p{Nd}/gu, digit => {
    const n = digit.codePointAt(0)!;
    let start = n;
    while (start > 0 && /\p{Nd}/u.test(String.fromCodePoint(start - 1))) start--;
    return String((n - start) % 10);
  });
}
export function defaultHome() { return process.platform === "darwin" ? join(homedir(), "Library/Application Support/Luban") : join(homedir(), ".local/share/luban"); }

export function integer(value: unknown): number {
  if (typeof value === "number" && Number.isSafeInteger(value)) return value;
  if (typeof value === "string") {
    // Python int permits Unicode decimal digits and single underscores between them.
    const digits = decimalDigits(value.trim());
    if (/^[+-]?\d(?:_?\d)*$/.test(digits)) {
      const parsed = Number(digits.replaceAll("_", ""));
      if (Number.isSafeInteger(parsed)) return parsed;
    }
  }
  throw new Error("需要整数");
}
export function validKey(value: unknown): value is string { return typeof value === "string" && /^[\x21-\x7e]+$/.test(value); }

function regularDestination(path: string) {
  if (!existsSync(path)) {
    // existsSync follows links, including broken links; lstat catches those too.
    try { if (lstatSync(path).isSymbolicLink()) throw new Error("私有文件不能使用符号链接"); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
    return;
  }
  if (!lstatSync(path).isFile()) throw new Error("私有文件路径需要普通文件");
}
function prepareJSON(path: string, data: unknown): string {
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  regularDestination(path);
  const temp = join(dirname(path), `.${basename(path)}.${randomBytes(8).toString("hex")}.tmp`);
  const fd = openSync(temp, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try { writeFileSync(fd, stringifyJSON(data) + "\n", "utf8"); fchmodSync(fd, 0o600); fsyncSync(fd); }
  catch (error) { unlinkSync(temp); throw error; }
  finally { closeSync(fd); }
  return temp;
}
export function privateJSON(path: string, data: unknown) {
  const temp = prepareJSON(path, data);
  try { regularDestination(path); renameSync(temp, path); }
  finally { if (existsSync(temp)) unlinkSync(temp); }
}

export function requireStorage(path: string): string {
  const root = normalize(path);
  if (root.startsWith("/Volumes/")) {
    const volume = join("/Volumes", root.split("/")[2]!);
    let mounted = false;
    try {
      const current = statSync(volume), parent = statSync("/Volumes");
      mounted = current.dev !== parent.dev || current.ino === parent.ino;
      // ExFAT/APFS mounted roots are distinct devices; an ordinary directory is not a mount.
    } catch {}
    if (!mounted) throw new Error("存储盘未连接，请连接后重试");
  }
  mkdirSync(root, { recursive: true });
  return root;
}

function teacherURL(value: string): string {
  const url = value.trim().replace(/\/+$/, "");
  if (!url) return url;
  // Match urllib.urlsplit validation, without WHATWG URL's repair or port restrictions.
  const m = /^([A-Za-z][A-Za-z0-9+.-]*):\/\/([^/?#]*)([^?#]*)(?:\?([^#]*))?(?:#(.*))?$/s.exec(url.replace(/[\t\r\n]/g, ""));
  const authority = m?.[2] || "", at = authority.lastIndexOf("@"), user = at < 0 ? "" : authority.slice(0, at);
  if (user.replace(":", "") || m?.[4] || m?.[5]) throw new Error("教师地址不能包含密码、查询参数或片段");
  if (!/^https?:/i.test(url)) throw new Error("教师地址需要使用 HTTP 或 HTTPS");
  let host = at < 0 ? authority : authority.slice(at + 1);
  if (host.includes("[") || host.includes("]")) {
    const bracket = /^\[([^\]]+)\]/.exec(host);
    if (!bracket || !(isIP(bracket[1]!.split("%", 1)[0]!) === 6 || /^v[0-9a-fA-F]+\..+$/.test(bracket[1]!))) throw new Error("教师地址缺少域名");
    host = bracket[1]!;
  } else host = host.split(":", 1)[0]!;
  if (!m || !host) throw new Error("教师地址缺少域名");
  return url;
}

export class Settings {
  readonly home: string;
  readonly path: string;
  readonly secret_path: string;
  private data: SettingsData;
  private credentials: Credentials;

  constructor(home = defaultHome()) {
    this.home = home;
    mkdirSync(home, { recursive: true, mode: 0o700 });
    if (!lstatSync(home).isDirectory()) throw new Error("私有设置目录需要普通目录");
    chmodSync(home, 0o700);
    this.path = join(home, "settings.json"); this.secret_path = join(home, "credentials.json");
    this.data = copy(DEFAULTS);
    if (!existsSync(this.path)) {
      const root = join(home, "data");
      Object.assign(this.data, { storage_root: root, model_path: join(root, "models/metadata"),
        speech_model_path: join(root, "models/speech"), native_mlx_model_path: join(root, "models/mlx"),
        native_inference_precision: "mlx-q4" });
    }
    regularDestination(this.path); regularDestination(this.secret_path);
    if (existsSync(this.path)) {
      const saved: unknown = parseJSON(readFileSync(this.path, "utf8"));
      if (!object(saved)) throw new Error("设置需要 JSON 对象");
      Object.assign(this.data, Object.fromEntries(Object.entries(saved).filter(([key]) => key !== "teacher")));
      this.data.automatic = { ...DEFAULTS.automatic };
      if (object(saved.automatic)) for (const key of Object.keys(DEFAULTS.automatic)) {
        if (key in saved.automatic) (this.data.automatic as Record<string, unknown>)[key] = saved.automatic[key];
      }
      if (typeof this.data.automatic.learning_enabled !== "boolean") this.data.automatic.learning_enabled = true;
      if (saved.teacher !== undefined && !object(saved.teacher)) throw new Error("教师设置需要 JSON 对象");
      if (object(saved.teacher)) for (const key of Object.keys(DEFAULTS.teacher)) {
        if (key in saved.teacher) (this.data.teacher as Record<string, unknown>)[key] = saved.teacher[key];
      }
      if (saved.model_path === LEGACY_MODEL_PATH && existsSync(join(DEFAULTS.model_path, "config.json")) && statSync(join(DEFAULTS.model_path, "config.json")).isFile()) this.data.model_path = DEFAULTS.model_path;
      if (!["auto", "manual"].includes(saved.context_mode as string) || typeof saved.context_tokens !== "number"
        || !PRESETS.includes(saved.context_tokens)) Object.assign(this.data, { context_mode: "auto", context_tokens: 262144 });
      else if (this.data.context_mode === "auto") this.data.context_tokens = 262144;
    }
    this.data.teacher.teaching_modalities = teachingModalities(this.data.teacher.teaching_modalities);
    if (existsSync(this.secret_path)) {
      const saved: unknown = parseJSON(readFileSync(this.secret_path, "utf8"));
      if (!object(saved) || Object.values(saved).some(v => typeof v !== "string")) throw new Error("凭据文件格式错误");
      this.credentials = saved as Credentials;
    } else this.credentials = { access_key: "lb-" + token(), ui_token: token(), teacher_key: "" };
    this.persist(this.data, this.credentials);
  }

  private persist(data: SettingsData, credentials: Credentials) {
    // Prepare both writes before replacing either file. Failed second replacement restores the first.
    const old = existsSync(this.path) ? readFileSync(this.path) : null;
    const first = prepareJSON(this.path, data);
    let second: string | undefined;
    let replaced = false;
    try {
      second = prepareJSON(this.secret_path, credentials);
      regularDestination(this.path); regularDestination(this.secret_path);
      renameSync(first, this.path); replaced = true;
      renameSync(second, this.secret_path);
    } catch (error) {
      if (replaced) {
        if (old) {
          const rollback = join(dirname(this.path), `.${basename(this.path)}.${randomBytes(8).toString("hex")}.tmp`);
          const fd = openSync(rollback, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
          try { writeFileSync(fd, old); fchmodSync(fd, 0o600); fsyncSync(fd); } finally { closeSync(fd); }
          try { renameSync(rollback, this.path); } finally { if (existsSync(rollback)) unlinkSync(rollback); }
        }
        else unlinkSync(this.path);
      }
      throw error;
    } finally {
      for (const path of [first, second]) if (path && existsSync(path)) unlinkSync(path);
    }
  }
  snapshot(): SettingsData { return copy(this.data); }
  public() { return { ...this.snapshot(), teacher: { ...this.data.teacher, key_configured: Boolean(this.secret("teacher_key")) }, access_key_configured: Boolean(this.secret("access_key")) }; }
  secret(name: string): string { return this.credentials[name] || ""; }
  agreement() { return agreementStatus(this.snapshot()); }
  accept_agreement() {
    if (!this.agreement().accepted) {
      const data = this.snapshot(); data.agreement = { version: AGREEMENT_VERSION, accepted_at: new Date().toISOString() };
      privateJSON(this.path, data); this.data = data;
    }
    return this.agreement();
  }
  acceptAgreement() { return this.accept_agreement(); }
  validated(changes: unknown): [SettingsData, Credentials] {
    if (!object(changes)) throw new Error("设置需要 JSON 对象");
    const data = this.snapshot(), credentials = { ...this.credentials };
    for (const key of ["model_path", "speech_model_path", "storage_root"] as const) if (key in changes) {
      if (typeof changes[key] !== "string" || !isAbsolute(changes[key])) throw new Error("模型和存储目录必须是绝对路径");
      data[key] = changes[key];
    }
    if ("local_thinking" in changes) {
      if (!["off", "low", "medium", "high"].includes(changes.local_thinking as string)) throw new Error("无效的思考程度");
      data.local_thinking = changes.local_thinking as string;
    }
    if ("native_inference_precision" in changes) {
      if (!["mlx-q4", "bf16", "mixed-q4-g32", "mixed-q4-g16", "mixed-q4-g16-packed", "mixed-q8"].includes(changes.native_inference_precision as string)) throw new Error("原生推理精度无效");
      data.native_inference_precision = changes.native_inference_precision as string;
      if (data.native_inference_precision !== "mixed-q4-g16-packed") data.native_q4_profile = "";
    }
    if ("native_mlx_model_path" in changes) {
      const value = changes.native_mlx_model_path;
      if (typeof value !== "string" || value && !isAbsolute(value)) throw new Error("MLX 模型目录必须是绝对路径");
      data.native_mlx_model_path = value;
    }
    if (data.native_inference_precision === "mlx-q4" && !data.native_mlx_model_path) throw new Error("MLX 推理需要指定模型目录");
    if ("native_q4_profile" in changes) {
      if (typeof changes.native_q4_profile !== "string") throw new Error("Q4 转换文件路径需要字符串");
      data.native_q4_profile = changes.native_q4_profile;
    }
    if ("native_inference_precision" in changes || "native_q4_profile" in changes) {
      if (data.native_inference_precision === "mixed-q4-g16-packed") {
        if (!isAbsolute(data.native_q4_profile) || !existsSync(data.native_q4_profile) || !statSync(data.native_q4_profile).isFile()) {
          throw new Error("Q4 转换文件必须是已存在文件的绝对路径");
        }
      } else if (data.native_q4_profile) throw new Error("Q4 转换文件仅用于 packed Q4 精度");
    }
    for (const key of ["native_skip_layer_mask", "native_exit_after"] as const) {
      if (key in changes) {
        const value = changes[key];
        if (typeof value !== "number" || !Number.isInteger(value) || (key === "native_skip_layer_mask"
          ? value < 0 || value >= 2 ** 24 || (value & 1) !== 0 : value < 1 || value > 24)) {
          throw new Error(key === "native_skip_layer_mask" ? "原生跳层掩码必须是24位非负整数且保留第0层" : "原生退出层数必须是1至24的整数");
        }
        data[key] = value;
      }
    }
    if ("native_layer_paging" in changes) {
      if (typeof changes.native_layer_paging !== "boolean") throw new Error("原生逐层读取必须是布尔值");
      data.native_layer_paging = changes.native_layer_paging;
    }
    if (["native_skip_layer_mask", "native_exit_after", "native_layer_paging"].some(key => key in changes)
      && data.native_skip_layer_mask >= 2 ** data.native_exit_after) throw new Error("原生跳层掩码不能包含退出位置及之后的层");
    if ("max_new_tokens" in changes) {
      const value = integer(changes.max_new_tokens);
      if (value !== 0 && (value < 16 || value > 32768)) throw new Error("max_new_tokens 超出范围 (16, 32768)，0 表示自动");
      data.max_new_tokens = value;
    }
    if ("context_mode" in changes) {
      if (!["auto", "manual"].includes(changes.context_mode as string)) throw new Error("上下文请选择自动或手动档");
      data.context_mode = changes.context_mode as string;
    }
    if ("context_tokens" in changes) {
      const value = integer(changes.context_tokens);
      if (!PRESETS.includes(value)) throw new Error("手动上下文请选择64K、128K或256K");
      data.context_tokens = value;
      if (!("context_mode" in changes)) data.context_mode = "manual";
    }
    if (data.context_mode === "auto") data.context_tokens = 262144;
    if ("temperature" in changes) {
      const input = changes.temperature;
      if (typeof input !== "number" && typeof input !== "string" && !(input instanceof Number)) throw new Error("温度需要数值");
      const normalized = typeof input === "string" ? decimalDigits(input.trim()) : input;
      if (typeof normalized === "string" && !/^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?$/.test(normalized)) throw new Error("温度需要数值");
      const value = Number(typeof normalized === "string" ? normalized.replaceAll("_", "") : normalized);
      if (!Number.isFinite(value) || value < 0 || value > 2) throw new Error("温度需要在 0 到 2 之间");
      data.temperature = value;
    }
    if ("automatic" in changes) {
      const change = changes.automatic;
      if (!object(change) || !["learning_enabled", "mode,scheduled_time", "learning_enabled,mode,scheduled_time"].includes(Object.keys(change).sort().join(","))) throw new Error("进化安排需要手动/自动及有效的24小时时间");
      const value = { ...data.automatic, ...change };
      if (typeof value.learning_enabled !== "boolean" || !["manual", "auto"].includes(value.mode as string)
        || typeof value.scheduled_time !== "string" || !/^(?:[01]\p{Nd}|2[0-3]):[0-5]\p{Nd}$/u.test(value.scheduled_time)) throw new Error("进化安排需要手动/自动及有效的24小时时间");
      data.automatic = { mode: value.mode as string, scheduled_time: value.scheduled_time, learning_enabled: value.learning_enabled };
    }
    if ("teacher" in changes) {
      const teacher = changes.teacher;
      if (!object(teacher)) throw new Error("教师设置需要 JSON 对象");
      for (const key of Object.keys(data.teacher)) if (key in teacher) (data.teacher as Record<string, unknown>)[key] = teacher[key];
      for (const key of ["name", "base_url", "model", "thinking"] as const) if (typeof data.teacher[key] !== "string") throw new Error(`教师 ${key} 需要文本`);
      if (/[\x00-\x1f\x7f-\x9f]/.test(data.teacher.name)) throw new Error("教师名称不能包含控制字符");
      data.teacher.name = data.teacher.name.trim();
      if ([...data.teacher.name].length > 60) throw new Error("教师名称不能超过60个字符");
      if (typeof data.teacher.send_reasoning_effort !== "boolean") throw new Error("教师 send_reasoning_effort 需要布尔值");
      if (!["local_first", "teacher_first", "auto"].includes(data.teacher.answer_mode)) throw new Error("教师回答方式无效");
      data.teacher.teaching_modalities = teachingModalities(data.teacher.teaching_modalities);
      data.teacher.base_url = teacherURL(data.teacher.base_url);
      if (!["default", "none", "minimal", "low", "medium", "high"].includes(data.teacher.thinking)) throw new Error("教师思考程度无效");
      const limit = integer(data.teacher.session_call_limit), timeout = integer(data.teacher.timeout_seconds);
      if (limit < 0 || limit > 1000) throw new Error("单次启动的教师调用上限应为0（不限）或1–1000");
      if (timeout < 5 || timeout > 600) throw new Error("教师超时应为 5–600 秒");
      data.teacher.session_call_limit = limit; data.teacher.timeout_seconds = timeout;
      if (truthy(teacher.api_key)) {
        if (!validKey(teacher.api_key)) throw new Error("教师密钥需要不含换行的文本");
        credentials.teacher_key = teacher.api_key;
      }
      if (truthy(teacher.clear_key)) credentials.teacher_key = "";
    }
    if ("access_key" in changes) {
      if (!validKey(changes.access_key) || changes.access_key.length < 16) throw new Error("本地接口密钥至少 16 个字符");
      credentials.access_key = changes.access_key;
    }
    return [data, credentials];
  }
  update(changes: unknown) { const [data, credentials] = this.validated(changes); this.persist(data, credentials); this.data = data; this.credentials = credentials; return this.public(); }
  rotate_key() { const key = "lb-" + token(); this.update({ access_key: key }); return key; }
  rotateKey() { return this.rotate_key(); }
  mask(value: unknown) { let text = String(value); for (const key of Object.values(this.credentials).filter(Boolean).sort((a, b) => b.length - a.length)) text = text.replaceAll(key, "[redacted]"); return text; }
}
export const private_json = privateJSON;
export const require_storage = requireStorage;
export const valid_key = validKey;
export const default_home = defaultHome;
