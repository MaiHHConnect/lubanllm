import { afterEach, describe, expect, spyOn, test } from "bun:test";
import * as fs from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { AGREEMENT_TEXT, AGREEMENT_VERSION, agreementStatus } from "../backend/agreement.ts";
import { DEFAULTS, Settings, integer, privateJSON, requireStorage, validKey } from "../backend/settings.ts";
import { parseJSON } from "../backend/wire.ts";

const roots: string[] = [];
const fixture = () => { const root = fs.mkdtempSync(join(tmpdir(), "luban-bun-settings-")); roots.push(root); return root; };
afterEach(() => { for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true }); });
const credentials = { access_key: "lb-offline-access-1234567890", ui_token: "offline-ui-1234567890", teacher_key: "offline-teacher-1234567890" };
function seeded() { const root = fixture(); privateJSON(join(root, "credentials.json"), credentials); privateJSON(join(root, "settings.json"), { model_path: DEFAULTS.model_path }); return new Settings(root); }
function oracle(input: unknown) {
  const run = Bun.spawnSync([resolve(".venv/bin/python"), "-c", `
import sys,json,tempfile,os
from pathlib import Path
from luban.settings import Settings, DEFAULTS
from luban.agreement import agreement_status,AGREEMENT_TEXT,AGREEMENT_VERSION
inputs=json.load(sys.stdin)
result=[]
for changes in inputs['changes']:
 with tempfile.TemporaryDirectory() as home:
  Path(home,'credentials.json').write_text(json.dumps(inputs['credentials']))
  Path(home,'settings.json').write_text(json.dumps({'model_path':DEFAULTS['model_path']}))
  s=Settings(home)
  try:
   data,credentials=s.validated(changes)
   result.append({'ok':True,'data':data,'credentials':credentials})
  except (ValueError,TypeError,OverflowError):result.append({'ok':False})
json.dump({'cases':result,'agreements':[agreement_status({'agreement':v}) for v in inputs['agreements']], 'version':AGREEMENT_VERSION,'text':AGREEMENT_TEXT},sys.stdout,ensure_ascii=False)
`], { env: { ...process.env, PYTHONPATH: resolve("src"), LUBAN_BUN_RPC_URL: "", LUBAN_BUN_RPC_TOKEN: "" }, stdin: Buffer.from(JSON.stringify(input)) });
  expect(run.exitCode).toBe(0);
  return JSON.parse(run.stdout.toString());
}

describe("Bun private settings", () => {
  test("Python validation, credentials and agreement text are equivalent for real request shapes", () => {
    const changes: unknown[] = [{}, { model_path: "/tmp/model", speech_model_path: "/tmp/asr", storage_root: "/tmp/资料" },
      { local_thinking: "high", temperature: "0.25", max_new_tokens: "2_048" },
      { automatic: { mode: "auto", scheduled_time: "23:59" } },
      { agreement: { version: AGREEMENT_VERSION, accepted_at: "2026-10-09T00:00:00+00:00" } },
      { teacher: { base_url: " http://teacher.invalid:8000/v1/// ", model: "offline", thinking: "minimal", session_call_limit: "20", timeout_seconds: "60", api_key: "new-offline-key", send_reasoning_effort: true } },
      { teacher: { api_key: "" } }, { teacher: { clear_key: true } },
      { teacher: { api_key: "new-key", clear_key: true } },
      { teacher: { training_allowed: true, terms_reference: "ignored" } },
      ...["http://[::1]:8080/v1", "http://host:bad", "http://host?", "http://host#", "http://@host", "http://host:99999", "http://[v1.test]", "http://bad host", "http://host\\path"].map(base_url => ({ teacher: { base_url } })),
      { teacher: { api_key: [], clear_key: [] } }, { teacher: { clear_key: {} } },
      { temperature: "１.５" }, { automatic: { mode: "manual", scheduled_time: "0１:0５" } },
      { access_key: "new-offline-key-12345678" }, null, [], "bad"];
    for (const max_new_tokens of [true, false, null, [], {}, 0, 15, 16, 2048, 2049, 32768, 32769, 16.5, "16", "  +１６  ", "16.0", "0x20", "", "1__6", "9007199254740993"]) changes.push({ max_new_tokens });
    for (const temperature of [true, false, null, [], {}, -0.1, 0, 2, 2.1, "0.5", ".6", "1_0e-1", "NaN", "Infinity", "0x1", ""]) changes.push({ temperature });
    for (const context_tokens of [65536, 131072, 262144, "65536", true, 4096, 64000]) changes.push({ context_tokens });
    for (const context_mode of ["auto", "manual", "bad", null, true]) changes.push({ context_mode, context_tokens: 65536 });
    for (const teacher of [null, [], true, { base_url: "ftp://teacher.invalid" }, { base_url: "http:///bad" }, { base_url: "https://user:pass@teacher.invalid" }, { base_url: "https://teacher.invalid/?a=1" }, { base_url: "https://teacher.invalid/#a" }, { thinking: "invalid" }, { session_call_limit: true }, { session_call_limit: 1001 }, { timeout_seconds: 4 }, { timeout_seconds: 601 }, { send_reasoning_effort: "false" }, { api_key: "bad\nkey" }, { api_key: "密钥" }]) changes.push({ teacher });
    for (const teaching_modalities of [[], ["audio", "text", "video", "image"], ["image", "audio"], null, "text", true, 1, {}, ["unknown"], ["TEXT"], ["text", "text"], [1], [null], [[]], [{}]]) changes.push({ teacher: { teaching_modalities } });
    changes.push({ teacher: { answer_mode: "auto" } });
    for (const answer_mode of ["", null, true, [], {}]) changes.push({ teacher: { answer_mode } });
    for (const name of ["", "  千问老师  ", "\ufeff \u3000老师\u00a0\ufeff", "🧑‍🏫".repeat(20), "师".repeat(60), "师".repeat(61), "\n老师", "老师\t", "老\x00师", "老\x7f师", "老\x85师", null, true, 12, [], {}]) changes.push({ teacher: { name } });
    for (const automatic of [{ mode: "auto", scheduled_time: "24:00" }, { mode: "manual", scheduled_time: "2:00" }, { mode: "bad", scheduled_time: "12:00" }, { mode: "auto", scheduled_time: "02:00", extra: true }, {}]) changes.push({ automatic });
    for (const learning_enabled of [true, false, 0, 1, "false", null, [], {}]) {
      changes.push({ automatic: { learning_enabled } }, { automatic: { mode: "manual", scheduled_time: "02:00", learning_enabled } });
    }
    for (const model_path of ["relative", null, true, 12]) changes.push({ model_path });
    changes.push({ native_inference_precision: "mlx-q4", native_mlx_model_path: "/tmp/mlx-model" },
      { native_inference_precision: "mlx-q4" }, { native_mlx_model_path: "relative" },
      { native_mlx_model_path: null }, { native_mlx_model_path: "" });
    const agreements = [null, {}, { version: "old", accepted_at: "2026-10-09T00:00:00+00:00" },
      ...["2026-10-09T00:00:00+00:00", "2026-10-09T00:00:00Z", "20261009T000000+0000", "2026-W41-5T00:00:00Z", "2026-10-09X00+00", "2026-10-09T00:00:00.123456+08:00", "2026-10-09T00:00:00", "invalid", "2026-02-30T00:00:00Z", "2026-10-09T24:00:00Z", "2026-10-09T00:00:00+24:00", "2026-10-09T00:0000Z", "2026-10-09T0000:00Z", "2026-10-09\n00Z", "2026-1009T00Z", "202610-09T00Z", "2026-W415T00Z", "2026W41-5T00Z", "2026-10-09T00+00:0000", "2026-10-09T00+0000:00", "2026-10-09T00+00:99", "2026-10-09T00+00:00.5", "9999-W52-7T00Z", null].map(accepted_at => ({ version: AGREEMENT_VERSION, accepted_at }))];
    const python = oracle({ changes, agreements, credentials });
    console.info(`Python settings differential: ${changes.length} request cases, ${agreements.length} agreement records`);
    expect(AGREEMENT_TEXT).toBe(python.text); expect(AGREEMENT_VERSION).toBe(python.version);
    const settings = seeded();
    expect(changes.map(change => { try { const [data, credentials] = settings.validated(change); return { ok: true, data, credentials }; } catch { return { ok: false }; } })).toEqual(python.cases);
    expect(agreements.map(agreement => agreementStatus({ agreement }))).toEqual(python.agreements);
    expect(settings.snapshot()).toEqual(DEFAULTS);
    expect(settings.public().teacher.session_call_limit).toBe(0);
  });

  test("fresh keys, detached snapshots, private permissions and reopen", () => {
    const root = fixture(), settings = new Settings(root);
    expect(settings.snapshot().storage_root).toBe(join(root, "data"));
    expect(settings.snapshot().model_path).toBe(join(root, "data/models/metadata"));
    expect(settings.snapshot().native_mlx_model_path).toBe(join(root, "data/models/mlx"));
    expect(settings.snapshot().native_inference_precision).toBe("mlx-q4");
    expect(settings.secret("access_key")).toMatch(/^lb-[A-Za-z0-9_-]{43}$/);
    expect(settings.secret("ui_token")).toHaveLength(43);
    expect(settings.secret("missing")).toBe("");
    const detached = settings.snapshot(); detached.teacher.model = "mutated";
    expect(settings.snapshot().teacher.model).toBe("");
    settings.update({ teacher: { model: "persisted", api_key: "offline-only" } });
    expect(JSON.stringify(new Settings(root).snapshot())).toBe(JSON.stringify(settings.snapshot()));
    expect(new Settings(root).secret("teacher_key")).toBe("offline-only");
    for (const path of [settings.path, settings.secret_path]) expect(fs.statSync(path).mode & 0o777).toBe(0o600);
    expect(fs.statSync(root).mode & 0o777).toBe(0o700);
  });

  test("draft validation never writes; invalid batches leave both files and memory unchanged", () => {
    const settings = seeded(), before = settings.snapshot(), disk = [settings.path, settings.secret_path].map(p => fs.readFileSync(p));
    expect(settings.validated({ teacher: { model: "draft", api_key: "draft-only-key" } })[0].teacher.model).toBe("draft");
    expect(() => settings.update({ temperature: 0.2, teacher: { api_key: "private-bad\nkey" } })).toThrow();
    expect(settings.snapshot()).toEqual(before); expect(settings.secret("teacher_key")).toBe(credentials.teacher_key);
    expect([settings.path, settings.secret_path].map(p => fs.readFileSync(p))).toEqual(disk);
  });

  test("optional teacher name persists and clears without replacing provider settings or key", () => {
    const settings = seeded();
    settings.update({ teacher: { base_url: "https://teacher.invalid/v1", model: "offline", name: "  千问老师  " } });
    expect(settings.public().teacher.name).toBe("千问老师");
    const reopened = new Settings(settings.home);
    expect(reopened.public().teacher.name).toBe("千问老师");
    reopened.update({ teacher: { name: "" } });
    expect(reopened.public().teacher.name).toBe("");
    expect(reopened.snapshot().teacher.base_url).toBe("https://teacher.invalid/v1");
    expect(reopened.snapshot().teacher.model).toBe("offline");
    expect(reopened.secret("teacher_key")).toBe(credentials.teacher_key);
    const saved = reopened.snapshot(); delete (saved.teacher as Partial<typeof saved.teacher>).name;
    privateJSON(reopened.path, saved);
    expect(new Settings(reopened.home).public().teacher.name).toBe("");
  });

  test("teaching scope defaults, canonicalizes, permits empty and rejects invalid saved config", () => {
    const settings = seeded();
    expect(settings.snapshot().teacher.teaching_modalities).toEqual(["text"]);
    const saved = settings.snapshot(); delete (saved.teacher as Partial<typeof saved.teacher>).teaching_modalities;
    privateJSON(settings.path, saved);
    expect(new Settings(settings.home).snapshot().teacher.teaching_modalities).toEqual(["text"]);
    settings.update({ teacher: { teaching_modalities: ["audio", "text", "image", "video"] } });
    expect(new Settings(settings.home).snapshot().teacher.teaching_modalities).toEqual(["text", "image", "video", "audio"]);
    settings.update({ teacher: { teaching_modalities: [] } });
    expect(new Settings(settings.home).snapshot().teacher.teaching_modalities).toEqual([]);
    for (const value of [null, "text", ["operation"], ["text", "text"], [true]]) {
      const before = fs.readFileSync(settings.path);
      expect(() => settings.update({ teacher: { teaching_modalities: value } })).toThrow();
      expect(fs.readFileSync(settings.path)).toEqual(before);
      privateJSON(settings.path, { teacher: { teaching_modalities: value } });
      const raw = fs.readFileSync(settings.path);
      expect(() => new Settings(settings.home)).toThrow();
      expect(fs.readFileSync(settings.path)).toEqual(raw);
      fs.writeFileSync(settings.path, before);
    }
  });

  test("second replacement I/O failure rolls back exact first-file bytes and all in-memory state", () => {
    const settings = seeded(), before = settings.snapshot();
    const raw = fs.readFileSync(settings.path).toString().replace('"temperature":0.6', '"temperature":0.600');
    fs.writeFileSync(settings.path, raw);
    const keyDisk = fs.readFileSync(settings.secret_path);
    const original = fs.renameSync;
    const mocked = spyOn(fs, "renameSync").mockImplementation((from, to) => {
      if (String(to) === settings.secret_path) throw new Error("offline disk");
      return original(from, to);
    });
    try { expect(() => settings.update({ temperature: 0.3, access_key: "new-offline-access-123456" })).toThrow("offline disk"); }
    finally { mocked.mockRestore(); }
    expect(fs.readFileSync(settings.path).toString()).toBe(raw);
    expect(fs.readFileSync(settings.secret_path)).toEqual(keyDisk);
    expect(settings.snapshot()).toEqual(before);
    expect(settings.secret("access_key")).toBe(credentials.access_key);
    expect(fs.readdirSync(settings.home).filter(p => p.endsWith(".tmp"))).toEqual([]);
  });

  test("existing and broken private file links are refused without modifying their target", () => {
    const root = fixture(), target = join(root, "outside.json"), home = join(root, "home");
    fs.mkdirSync(home); fs.writeFileSync(target, "original"); fs.symlinkSync(target, join(home, "credentials.json"));
    expect(() => new Settings(home)).toThrow("普通文件"); expect(fs.readFileSync(target, "utf8")).toBe("original");
    fs.unlinkSync(join(home, "credentials.json")); fs.symlinkSync(join(root, "missing"), join(home, "settings.json"));
    expect(() => new Settings(home)).toThrow("符号链接"); expect(fs.existsSync(join(root, "missing"))).toBe(false);
  });

  test("non-file settings, invalid JSON and malformed credentials never get silently replaced", () => {
    const root = fixture(), path = join(root, "settings.json");
    fs.mkdirSync(path); expect(() => new Settings(root)).toThrow(); fs.rmdirSync(path);
    fs.writeFileSync(path, "invalid JSON"); expect(() => new Settings(root)).toThrow(); expect(fs.readFileSync(path, "utf8")).toBe("invalid JSON");
    fs.unlinkSync(path); fs.writeFileSync(join(root, "credentials.json"), '{"access_key":true}');
    expect(() => new Settings(root)).toThrow("凭据文件格式错误");
  });

  test("unaccepted read stays read-only; dedicated acceptance is durable and idempotent", () => {
    const settings = seeded(), before = fs.readFileSync(settings.path);
    expect(settings.agreement().accepted).toBe(false); expect(fs.readFileSync(settings.path)).toEqual(before);
    settings.update({ agreement: { version: AGREEMENT_VERSION, accepted_at: "2026-10-09T00:00:00Z" } });
    expect(settings.agreement().accepted).toBe(false);
    expect(settings.accept_agreement().accepted).toBe(true);
    const accepted = settings.snapshot().agreement;
    expect(new Settings(settings.home).accept_agreement().accepted).toBe(true);
    settings.accept_agreement(); expect(settings.snapshot().agreement).toEqual(accepted);
  });

  test("agreement write failure cannot accept in-memory", () => {
    const settings = seeded(), before = fs.readFileSync(settings.path), original = fs.renameSync;
    const mocked = spyOn(fs, "renameSync").mockImplementation((from, to) => { if (String(to) === settings.path) throw new Error("offline"); return original(from, to); });
    try { expect(() => settings.accept_agreement()).toThrow("offline"); } finally { mocked.mockRestore(); }
    expect(settings.agreement().accepted).toBe(false); expect(fs.readFileSync(settings.path)).toEqual(before);
    expect(fs.readdirSync(settings.home).filter(p => p.endsWith(".tmp"))).toEqual([]);
  });

  test("old context and automatic settings migrate without copying provider rights fields", () => {
    const settings = seeded(), saved = settings.snapshot();
    saved.teacher = { ...saved.teacher, training_allowed: true, terms_reference: "old" } as typeof saved.teacher;
    saved.context_mode = "invalid"; saved.context_tokens = 4096;
    saved.automatic = { mode: "auto", scheduled_time: "03:21", obsolete: true } as typeof saved.automatic;
    privateJSON(settings.path, saved);
    const reopened = new Settings(settings.home);
    expect(reopened.snapshot().context_mode).toBe("auto"); expect(reopened.snapshot().context_tokens).toBe(262144);
    expect(reopened.snapshot().automatic).toEqual({ mode: "auto", scheduled_time: "03:21", learning_enabled: true });
    expect(reopened.snapshot().teacher).not.toHaveProperty("training_allowed");
    expect(reopened.secret("teacher_key")).toBe(credentials.teacher_key);
    reopened.update({ context_mode: "manual", context_tokens: 65536 });
    expect(new Settings(settings.home).snapshot().context_tokens).toBe(65536);
  });

  test("secret rotation persists, blank preserves, clear removes and masks all credentials", () => {
    const settings = seeded(); settings.update({ teacher: { api_key: "" } }); expect(settings.secret("teacher_key")).toBe(credentials.teacher_key);
    const key = settings.rotate_key(); expect(key).not.toBe(credentials.access_key); expect(new Settings(settings.home).secret("access_key")).toBe(key);
    const publicJSON = JSON.stringify(settings.public());
    for (const name of ["access_key", "ui_token", "teacher_key"]) expect(publicJSON).not.toContain(settings.secret(name));
    const secretLine = [key, settings.secret("ui_token"), settings.secret("teacher_key"), key].join(" ");
    expect(settings.mask(secretLine)).toBe("[redacted] [redacted] [redacted] [redacted]");
    settings.update({ teacher: { clear_key: true } }); expect(settings.secret("teacher_key")).toBe(""); expect(settings.public().teacher.key_configured).toBe(false);
  });

  test("privateJSON repairs permissive file modes, creates no predictable tmp and cleans serialization failures", () => {
    const root = fixture(), path = join(root, "private", "pointers.json"); privateJSON(path, { current: "old", extras: { custom: true } });
    fs.chmodSync(path, 0o666); privateJSON(path, { current: "new" }); expect(fs.statSync(path).mode & 0o777).toBe(0o600);
    const bytes = fs.readFileSync(path); expect(() => privateJSON(path, { bad: Infinity })).toThrow(); expect(fs.readFileSync(path)).toEqual(bytes);
    privateJSON(path, { large: 9007199254740993n }); expect(fs.readFileSync(path, "utf8")).toBe('{"large":9007199254740993}\n');
    const oldMask = process.umask(0o777);
    try { privateJSON(path, { protected: true }); } finally { process.umask(oldMask); }
    expect(fs.statSync(path).mode & 0o777).toBe(0o600);
    expect(fs.readdirSync(join(root, "private"))).toEqual(["pointers.json"]);
  });

  test("storage fixture works and disconnected volumes are rejected before mkdir", () => {
    const root = join(fixture(), "storage"); expect(requireStorage(root)).toBe(root); expect(fs.statSync(root).isDirectory()).toBe(true);
    const offline = `/Volumes/luban-test-unmounted-${process.pid}-${Date.now()}/lubanllm`;
    expect(() => requireStorage(offline)).toThrow("存储盘未连接"); expect(fs.existsSync(offline)).toBe(false);
    if (fs.existsSync("/Volumes/移动数据库")) expect(requireStorage("/Volumes/移动数据库/lubanllm")).toBe("/Volumes/移动数据库/lubanllm");
  });

  test("strict helpers reject booleans, fractions, unsafe numeric values and non-ASCII secrets", () => {
    expect(integer(" +１_６ ")).toBe(16); expect(integer("١٦")).toBe(16);
    for (const value of [true, false, null, 1.5, 9007199254740992, 1n, NaN, Infinity, "", "1.5"]) expect(() => integer(value)).toThrow();
    for (const key of ["", "space key", "new\nline", "密钥", null, true]) expect(validKey(key)).toBe(false);
    expect(validKey("ascii-private-key")).toBe(true);
    for (const value of [NaN, Infinity, -Infinity]) expect(() => seeded().update({ temperature: value })).toThrow();
  });

  test("native inference precision defaults to BF16 and accepts the shipped quantized modes", () => {
    const settings = seeded();
    expect(settings.snapshot().native_inference_precision).toBe("bf16");
    for (const mode of ["mixed-q4-g32", "mixed-q4-g16", "mixed-q8"]) {
      expect(settings.update({ native_inference_precision: mode }).native_inference_precision).toBe(mode);
    }
    expect(() => settings.update({ native_inference_precision: "mixed-q4" })).toThrow("原生推理精度无效");
  });

  test("packed Q4 requires an existing absolute file and clears its profile when changing precision", () => {
    const settings = seeded(), profile = join(settings.home, "profile.bin");
    fs.writeFileSync(profile, "fixture");
    for (const invalid of ["", "relative.bin", settings.home, join(settings.home, "missing.bin"), null]) {
      expect(() => settings.update({ native_inference_precision: "mixed-q4-g16-packed", native_q4_profile: invalid })).toThrow();
    }
    expect(settings.snapshot().native_inference_precision).toBe("bf16");
    expect(() => settings.update({ native_q4_profile: profile })).toThrow("仅用于 packed Q4");
    settings.update({ native_inference_precision: "mixed-q4-g16-packed", native_q4_profile: profile });
    expect(new Settings(settings.home).snapshot().native_q4_profile).toBe(profile);
    settings.update({ native_inference_precision: "bf16" });
    expect(settings.snapshot().native_q4_profile).toBe("");
    expect(() => settings.update({ native_inference_precision: "mixed-q4-g16-packed" })).toThrow();
  });

  test("native execution controls remain opt in and strictly typed", () => {
    const settings = seeded();
    expect([settings.snapshot().native_skip_layer_mask, settings.snapshot().native_exit_after, settings.snapshot().native_layer_paging]).toEqual([0, 24, false]);
    for (const [key, values] of Object.entries({ native_skip_layer_mask: [true, "2", 2.5, -1, 1, 2 ** 24, 2 ** 24 - 1],
      native_exit_after: [true, "12", 12.5, 0, 25], native_layer_paging: [null, "true", 0, 1] })) {
      for (const value of values) expect(() => settings.update({ [key]: value })).toThrow();
    }
    const policy = { native_skip_layer_mask: 2, native_exit_after: 12, native_layer_paging: true };
    settings.update(policy);
    expect(new Settings(settings.home).snapshot()).toMatchObject(policy);
    for (const changes of [{ native_skip_layer_mask: 2 ** 12 }, { native_exit_after: 1 },
      { native_skip_layer_mask: 2 ** 23, native_exit_after: 23 }]) {
      expect(() => settings.update(changes)).toThrow("退出位置");
    }
    expect(settings.snapshot().native_skip_layer_mask).toBe(2);
    for (const key of ["native_skip_layer_mask", "native_exit_after"]) {
      expect(() => settings.validated(parseJSON(`{"${key}":2.0}`))).toThrow();
    }
  });

  test("JSON lexical float and large integers retain Python strict validation", () => {
    const settings = seeded();
    expect(settings.validated(parseJSON('{"max_new_tokens":16}'))[0].max_new_tokens).toBe(16);
    for (const raw of ['{"max_new_tokens":16.0}', '{"max_new_tokens":1.6e1}', '{"max_new_tokens":9007199254740993}', '{"teacher":{"timeout_seconds":5.0}}']) expect(() => settings.validated(parseJSON(raw))).toThrow("需要整数");
    expect(settings.validated(parseJSON('{"temperature":1.0}'))[0].temperature).toBe(1);
    const raw = JSON.stringify({ ...settings.snapshot(), context_mode: "manual", context_tokens: 65536 }).replace('"context_tokens":65536', '"context_tokens":65536.0');
    fs.writeFileSync(settings.path, raw);
    expect(new Settings(settings.home).snapshot().context_mode).toBe("auto");
    expect(new Settings(settings.home).snapshot().context_tokens).toBe(262144);
  });
});
