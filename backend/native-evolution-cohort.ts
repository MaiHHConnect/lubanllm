/** One frozen material view per actual native round. No observer reads the ledger again. */
import assert from "node:assert/strict";
import { constants, closeSync, fstatSync, lstatSync, openSync, readFileSync, realpathSync } from "node:fs";
import { join, resolve } from "node:path";
import { ledgerParse, ledgerStringify } from "./memory";
import { encoded } from "./canonical-json";
import { collectNativeEvolutionMaterial, materialContentSHA, type NativeMaterialInput } from "./native-evolution-material";
import { nativeCoreProtectionAdmission, readNativeEffectiveCore } from "./native-effective-core.ts";

type MemorySource = { model_identity: string | null; samples(): Record<string, any>[]; connect<T>(read: (db: any) => T): T };
type Base = { identity: string; original_down_sha256: string };
export type NativeEvolutionMaterial = Pick<NativeMaterialInput, "samples" | "directedTeaching" | "retentionRegressions">;
type Scope = { active: boolean; material: NativeEvolutionMaterial | null; selection?: { store: string; base: string; cohort: ReturnType<typeof loadCohort> } };
const scopes = new WeakMap<object, Scope>();
const clone = <T>(v: T): T => ledgerParse(ledgerStringify(v));
function freeze<T>(v: T): T {
  if (v && typeof v === "object" && !Object.isFrozen(v)) { for (const item of Object.values(v)) freeze(item); Object.freeze(v); }
  return v;
}
/** The worker already owns its round lock; this scope freezes directed materials without changing Memory. */
export async function withNativeEvolutionMaterial<T>(memory: MemorySource, material: NativeEvolutionMaterial | null, consume: () => Promise<T>): Promise<T> {
  assert(!scopes.has(memory), "同一资料库已经有活动的原生进化材料作用域");
  const scope: Scope = { active: true, material: material === null ? null : freeze(clone(material)) };
  scopes.set(memory, scope);
  try { return await consume(); } finally { scope.active = false; scopes.delete(memory); }
}
function readJSON(store: string, name: string, optional = false): { raw: Buffer; value: any } | null {
  assert(/^(?:active-core\.json|original-core-protection\.json|core-[a-f0-9]{12}(?:\.retention)?\.json|evolution-[a-f0-9]{12}\.json)$/.test(name), "核心材料文件名无效");
  const path = join(store, name); let fd: number;
  try { fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW); }
  catch (error: any) { if (optional && error.code === "ENOENT") return null; throw error; }
  try {
    const before = fstatSync(fd); assert(before.isFile() && before.nlink === 1 && before.size > 0 && before.size <= (16 << 20), "核心材料文件类型或长度无效");
    const raw = readFileSync(fd), after = fstatSync(fd), named = lstatSync(path);
    assert(before.dev === named.dev && before.ino === named.ino && before.size === raw.length && before.size === after.size && before.mtimeMs === after.mtimeMs && before.ctimeMs === after.ctimeMs, "核心材料文件在读取时发生变化");
    return { raw, value: ledgerParse(raw.toString("utf8")) };
  } finally { closeSync(fd); }
}
function runs(memory: MemorySource, identity: string): Record<string, any>[] {
  return memory.connect(db => {
    if (!db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'").get()) return [];
    const rows = db.query("SELECT * FROM training_runs WHERE identity=? ORDER BY updated DESC,id DESC").all(identity) as any[];
    return rows.map(row => {
      assert(typeof row.body === "string" && Buffer.byteLength(row.body) <= (16 << 20), "进化经验账本正文无效"); const body = ledgerParse(row.body);
      assert(body && body.id === row.id && body.kind === row.kind && body.binding?.identity === row.identity, "进化经验账本身份不一致");
      return { ...body, state: row.state };
    });
  });
}
function loadCohort(memory: MemorySource, store: string, base: Base, scope: Scope) {
  assert(resolve(store) === store && realpathSync(store) === store && !lstatSync(store).isSymbolicLink(), "原生材料存储必须是无链接的实际目录");
  const physical = readJSON(store, "active-core.json", true)?.value ?? null; let core = physical;
  let core_requires_review: ReturnType<typeof readNativeEffectiveCore>["core_requires_review"] = null;
  let retention: any, originalProtection: any;
  // Reuse the weights/retention bindings already stored by the publication lifecycle.
  // Do not add hashes of pointer/metadata/original-baseline JSON files.
  const accepted_files_sha256: Record<string, string> = {};
  if (core) {
    assert(core.identity === base.identity && /^core-[a-f0-9]{12}$/.test(core.version) && /^[a-f0-9]{64}$/.test(core.sha256), "活动核心指针无效");
    const meta = readJSON(store, core.version + ".json")!.value, retained = readJSON(store, core.version + ".retention.json")!;
    assert(meta.identity === base.identity && meta.version === core.version && meta.sha256 === core.sha256, "活动核心保护快照绑定不一致");
    // Existing publication binding, checked once when freezing the round, never per observation.
    const retentionSHA = Bun.CryptoHasher.hash("sha256", retained.raw, "hex");
    assert(meta.retention_sha256 === retentionSHA, "活动核心保护快照与元数据不匹配"); retention = retained.value;
    assert(typeof meta.report === "string" && /^evolution-[a-f0-9]{12}\.json$/.test(meta.report), "活动核心缺少采纳报告");
    const report = readJSON(store, meta.report)!.value;
    if (!nativeCoreProtectionAdmission(meta, report).qualified) {
      // A bad identity/weight/retention must still throw. Only a fully validated physical legacy core may fall back.
      const checked = readNativeEffectiveCore(store, base as Parameters<typeof readNativeEffectiveCore>[1]);
      assert(checked.state.core === null && checked.core_requires_review, "旧核心回退缺少已验证的实际来源");
      core_requires_review = checked.core_requires_review; core = null; retention = undefined;
      originalProtection = readJSON(store, "original-core-protection.json")!.value;
    }
    accepted_files_sha256[physical.version + ".safetensors"] = physical.sha256;
    accepted_files_sha256[physical.version + ".retention.json"] = retentionSHA;
  } else originalProtection = readJSON(store, "original-core-protection.json")!.value;
  const samples = scope.material ? scope.material.samples : memory.samples();
  const selected = collectNativeEvolutionMaterial({ identity: base.identity, activeCore: core?.version ?? null,
    originalTensorSHA256: base.original_down_sha256, originalProtection, retention, samples,
    ...(scope.material ? { directedTeaching: scope.material.directedTeaching, retentionRegressions: scope.material.retentionRegressions } : { runs: runs(memory, base.identity) }) });
  const records = freeze(selected.records), excluded_validation: any[] = [];
  const assertUnchanged = () => assert(scope.active && scopes.get(memory) === scope && memory.model_identity === base.identity, "原生进化材料作用域已关闭或基座改变");
  assertUnchanged();
  return Object.freeze({ records, unspent_count: selected.unspent_count, records_sha256: materialContentSHA(records), samples_sha256: materialContentSHA(samples),
    accepted_files_sha256: freeze(accepted_files_sha256), core: freeze(clone(core)), physical_core: freeze(clone(physical)),
    core_requires_review: freeze(clone(core_requires_review)), excluded_validation: freeze(excluded_validation),
    exclusion_sha256: materialContentSHA(excluded_validation), assertUnchanged, training_started: false, quality_admitted: false });
}
export function selectIndependentNativeEvolutionCohort(memory: MemorySource, store: string, base: Base) {
  const scope = scopes.get(memory); assert(scope?.active, "原生进化材料必须处于独占整轮作用域");
  assert(memory.model_identity === base.identity && /^[a-f0-9]{64}$/.test(base.identity) && /^[a-f0-9]{64}$/.test(base.original_down_sha256), "进化资料库模型身份不一致");
  const binding = encoded(base);
  if (!scope.selection) scope.selection = { store, base: binding, cohort: loadCohort(memory, store, base, scope) };
  assert(scope.selection.store === store && scope.selection.base === binding, "整轮材料不能切换基座或存储目录");
  scope.selection.cohort.assertUnchanged(); return scope.selection.cohort;
}
