/** Python-parity effective source selection after physical identity, tensor and retention validation. */
import assert from "node:assert/strict";
import { ledgerParse } from "./memory.ts";
import { collectLearningModelFiles } from "../.native-migration/research/native-learning-product/model-files.ts";
import { readSafe, validateModelFiles, type ModelFiles, type SnapshotBase } from "../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts";
import { readNativeFoundationAdmission } from "../.native-migration/research/native-product-runtime/backend/native-foundation-admission.ts";


/** Trial adapters are usable for chat, never an accepted parent for formal training. */
export function nativeTrialSourceReason(store: string): string | null {
  const trial = (value: any) => value?.execution_scope === "local-trial" || value?.migration?.scope === "local-trial"
    || value?.quality_admitted === false || value?.training_admitted === false;
  for (const [name, prefix] of [["active.json", "layer-23/"], ["active-router.json", ""]] as const) {
    let pointer: any;
    try { pointer = ledgerParse(readSafe(store, name, 16 << 20).toString()); }
    catch (error: any) { if (error.code === "ENOENT") continue; throw error; }
    if (pointer.inference_engine === "mlx-q4") return "MLX专家须使用原生MLX学习和验证流程，不能交给旧精度训练";
    if (trial(pointer)) return "当前MLX专家组合仅获本地试用许可；恢复已正式采纳来源后可继续学习和无限进化";
    assert(typeof pointer.id === "string" && /^[a-zA-Z0-9_-]{1,80}$/.test(pointer.id), "训练来源版本无效");
    if (trial(ledgerParse(readSafe(store, prefix + pointer.id + ".json", 16 << 20).toString())))
      return "当前MLX专家组合仅获本地试用许可；恢复已正式采纳来源后可继续学习和无限进化";
  }
  return null;
}

export const CORE_PROTECTION_GATES = ["protected_core_retention_generation", "protected_core_retention_loss",
  "protected_replay_generation", "protected_replay_loss_retained"] as const;
export type NativeCoreReview = { version: string; qualified: false; reason: string; required_gates?: string[] };
const reviews = new WeakMap<object, NativeCoreReview>();

export function nativeCoreProtectionAdmission(metadata: Record<string, any>, report: Record<string, any> | null) {
  if (typeof metadata.report !== "string" || !/^evolution-[a-f0-9]{12}\.json$/.test(metadata.report))
    return { qualified: true as const, scope: "legacy metadata without bound report; compatibility fixture" };
  if (!report) return { qualified: false as const, reason: "核心采纳报告缺失或损坏" };
  if (CORE_PROTECTION_GATES.some(key => report.gates?.[key] !== true))
    return { qualified: false as const, reason: "核心报告没有当前逐条保护门", required_gates: [...CORE_PROTECTION_GATES] };
  if (report.core_retention_loss?.passed !== true || !Array.isArray(report.core_retention_generation))
    return { qualified: false as const, reason: "核心报告缺少逐条保护复验" };
  return { qualified: true as const, scope: "当前逐条核心保护、重放生成与损失协议" };
}

/** The build inserts this call only AFTER the existing full physical validator has succeeded. */
export function applyNativeEffectiveCore<T extends { core: any; expert: any; router: any }>(files: ModelFiles, physical: T): T {
  if (!physical.core) return physical;
  const meta = ledgerParse(files[physical.core.version + ".json"]!.toString());
  const report = files[meta.report] ? ledgerParse(files[meta.report]!.toString()) : null;
  const admission = nativeCoreProtectionAdmission(meta, report);
  if (admission.qualified) return physical;
  const state = { ...physical, core: null };
  reviews.set(state, Object.freeze({ version: physical.core.version, ...admission })); return state;
}
export function nativeEffectiveCoreReview(state: object): NativeCoreReview | null { return reviews.get(state) ?? null; }

/** Called after the existing private reload receipt validator, including rollback to an ignored physical core. */
export function assertNativeEffectiveCoreReload(pointer: any, receipt: { loaded: Record<string, any> }, state: { core: any }) {
  assert.equal(receipt.loaded.core_loaded, Boolean(state.core), "实际重载核心与有效来源不一致");
  if (state.core) {
    assert(pointer && state.core.version === pointer.version && state.core.file_sha256 === pointer.sha256);
    assert(receipt.loaded.core_file_sha256 === pointer.sha256 && receipt.loaded.core_tensor_sha256 === state.core.tensor_sha256);
  } else {
    if (pointer !== null) assert(nativeEffectiveCoreReview(state)?.version === pointer.version,
      "非空物理指针不能凭空声称已回退原始模型");
    assert(receipt.loaded.core_file_sha256 == null && receipt.loaded.core_tensor_sha256 == null);
  }
}

/** Startup/inspection uses the same physical validator as Engine, snapshots and retained-source reload. */
export function readNativeEffectiveCore(store: string, base: SnapshotBase) {
  const trialReason = nativeTrialSourceReason(store);
  assert(!trialReason, trialReason ?? "试用来源不能用于正式训练");
  const files = collectLearningModelFiles(store);
  const state = applyNativeEffectiveCore(files, validateModelFiles(files, base, readNativeFoundationAdmission(store, files)));
  const physical = files["active-core.json"] ? ledgerParse(files["active-core.json"]!.toString()) : null;
  const review = nativeEffectiveCoreReview(state);
  assert(!review || state.core === null && physical?.version === review.version);
  return { state, physical, core_requires_review: review };
}
