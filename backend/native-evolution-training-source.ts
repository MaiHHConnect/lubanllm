/** Training binds the effective core and saved expert identity; physical files still own CAS and rollback. */
import assert from "node:assert/strict";
import { NativeEngine, assertNativeExpertScopeGuard } from "../.native-migration/research/native-unified-expert/native-engine.ts";
import { borrowNativeEnginePeer } from "../.native-migration/research/native-unified-peer/engine-peer.ts";
import type { selectIndependentNativeEvolutionCohort } from "./native-evolution-cohort.ts";

type Guard = Parameters<Parameters<NativeEngine["withQuiescent"]>[0]>[0];
export type NativeEvolutionTrainingSource = Readonly<{ core_version: string | null; expert_id: string | null; physical_core_version: string | null }>;
type Request = { kind: string; scope: string; identity: string; store: string; core_version: string | null; expert_id: string | null;
  guard: object; evolutionSource?: NativeEvolutionTrainingSource };
const sources = new WeakMap<NativeEvolutionTrainingSource, { identity: string; store: string; guard: Guard; assertLive(): void }>();

export function nativeEvolutionTrainingSource(engine: NativeEngine, guard: Guard,
  cohort: ReturnType<typeof selectIndependentNativeEvolutionCohort>): NativeEvolutionTrainingSource {
  assertNativeExpertScopeGuard(engine, guard); cohort.assertUnchanged();
  const peer = borrowNativeEnginePeer(engine, guard);
  const core = peer.source.runtime.core_label ?? null, physical = cohort.physical_core?.version ?? null;
  assert.equal(core, cohort.core?.version ?? null, "训练有效核心与实际驻留来源不同");
  if (physical !== core) assert(core === null && cohort.core_requires_review?.version === physical,
    "物理核心只有经原保护门拒绝后才能按原始核心训练");
  const assertLive = () => {
    assertNativeExpertScopeGuard(engine, guard); cohort.assertUnchanged(); peer.worker.assertLive();
    assert(!peer.worker.isBranchEnabled, "核心训练与质量观察必须保持专家分支关闭");
  };
  assertLive();
  const cap = Object.freeze({ core_version: core, expert_id: peer.previous?.id ?? null, physical_core_version: physical });
  sources.set(cap, { identity: engine.config.identity, store: peer.store, guard, assertLive }); return cap;
}

export function assertNativeEvolutionTrainingSource(request: Request) {
  if (!request.evolutionSource) return;
  const held = sources.get(request.evolutionSource);
  assert(held && request.kind === "evolution" && request.scope === "product" && request.guard === held.guard
    && request.identity === held.identity && request.store === held.store, "有效训练来源必须来自同一真实原生轮次");
  held.assertLive();
  assert(request.core_version === request.evolutionSource.core_version && request.expert_id === request.evolutionSource.expert_id,
    "训练绑定必须保留有效核心与已保存专家身份，关闭分支不等于删除专家");
}

/** Used at original session creation; all original physical active_files hashes and full-row CAS remain unchanged. */
export function assertNativeTrainingPointerIdentity(request: Request, pointers: { core_version: unknown; expert_id: unknown }) {
  assertNativeEvolutionTrainingSource(request);
  assert.equal(pointers.core_version, request.evolutionSource ? request.evolutionSource.physical_core_version : request.core_version,
    "Native start physical Core pointer identity differs");
  assert.equal(pointers.expert_id, request.expert_id, "Native start saved expert pointer identity differs");
}
