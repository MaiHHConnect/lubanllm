/** Select already-resident down projections; no state export, RNG/router restore or source reads. */
import assert from "node:assert/strict";
import { NativeEngine, assertNativeExpertScopeGuard } from "../.native-migration/research/native-unified-expert/native-engine.ts";
import type { SnapshotGuard } from "../.native-migration/research/native-unified-expert/class-closure/native-product-runtime/backend/evolution-snapshots.ts";
import { borrowResidentNativeEnginePeer } from "../.native-migration/research/native-unified-peer/engine-peer.ts";
import { residentCoreObservation, type CoreCheckpoint } from "../.native-migration/research/native-unified-core-control/install.ts";
import { assertObservationMutationAllowed, withObservation, observationRowCheck } from "../.native-migration/research/native-observation-python-parity-v2/scope.ts";
import { assertEvolutionBranch } from "../.native-migration/research/native-observation-python-parity-v2/round.ts";

type Mode = "current" | "previous";
type Receipt = { pid: number; observation: Mode; core_tensor_sha256: string; step: number; optimizer_step: number };
type Selection = { guard: SnapshotGuard; cap: CoreCheckpoint; worker: object; pid: number; mode: Mode;
  current: ReturnType<typeof residentCoreObservation>; previous: Readonly<{ guard: SnapshotGuard; cap: CoreCheckpoint; pid: number; core_tensor_sha256: string }> };
const selections = new WeakMap<NativeEngine, Selection>();
function peerWithSelection(engine: NativeEngine, guard: SnapshotGuard) {
  const peer = borrowResidentNativeEnginePeer(engine, guard);
  const worker = peer.worker as typeof peer.worker & {
    readonly coreObservationMode: Mode;
    selectCoreObservation(mode: Mode, signal?: AbortSignal): Promise<Receipt>;
  };
  assert(typeof worker.selectCoreObservation === "function" && ["current", "previous"].includes(worker.coreObservationMode), "驻留执行器缺少原生观察选择能力");
  return { peer, worker };
}
export async function selectNativeCoreObservation(engine: NativeEngine, guard: SnapshotGuard, cap: CoreCheckpoint,
  mode: Mode, options: { signal?: AbortSignal } = {}) {
  assert(mode === "previous" || mode === "current"); assertObservationMutationAllowed(engine);
  assertNativeExpertScopeGuard(engine, guard); options.signal?.throwIfAborted();
  const { peer, worker } = peerWithSelection(engine, guard), pid = worker.pid;
  const current = residentCoreObservation(engine, guard, cap, "current");
  assert("receipt" in current && current.cap === cap && "learning_reference" in cap, "观察切换需要本轮实际安装的完整核心与原始引用");
  const previousSHA = cap.learning_reference.sha256, expectedSHA = mode === "previous" ? previousSHA : current.receipt.core_tensor_sha256;
  const expectedStep = mode === "previous" ? 0 : current.receipt.step;
  const optimizerStep = worker.status.optimizer_step;
  assert(Number.isSafeInteger(optimizerStep) && optimizerStep >= 0);
  const fast = async () => {
    assertNativeExpertScopeGuard(engine, guard); worker.assertLive(); assert.equal(worker.pid, pid);
    assert.equal(residentCoreObservation(engine, guard, cap, "current"), current); assertEvolutionBranch(engine, worker);
  };
  // The dedicated command scope bypasses Process.check's unrelated whole-source reads.
  // It authorizes only core-observation; normal sample scopes still prohibit switching.
  const receipt = await withObservation({ engine, guard, worker, cap, value: { selection_command: true, mode },
    owners: [peer.source, peer.build, peer.owner, peer.core, peer.previous, peer.routerOwner, current.owner].filter(Boolean),
    fast, signal: options.signal }, () => worker.selectCoreObservation(mode, options.signal));
  assert(receipt.pid === pid && receipt.observation === mode && receipt.core_tensor_sha256 === expectedSHA
    && receipt.step === expectedStep && receipt.optimizer_step === optimizerStep, "驻留观察选择回执与实际核心不一致");
  assert.equal(worker.coreObservationMode, mode); await fast();
  selections.set(engine, { guard, cap, worker, pid, mode, current,
    previous: Object.freeze({ guard, cap, pid, core_tensor_sha256: previousSHA }) });
  return Object.freeze({ ...receipt, actual: true, product_admission: false });
}
export async function withNativeCoreObservation<T>(engine: NativeEngine, guard: SnapshotGuard, cap: CoreCheckpoint,
  mode: Mode, body: () => Promise<T>, signal?: AbortSignal) {
  assertNativeExpertScopeGuard(engine, guard); const { peer, worker } = peerWithSelection(engine, guard), pid = worker.pid;
  let selected = selections.get(engine);
  // A later legitimate checkpoint installation invalidates the previous observation cache.
  if (selected && selected.cap !== cap && mode === "current" && worker.coreObservationMode === "current") { selections.delete(engine); selected = undefined; }
  const value = selected ? (mode === "previous" ? selected.previous : selected.current) : residentCoreObservation(engine, guard, cap, mode);
  if (selected) assert(selected.guard === guard && selected.cap === cap && selected.worker === worker && selected.mode === mode, "观察选择不属于当前驻留核心作用域");
  assert.equal(worker.coreObservationMode, mode);
  const fast = async () => {
    assertNativeExpertScopeGuard(engine, guard); worker.assertLive(); assert.equal(worker.pid, pid); assertEvolutionBranch(engine, worker);
    assert.equal(worker.coreObservationMode, mode);
    if (selected) { assert.equal(selections.get(engine), selected); assert.equal(residentCoreObservation(engine, guard, cap, "current"), selected.current); }
    else assert.equal(residentCoreObservation(engine, guard, cap, mode), value);
  };
  const installedOwner = selected && "owner" in selected.current ? selected.current.owner : "owner" in value ? value.owner : undefined;
  const owners = [peer.source, peer.build, peer.owner, peer.core, peer.previous, peer.routerOwner, installedOwner].filter(Boolean);
  return withObservation({ engine, guard, worker, cap, value, owners, fast, signal }, body);
}
export { observationRowCheck };
