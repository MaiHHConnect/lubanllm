/** Actual resident final observations with the current twelve gates. Arithmetic cannot publish. */
import assert from "node:assert/strict";
import { borrowNativeEvolutionPreparation, type NativeEvolutionPrepared } from "../.native-migration/research/native-evolution-producer/producer.ts";
import type { NativeEvolutionAmbient } from "./native-evolution-ambient.ts";
import { type NativeRouterEngineSnapshot } from "../.native-migration/research/native-unified-peer/engine-peer.ts";
import { NativeCore450Owner } from "../.native-migration/research/native-core450-owner-controller/owner.ts";
import { core450ControlMetadata } from "../.native-migration/research/native-core450-owner-controller/metadata.ts";
import { NativeEngine, assertNativeExpertScopeGuard } from "../.native-migration/research/native-unified-expert/native-engine.ts";
import { type Core450Checkpoint } from "../.native-migration/research/native-core450-runner/checkpoint.ts";
import { validateCore450TrainingHistory } from "../.native-migration/research/native-core450-runner/training-safety.ts";
import { coreCheckpointField as field, coreContentSHA } from "../.native-migration/research/native-product-runtime/backend/native-core-resume.ts";
import { checkpointApprovedBinding, equalCheckpointValues } from "../.native-migration/research/native-unified-service/checkpoint-owner.ts";
import { floatValue } from "../.native-migration/research/native-product-runtime/backend/wire.ts";
import { requireCore450PeerOperations, generateCore450Rows, meanCore450Loss } from "../.native-migration/research/native-core450-owner-controller/runner.ts";
import { saveNativeTrainingCheckpoint, type NativeCheckpointSaveRequest } from "../.native-migration/research/native-learning-product/checkpoint-save-product.ts";
import { equal } from "../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts";
import { selectIndependentNativeEvolutionCohort } from "./native-evolution-cohort.ts";
import { uniqueNativeEvolutionRecords } from "./native-evolution-material.ts";
import { finishNativeEvolutionQuality, nativeEvolutionProgress } from "./native-evolution-quality.ts";
import { selectNativeCoreObservation } from "./native-evolution-observation.ts";
import { retentionLossRecords } from "./native-evolution-loss-targets.ts";

type Guard = Parameters<Parameters<NativeEngine["withQuiescent"]>[0]>[0];
type Control = Parameters<Parameters<NativeEngine["withQuiescent"]>[0]>[1];
type Save = Omit<NativeCheckpointSaveRequest, "snapshot" | "progress" | "guard">;

export async function observeCore450ProductFinal(engine: NativeEngine, guard: Guard, actualControl: Control,
  cap: Core450Checkpoint, previous: NativeEvolutionAmbient, r: Save, owners: NativeCore450Owner, product: NativeEvolutionPrepared) {
  assertNativeExpertScopeGuard(engine, guard);
  assert(cap.approval.step === 450 && r.expected.kind === "evolution" && r.expected.id === cap.approval.id, "FINAL checkpoint step/kind/run differs");
  assert(equalCheckpointValues(cap.checkpoint.binding, checkpointApprovedBinding(r.expected.binding)), "FINAL checkpoint typed binding differs");
  const control = requireCore450PeerOperations(actualControl), cohort = selectIndependentNativeEvolutionCohort(r.memory, r.store, engine.config.aten!.base);
  const frozen = cohort.records, records = uniqueNativeEvolutionRecords(frozen.groups, frozen.protection, frozen.retention_regressions);
  const lossMaterials = uniqueNativeEvolutionRecords({}, frozen.protection, frozen.retention_regressions);
  let lossRecords!: ReturnType<typeof retentionLossRecords>;
  const saved = core450ControlMetadata(cap.checkpoint.payload), report = saved.report;
  assert.equal(cohort.records_sha256, cap.binding.records_sha256);
  borrowNativeEvolutionPreparation(engine, guard, product); assert(equal(frozen, product.records));
  assert(records.length > 0); const safety = validateCore450TrainingHistory(saved, frozen.groups);
  assert(safety.complete_training_safety && safety.completed_steps === 450, "完整450逐次训练安全检查未通过");
  assert.equal(coreContentSHA(records), report.partial_records_sha256);
  const cached = new Map<string, any>();
  for (const row of [...report.partial_training_baseline, ...report.baseline_training_replay_generation, ...(report.baseline_core_retention_generation ?? [])]) {
    const prior = cached.get(row.id); assert(!prior || equal(prior, row), "Conflicting original baseline observations"); cached.set(row.id, row);
  }
  const deferred = records.filter(row => !cached.has(row.id));
  let baselineReplay!: number, baselineNew: number | null = null, baselineCoreLoss!: { id: string; loss: number }[];
  // The resident executor selects its existing accepted/candidate down references only.
  const losses = async (observation: "previous" | "current") => {
    const rows: { id: string; loss: number }[] = [];
    // Reuse the existing tokenization, PID/SHA/step, finite-loss and cancellation checks.
    // Its Python-parity observation scope never snapshots or rereads model/ledger files.
    for (const row of lossRecords) { r.signal?.throwIfAborted(); rows.push({ id: row.id, loss: await meanCore450Loss(control, [row], cap, r.signal, observation) }); }
    return rows;
  };
  let baselineFailure: unknown;
  try {
    await selectNativeCoreObservation(engine, guard, cap, "previous", { signal: r.signal });
    for (const row of await generateCore450Rows(control, deferred, cap, r.signal, "previous")) cached.set(row.id, row);
    lossRecords = retentionLossRecords(lossMaterials, [...cached.values()]);
    baselineReplay = await meanCore450Loss(control, frozen.groups.replay_validation, cap, r.signal, "previous");
    baselineNew = frozen.groups.new_validation.length ? await meanCore450Loss(control, frozen.groups.new_validation, cap, r.signal, "previous") : null;
    baselineCoreLoss = await losses("previous");
  } catch (error) { baselineFailure = error; throw error; }
  finally {
    // Restoration must run even when the user cancelled the observation signal.
    try { await selectNativeCoreObservation(engine, guard, cap, "current"); }
    catch (error) { throw baselineFailure === undefined ? error : new AggregateError([baselineFailure, error], "基线观察失败且驻留候选选择未恢复", { cause: baselineFailure }); }
  }
  const baseline = records.map(row => { const value = cached.get(row.id); assert(value); return value; });
  const first = await generateCore450Rows(control, records, cap, r.signal), second = await generateCore450Rows(control, records, cap, r.signal);
  const validationReplay = await meanCore450Loss(control, frozen.groups.replay_validation, cap, r.signal);
  const validationNew = frozen.groups.new_validation.length ? await meanCore450Loss(control, frozen.groups.new_validation, cap, r.signal) : null;
  const candidateCoreLoss = await losses("current");
  const binding = { core_version: "core-" + cap.approval.id, core_sha256: cap.state.candidate.sha256, expert_id: null, expert_sha256: null } as const;
  const quality = finishNativeEvolutionQuality({ records: frozen, baseline, first, second,
    baselineCoreGeneration: frozen.protection.map(row => { const value = cached.get(row.id); assert(value); return value; }),
    baselineCoreLoss, candidateCoreLoss, baselineReplayLoss: baselineReplay, candidateReplayLoss: validationReplay,
    completedSteps: safety.completed_steps, trainingSafetyFailed: safety.partial_safety_failed, identity: cap.binding.identity,
    binding, previousCoreSHA: cap.learning_reference.sha256, frozenRecordsSHA: report.partial_records_sha256 });
  const progress = nativeEvolutionProgress(records, baseline, first, second, binding, false);
  const evidence = { ...quality, promoted: false, learning_progress: progress, arithmetic_gates_passed: quality.promoted,
    training_safety: safety, source_authority: false, quality_admitted: false, product_admission: false, adopted: false };
  // Quality's general policy label must not replace the checkpoint's fixed-450 TRAIN contract.
  const finalReport = { ...report, ...quality, partial_policy: report.partial_policy, learning_progress: progress,
    core_retention_loss_targets: lossRecords.map(({ id, answer }) => ({ id, answer })),
    baseline_replay_loss: floatValue(baselineReplay), baseline_new_loss: baselineNew === null ? null : floatValue(baselineNew),
    validation_replay_loss: floatValue(validationReplay), validation_new_loss: validationNew === null ? null : floatValue(validationNew),
    quality_checked: false, promoted: false, product_admission: false, adopted: false, arithmetic_gates_passed: quality.promoted,
    actual_same_peer_observations: true, quality_authority: "awaiting actual production graph and source qualification; arithmetic gates cannot publish" };
  const tree = structuredClone(cap.checkpoint.tree), payload = field(tree, "payload")!;
  const typed = field(checkpointApprovedBinding({ ...cap.binding, arguments: finalReport }), "arguments")!;
  assert(payload.kind === "dict"); payload.entries = payload.entries.map(([k, v]) => [k, k.kind === "string" && k.value === "report" ? typed : v]);
  cohort.assertUnchanged(); assertNativeExpertScopeGuard(engine, guard);
  const complete = await saveNativeTrainingCheckpoint({ ...r, guard, snapshot: { tree, files: cap.checkpoint.files,
    async assertUnchanged() { assertNativeExpertScopeGuard(engine, guard); cohort.assertUnchanged(); await cap.assertUnchanged(); }
  }, progress: { step: 450, steps: 450, phase: "awaiting_quality" } });
  owners.own(complete, () => complete.close());
  try {
    assert(!complete.activated && !complete.quality_checked && !complete.rng_applied);
    return { checkpoint: complete.descriptor, previous_checkpoint: complete.previous_checkpoint, writer: complete.execution,
      report: finalReport, evidence, final_training_state_retained: true, quality_admitted: false, product_admission: false, adopted: false };
  } finally { await owners.release(complete); }
}

/** Historical fixed-data resume cannot claim the current dynamic material authority. */
export async function observeCore450Final(_engine: NativeEngine, _guard: Guard, _control: Control, _cap: Core450Checkpoint,
  _previous: NativeRouterEngineSnapshot, _request: Save, _owners: NativeCore450Owner): Promise<Awaited<ReturnType<typeof observeCore450ProductFinal>>> {
  throw new Error("当前产品原生质量检查需要本轮动态材料准备，不能沿用旧固定资料入口");
}
