/** Product FINAL staging/adoption. Call only after the training producer's finally completed. */
import assert from "node:assert/strict";
import { join } from "node:path";
import { NativeEngine, type TextRequest } from "../.native-migration/research/native-unified-expert/native-engine.ts";
import { loadCore450FinalCheckpoint } from "../.native-migration/research/native-core450-runner/checkpoint.ts";
import { NativeCore450Owner } from "../.native-migration/research/native-core450-owner-controller/owner.ts";
import { withNativeEvolutionMaterial, type NativeEvolutionMaterial } from "./native-evolution-cohort.ts";
import { observeCore450ProductQuality, authorizeCore450Publication, validateCore450QualityObservations,
  type NativeEvolutionQ4Evidence } from "./native-evolution-admission.ts";
import type { NativeEvolutionGraphQualification } from "./native-evolution-graph-qualification.ts";
import { prepareCore450CandidateArtifacts, reopenCore450CandidateArtifacts } from "./native-evolution-artifacts.ts";
import { publishCore450Candidate } from "./native-evolution-publication.ts";
import { uniqueNativeEvolutionRecords } from "./native-evolution-material.ts";

type LoadRequest = Parameters<typeof loadCore450FinalCheckpoint>[2];
export type NativeEvolutionFinalRequest = LoadRequest & { material: NativeEvolutionMaterial | null;
  producerResult: { previous_model_restored: true; rollback_scope_completed: true;
    checkpoint: { name: string; sha256: string }; expected: LoadRequest["expected"] } };
export type NativeEvolutionStagedFinal = {
  directory: string; checkpoint_sha256: string; candidate_tensor_sha256: string; previous_tensor_sha256: string;
  identity: string; manifestSHA: string; core: { file: string; tensor_key: string; sha256: string };
  baselineCore: { file: string; tensor_key: string; sha256: string } | null; records: Record<string, any>[];
};
function checkRequest(request: NativeEvolutionFinalRequest) {
  const result = request.producerResult;
  assert(result.previous_model_restored === true && result.rollback_scope_completed === true,
    "必须等待训练producer finally恢复结束，不能在onFinal里采纳");
  assert(result.expected.id === request.expected.id && result.expected.kind === "evolution");
  assert(result.checkpoint.sha256 === request.final_checkpoint.sha256 && result.checkpoint.name === request.final_checkpoint.name);
}

/** First stage: finish/close this BF16 Engine before starting any Q4 observer. Keeps the journal paused and pointer unchanged. */
export async function stageNativeEvolutionFinal(engine: NativeEngine, request: NativeEvolutionFinalRequest,
  owners: NativeCore450Owner): Promise<NativeEvolutionStagedFinal> {
  checkRequest(request);
  return withNativeEvolutionMaterial(request.memory, request.material, () => engine.withSourceQuiescent(async guard => {
    const checkpoint = await loadCore450FinalCheckpoint(engine, guard, request);
    owners.own(checkpoint, () => checkpoint.close());
    try {
      const quality = await observeCore450ProductQuality(engine, guard, checkpoint, request);
      const candidate = await prepareCore450CandidateArtifacts(engine, guard, quality);
      const state = validateCore450QualityObservations(engine, guard, quality), previous = state.cohort.core;
      return { directory: candidate.directory, checkpoint_sha256: checkpoint.approval.checkpoint_sha256,
        candidate_tensor_sha256: checkpoint.state.candidate.sha256, previous_tensor_sha256: checkpoint.learning_reference.sha256,
        identity: request.base.identity, manifestSHA: request.base.model_manifest_sha256,
        core: { file: join(candidate.directory, candidate.pointer.version + ".safetensors"), tensor_key: request.base.core_parameter,
          sha256: candidate.pointer.sha256 },
        baselineCore: previous ? { file: join(request.store, previous.version + ".safetensors"),
          tensor_key: request.base.core_parameter, sha256: previous.sha256 } : null,
        records: uniqueNativeEvolutionRecords(state.cohort.records.groups, state.cohort.records.protection,
          state.cohort.records.retention_regressions) };
    } finally { await owners.release(checkpoint); }
  }));
}

/** Second stage: fresh BF16 scope, after all serial Q4 observations and their owner cleanup succeeded. */
export async function admitNativeEvolutionFinal(engine: NativeEngine, request: NativeEvolutionFinalRequest & {
  staged: NativeEvolutionStagedFinal; graph: NativeEvolutionGraphQualification; q4: NativeEvolutionQ4Evidence;
  messages: TextRequest["messages"] }, owners: NativeCore450Owner) {
  checkRequest(request);
  return withNativeEvolutionMaterial(request.memory, request.material, () => engine.withSourceQuiescent(async (guard, control) => {
    const checkpoint = await loadCore450FinalCheckpoint(engine, guard, request);
    let checkpointClosed = false;
    owners.own(checkpoint, async () => { await checkpoint.close(); checkpointClosed = true; });
    try {
      assert(request.staged.identity === request.base.identity
        && request.staged.checkpoint_sha256 === checkpoint.approval.checkpoint_sha256
        && request.staged.candidate_tensor_sha256 === checkpoint.state.candidate.sha256
        && request.staged.previous_tensor_sha256 === checkpoint.learning_reference.sha256, "Q4候选不属于本轮完整FINAL");
      const quality = await observeCore450ProductQuality(engine, guard, checkpoint, request);
      const candidate = await reopenCore450CandidateArtifacts(engine, guard, quality, request.staged.directory);
      assert.equal(candidate.pointer.sha256, request.staged.core.sha256);
      authorizeCore450Publication(engine, guard, quality, candidate.pointer.sha256, request.graph, request.q4);
      return await publishCore450Candidate(engine, guard, control, candidate, owners, request.messages);
    } finally { if (!checkpointClosed) await owners.release(checkpoint); }
  }));
}
