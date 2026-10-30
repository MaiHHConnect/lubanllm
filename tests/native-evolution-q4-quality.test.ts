import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { evaluateGeneration } from "../backend/generation-checks.ts";
import { uniqueNativeEvolutionRecords, provenance, collectNativeEvolutionMaterial, materialContentSHA, validateNativeProductRecords } from "../backend/native-evolution-material.ts";
import { evolutionContentSHA, finishNativeEvolutionQuality } from "../backend/native-evolution-quality.ts";
import { finishNativeQ4Quality } from "../backend/native-evolution-q4-quality.ts";

function python(code: string, input?: unknown) {
  const result = spawnSync(resolve(".venv/bin/python"), ["-c", code], { cwd: process.cwd(),
    env: { ...process.env, PYTHONPATH: "src:." }, input: JSON.stringify(input), encoding: "utf8", timeout: 30000 });
  expect(result.status, result.stderr).toBe(0); return JSON.parse(result.stdout);
}
const material = python("import json;from scripts.current_q4_identity_material import records;print(json.dumps(records(),ensure_ascii=False))");
function fixture() {
  const records = structuredClone(material);
  records.protection = [{ id: "original-protected", prompt: "original", answer: "ok", source: "human", purpose: "validation",
    approved: true, training_allowed: false, provenance: { evolution_role: "replay", retention_generation_only: true, retention_scope: "original_core", original_core_baseline: true,
      generation_check: { type: "exact", expected: "ok" } } }];
  const all = uniqueNativeEvolutionRecords(records.groups, records.protection, records.retention_regressions);
  const gained = records.groups.new_training[0].id, weak = records.groups.new_validation[0].id, weakTrain = records.groups.new_training[1].id;
  const observations = (bad: string[]) => all.map(row => {
    const answer = bad.includes(row.id) ? "尚未学会" : row.answer;
    return { id: row.id, answer, passed: evaluateGeneration(answer, provenance(row).generation_check, true).passed,
      complete: true, branch_enabled: false as const, reference_used: false as const };
  });
  const losses = uniqueNativeEvolutionRecords({}, records.protection, records.retention_regressions).map(row => ({ id: row.id, loss: .5 }));
  const binding = { core_version: "core-123456789abc", core_sha256: "a".repeat(64), expert_id: null, expert_sha256: null };
  const training = finishNativeEvolutionQuality({ records, baseline: observations([gained]), first: observations([]), second: observations([]),
    baselineCoreGeneration: observations([]).filter(row => row.id === "original-protected"), baselineCoreLoss: losses, candidateCoreLoss: losses,
    baselineReplayLoss: .5, candidateReplayLoss: .5, completedSteps: 450, trainingSafetyFailed: false, identity: "c".repeat(64),
    binding, previousCoreSHA: "b".repeat(64), frozenRecordsSHA: evolutionContentSHA(all) });
  const q4 = { baseline: [{ rows: observations([gained, weak, weakTrain]) }, { rows: observations([gained, weak, weakTrain]) }],
    candidate: [{ rows: observations([weak, weakTrain]) }, { rows: observations([weak, weakTrain]) }] } as any;
  return { records, training, q4, weak, weakTrain, gained, binding: { ...binding, core_sha256: "d".repeat(64) } };
}

test("BF16-known/Q4-unknown NEW stays pending; Q4 report, proof binding and retention pass the current Python validator", () => {
  const { records, training, q4, weak, gained, binding } = fixture();
  expect(training.promoted).toBe(true); expect(training.learning_progress.mastered_ids).toContain(weak);
  const report = finishNativeQ4Quality(records, training, q4, binding, "b".repeat(64));
  expect(report.promoted).toBe(true); expect(report.training_precision_evidence).toBe(training);
  expect(report.learning_progress.pending_ids).toContain(weak);
  expect(report.learning_progress.learned_ids).toContain(gained);
  expect(report.accepted_retention_groups.new_validation.some((row: any) => row.id === weak)).toBe(false);
  expect(report.learning_progress.mastery_proofs.every((proof: any) => proof.binding.core_sha256 === binding.core_sha256)).toBe(true);
  const checked = python(`import json,sys
from luban.evolution_progress import validate_partial_progress
r=json.load(sys.stdin);print(json.dumps(validate_partial_progress(r['report'],r['records'],r['report']['identity'],r['version'],r['sha']),ensure_ascii=False))`,
  { report, records, version: binding.core_version, sha: binding.core_sha256 });
  expect(checked).toEqual(report.learning_progress);
});

test("either actual Q4 baseline protects already-known answers and failed Q4 observations cannot publish", () => {
  const f = fixture(), i = f.q4.baseline[1].rows.findIndex((row: any) => row.id === f.weak);
  f.q4.baseline[1].rows[i] = { ...f.q4.baseline[1].rows[i], answer: f.records.groups.new_validation[0].answer, passed: true };
  expect(() => finishNativeQ4Quality(f.records, f.training, f.q4, f.binding, "b".repeat(64))).toThrow();
  const noGrowth = fixture(); noGrowth.q4.candidate = noGrowth.q4.baseline;
  expect(() => finishNativeQ4Quality(noGrowth.records, noGrowth.training, noGrowth.q4, noGrowth.binding, "b".repeat(64))).toThrow();
});

test("partial Q4 acceptance constructs the next directed round from pending NEW and inherited authorized replay", () => {
  const { records, training, q4, weak, weakTrain, gained, binding } = fixture();
  const report = finishNativeQ4Quality(records, training, q4, binding, "b".repeat(64));
  const content = { schema_version: 1, version: binding.core_version, identity: report.identity,
    groups: report.accepted_retention_groups, regression_validation: [...records.protection, ...records.retention_regressions] };
  const retention = { ...content, content_sha256: materialContentSHA(content, true) };
  const second = collectNativeEvolutionMaterial({ identity: report.identity, activeCore: binding.core_version, retention,
    samples: Object.values(records.groups).flat() as any[], directedTeaching: records.directed_teaching,
    retentionRegressions: records.retention_regressions }).records;
  expect(second.groups.new_training.map(r => r.id)).toEqual([weakTrain]);
  expect(second.groups.new_validation.map(r => r.id)).toEqual([weak]);
  expect(second.groups.replay_training.map(r => r.id)).toContain(gained);
  expect(second.directed_teaching).toEqual(records.directed_teaching);
  expect(second.protection.map(r => r.id)).toContain("original-protected");
  expect(python("import json,sys;from luban.evolution_progress import validate_product_records;r=json.load(sys.stdin);validate_product_records(r);print('true')", second)).toBe(true);
  for (const change of ["missing-retention", "different-authorization", "wrong-answer", "lost-validation"]) {
    const broken = structuredClone(second), prior = broken.groups.replay_training.find(r => r.id === gained)!;
    const p = provenance(prior); prior.provenance = p;
    if (change === "missing-retention") delete p.retention_from_version;
    if (change === "different-authorization") p.authorization_reference = "other";
    if (change === "wrong-answer") prior.answer = "林浩";
    if (change === "lost-validation") broken.groups.new_validation = [];
    expect(() => validateNativeProductRecords(broken)).toThrow();
    expect(python("import json,sys;from luban.evolution_progress import validate_product_records\nr=json.load(sys.stdin)\ntry:validate_product_records(r);print('false')\nexcept ValueError:print('true')", broken)).toBe(true);
  }
});
