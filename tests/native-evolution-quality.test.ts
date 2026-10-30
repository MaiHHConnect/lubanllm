import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { evaluateGeneration } from "../backend/generation-checks.ts";
import { encoded } from "../backend/canonical-json.ts";
import { parseJSON } from "../backend/wire.ts";
import { uniqueNativeEvolutionRecords } from "../backend/native-evolution-material.ts";
import { correctNativeEvolutionIds, evolutionContentSHA, finishNativeEvolutionQuality,
  nativeEvolutionProgress, nativeRetentionLossGuard, PARTIAL_GATES } from "../backend/native-evolution-quality.ts";

function python(code: string, input?: unknown) {
  const child = spawnSync(resolve(".venv/bin/python"), ["-c", code], { cwd: resolve("."),
    env: { ...process.env, PYTHONPATH: "src:." }, input: JSON.stringify(input), encoding: "utf8", timeout: 30000 });
  if (child.status !== 0) throw new Error(child.stderr || String(child.error));
  return JSON.parse(child.stdout);
}
const material = python("import json; from scripts.current_q4_identity_material import records; print(json.dumps(records(),ensure_ascii=False))");
const binding = { core_version: "core-123456789abc", core_sha256: "a".repeat(64), expert_id: null, expert_sha256: null } as const;
const row = (id: string, answer = "ok") => ({ id, prompt: id, answer, source: "human", purpose: "validation", approved: true,
  training_allowed: false, provenance: { evolution_role: "replay", retention_generation_only: true, ...(id === "original-protected" ? { retention_scope: "original_core", original_core_baseline: true } : {}), generation_check: { type: "exact", expected: answer } } });
const observed = (r: any, answer = r.answer) => ({ id: r.id, answer, complete: true, branch_enabled: false, reference_used: false,
  passed: evaluateGeneration(answer, r.provenance.generation_check, true).passed });
function fixture() {
  const records = structuredClone(material);
  records.protection = [row("original-protected")];
  const all = uniqueNativeEvolutionRecords(records.groups, records.protection, records.retention_regressions);
  const baseline = all.map(r => observed(r, r.id.startsWith("q4-id-model_") ? "尚未学会" : r.answer));
  const first = all.map(r => observed(r)), second = all.map(r => observed(r));
  const losses = uniqueNativeEvolutionRecords({}, records.protection, records.retention_regressions).map(r => ({ id: r.id, loss: .5 }));
  return { records, baseline, first, second, baselineCoreGeneration: records.protection.map((r: any) => observed(r)),
    baselineCoreLoss: losses, candidateCoreLoss: structuredClone(losses), baselineReplayLoss: .5, candidateReplayLoss: .5,
    completedSteps: 450, trainingSafetyFailed: false, identity: "c".repeat(64), binding: { ...binding },
    previousCoreSHA: "b".repeat(64), frozenRecordsSHA: evolutionContentSHA(all) };
}

test("actual current Python validator accepts the native twelve-gate report and identical progress", () => {
  const input = fixture(), report = finishNativeEvolutionQuality(input);
  expect(report.promoted).toBe(true);
  expect(Object.keys(report.gates).sort()).toEqual([...PARTIAL_GATES].sort());
  const result = python(`import json,sys
from luban.evolution_progress import validate_partial_progress
r=json.load(sys.stdin)
v=validate_partial_progress(r['report'],r['records'],r['identity'],r['version'],r['sha'])
print(json.dumps(v,ensure_ascii=False))`, { report, records: input.records, identity: input.identity,
    version: binding.core_version, sha: binding.core_sha256 });
  expect(report.learning_progress).toEqual(result);
});

test("a per-record CE regression rejects even when average replay loss improves", () => {
  const input = fixture(); input.candidateReplayLoss = .1; input.candidateCoreLoss[1]!.loss = .5001;
  const report = finishNativeEvolutionQuality(input);
  expect(report.gates.protected_replay_loss_retained).toBe(true);
  expect(report.gates.protected_core_retention_loss).toBe(false);
  expect(report.promoted).toBe(false);
  expect(report.learning_progress.learned_ids).toEqual([]);
});

test("one of two frozen candidate answers loses original capability and rejects", () => {
  const input = fixture(), index = input.second.findIndex(r => r.id === "original-protected");
  input.second[index] = { ...input.second[index]!, answer: "wrong", passed: false };
  const report = finishNativeEvolutionQuality(input);
  expect(report.gates.protected_core_retention_generation).toBe(false);
  expect(report.gates.original_correct_retained).toBe(false);
  expect(report.promoted).toBe(false);
});

test("replay protection includes every baseline-correct validation answer, not baseline failures", () => {
  const input = fixture(), id = input.records.groups.replay_validation[0].id;
  const index = input.first.findIndex(r => r.id === id);
  input.first[index] = { ...input.first[index]!, answer: "wrong", passed: false };
  expect(finishNativeEvolutionQuality(input).gates.protected_replay_generation).toBe(false);
  input.baseline[index] = { ...input.baseline[index]!, answer: "wrong", passed: false };
  const report = finishNativeEvolutionQuality(input);
  expect(report.gates.protected_replay_generation).toBe(true);
  expect(report.protected_replay_ids).not.toContain(id);
});

test("missing/reordered loss rows and incomplete original baseline fail closed", () => {
  for (const mutate of [(x: any) => x.candidateCoreLoss.pop(), (x: any) => x.baselineCoreLoss.reverse(),
    (x: any) => x.baselineCoreGeneration[0].complete = false]) {
    const input = fixture(); mutate(input); expect(() => finishNativeEvolutionQuality(input)).toThrow();
  }
});

test("450 actual updates, frozen material and original safety are all required", () => {
  const short = fixture(); short.completedSteps = 449; expect(() => finishNativeEvolutionQuality(short)).toThrow(/450/);
  const unsafe = fixture(); unsafe.trainingSafetyFailed = true; expect(finishNativeEvolutionQuality(unsafe).promoted).toBe(false);
  const changed = fixture(); changed.frozenRecordsSHA = "0".repeat(64); expect(finishNativeEvolutionQuality(changed).promoted).toBe(false);
  const unchanged = fixture(); unchanged.previousCoreSHA = binding.core_sha256; expect(finishNativeEvolutionQuality(unchanged).promoted).toBe(false);
});

test("claimed pass, reordered IDs, non-boolean flags and expert/reference use cannot forge mastery", () => {
  const r = row("proof"), proof = observed(r);
  for (const bad of [{ ...proof, answer: "wrong" }, { ...proof, complete: 1 }, { ...proof, reference_used: true },
    { ...proof, branch_enabled: true }, { ...proof, id: "other" }])
    expect(() => correctNativeEvolutionIds([bad as any], [r])).toThrow();
});

test("non-finite contains success is retained but never grants new exact mastery", () => {
  const r = row("contains"); r.provenance.generation_check.type = "contains";
  const proof = observed(r), result = nativeEvolutionProgress([r], [proof], [proof], [proof], binding);
  expect(result.mastered_ids).toEqual([]); expect(result.retained_ids).toEqual([r.id]);
});

test("fixed FP32 retention tolerance matches Python exactly and rejects duplicate/invalid losses", () => {
  const baseline = [{ id: "a", loss: 1 }, { id: "b", loss: .5 }];
  const candidate = [{ id: "a", loss: 1 + 64 * 2 ** -23 }, { id: "b", loss: .5 + 65 * 2 ** -23 }];
  expect(nativeRetentionLossGuard(baseline, candidate)).toEqual(python(`import json,sys
from luban.evolution import _retention_loss_guard
r=json.load(sys.stdin);print(json.dumps(_retention_loss_guard(r[0],r[1]),ensure_ascii=False))`, [baseline, candidate]));
  for (const loss of [NaN, Infinity, -1, true]) expect(() => nativeRetentionLossGuard([{ id: "x", loss: loss as any }], [])).toThrow();
  expect(() => nativeRetentionLossGuard([baseline[0]!, baseline[0]!], candidate)).toThrow(/重复/);
});

test("material numeric bindings normalize JSON float spelling like Python", () => {
  const source = '{"small":0.00001,"zero":-0.0,"whole":1e0,"large":1e20,"astral😀":0.0001}';
  const expected = python("import json; s=" + JSON.stringify(source) + "; print(json.dumps(json.dumps(json.loads(s),sort_keys=True,ensure_ascii=False)))");
  expect(encoded(parseJSON(source))).toBe(expected);
});

test("native scoring matches Python for Unicode, JSON, forbidden strings and syntax without executing code", () => {
  const cases = [
    ["  无限进化\u001c", { type: "exact", expected: "无限进化" }, true],
    ["１２３", { type: "semantic", validator: "integer", expected: 123 }, true],
    ['{"n":1,"n":2}', { type: "semantic", validator: "json", expected: { n: 2 } }, true],
    ['{"n":1.0}', { type: "semantic", validator: "json", expected: { n: 1 } }, true],
    ["ＡＢＣ", { type: "semantic", validator: "normalized_exact", expected: "ABC" }, true],
    ["Straße!", { type: "semantic", validator: "translation", accepted: ["STRASSE"] }, true],
    ["ok ＳＴＲＡＳＳＥ", { type: "contains", expected: "ok", forbidden: ["straße"] }, true],
    ["ok", { type: "exact", expected: "ok" }, false],
    ["def f(x):\n    return x.upper()", { type: "semantic", validator: "python_upper" }, true],
    ["def f(x=run()):\n    return x.upper()", { type: "semantic", validator: "python_upper" }, true],
    ["import os\nos._exit(99)", { type: "semantic", validator: "python_upper" }, true],
  ];
  const expected = python(`import json,sys
from luban.generation_checks import evaluate_generation
print(json.dumps([evaluate_generation(*row) for row in json.load(sys.stdin)],ensure_ascii=False))`, cases);
  expect(cases.map(([answer, check, complete]) => evaluateGeneration(answer, check, complete))).toEqual(expected);
});
