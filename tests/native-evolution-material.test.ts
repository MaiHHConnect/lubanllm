import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import {
  collectNativeEvolutionMaterial, consolidateMaterialGroups, DIRECTED_TEACHING,
  materialContentSHA, materialContext, originalCoreProtectionRows, provenance,
  uniqueNativeEvolutionRecords, validateMaterialGroups, validateNativeCoreRetention,
  validateNativeProductRecords,
} from "../backend/native-evolution-material";

const identity = "test-original-qwen", tensorSHA = "a".repeat(64), version = "core-123456789abc";
const copy = <T>(v: T): T => structuredClone(v);
function row(id: string, purpose = "training", role = "new", answer = "ok"): any {
  return { id, prompt: `Question ${id}`, answer, source: "human", purpose, approved: true, training_allowed: true,
    provenance: { evolution_role: role, split_family: id, verified_by: "local requester", generation_check: { type: "exact", expected: answer } } };
}
function original() {
  const cases = Array.from({ length: 8 }, (_, i) => row(`product-ordinary-${i}`, "validation", "replay", String(i)));
  return { schema_version: 1, protocol: "original-core-protection-v1", identity, passed: true,
    original_tensor_sha256: tensorSHA, cases, generation: cases.map(r => ({ id: r.id, answer: r.answer, passed: true, complete: true, branch_enabled: false })) };
}
function groups() {
  return { new_training: [row("new-train")], new_validation: [row("new-valid", "validation")], replay_training: [row("old-train", "training", "replay")], replay_validation: [row("old-valid", "validation", "replay")] };
}
function regression(id = "unknown-user") { const r = row(id, "validation", "replay", "不知道"); r.training_allowed = false; r.provenance.retention_generation_only = true; return r; }
function snapshot() {
  const content = { schema_version: 1, version, identity, groups: groups(), regression_validation: [regression()] };
  return { ...content, content_sha256: materialContentSHA(content, true) };
}
function ordinarySamples() {
  return [...Array.from({ length: 350 }, (_, i) => row(`n-${String(i).padStart(3, "0")}`, i % 5 ? "training" : "validation")), row("replay-t", "training", "replay"), row("replay-v", "validation", "replay")];
}
const base = () => ({ identity, activeCore: null, originalTensorSHA256: tensorSHA, originalProtection: original() });
function python(code: string, input?: any) {
  const result = spawnSync(resolve(".venv/bin/python"), ["-c", code], { cwd: resolve("."), env: { ...process.env, PYTHONPATH: "src:." }, encoding: "utf8", input: input === undefined ? undefined : JSON.stringify(input), timeout: 30000 });
  if (result.status !== 0) throw new Error(result.stderr || String(result.error)); return JSON.parse(result.stdout);
}
const directed = python("import json; from scripts.current_q4_identity_material import records; print(json.dumps(records(), ensure_ascii=False))");
const directedInput = () => ({ ...base(), samples: Object.values(directed.groups).flat() as any[], directedTeaching: directed.directed_teaching, retentionRegressions: directed.retention_regressions });

describe("native material source and full preservation", () => {
  test("real authored three-fact material matches Python contract, preserves unknown user and original8", () => {
    const result = collectNativeEvolutionMaterial(directedInput()), r = result.records;
    expect(result.training_started).toBe(false); expect(result.quality_admitted).toBe(false);
    expect(r.groups.new_training.length).toBe(20); expect(r.groups.new_validation.length).toBe(10);
    expect(r.protection.length).toBe(8); expect(r.retention_regressions.length).toBe(3);
    expect(r.protection.every(x => x.training_allowed === false && provenance(x).original_core_baseline === true)).toBe(true);
    expect(Object.values(r.groups).flat().some(x => x.id === "q4-id-user-boundary-2")).toBe(false);
    expect(r.retention_regressions[2].answer).toBe("不知道");
    expect((r as any).directed_teaching.facts.model_positioning.value).toBe("无限进化");
    expect(python("import json,sys; from luban.evolution_progress import validate_product_records; r=json.load(sys.stdin); validate_product_records(r); print('true')", r)).toBe(true);
  });
  test("missing/misbound/false original baseline cannot start core=None", () => {
    expect(() => collectNativeEvolutionMaterial({ ...directedInput(), originalProtection: undefined })).toThrow();
    expect(() => collectNativeEvolutionMaterial({ ...directedInput(), originalTensorSHA256: "b".repeat(64) })).toThrow(/张量/);
    const bad = original(); bad.generation[3]!.answer = "wrong";
    expect(() => originalCoreProtectionRows(bad, identity, tensorSHA)).toThrow(/逐条生成/);
    bad.generation[3]!.answer = "3"; bad.cases[7]!.id = "product-ordinary-6";
    expect(() => originalCoreProtectionRows(bad, identity, tensorSHA)).toThrow();
    expect(() => originalCoreProtectionRows(original(), "other", tensorSHA)).toThrow();
  });
  test("accepted retention survives deletion of all old database rows and becomes replay", () => {
    const retained = snapshot(), result = collectNativeEvolutionMaterial({ ...directedInput(), activeCore: version, retention: retained, originalProtection: undefined });
    expect(result.records.groups.replay_training.map(r => r.id)).toContain("new-train");
    expect(result.records.groups.replay_validation.map(r => r.id)).toContain("new-valid");
    expect(result.records.protection.map(r => r.id)).toEqual(["new-valid", "old-valid", "unknown-user"]);
    expect(uniqueNativeEvolutionRecords({}, result.records.protection, result.records.retention_regressions).map(r => r.id)).toContain("unknown-user");
    expect(provenance(result.records.groups.replay_training.find(r => r.id === "new-train")!).retention_from_version).toBe(version);
    expect(retained.groups.new_training[0]!.provenance.evolution_role).toBe("new");
    const broken = copy(retained); broken.groups.new_training[0]!.answer = "tampered";
    expect(() => validateNativeCoreRetention(broken, identity, version)).toThrow(/内容校验/);
    expect(() => collectNativeEvolutionMaterial({ ...directedInput(), activeCore: version })).toThrow();
    expect(() => validateNativeCoreRetention(retained, identity, "core-000000000000")).toThrow();
  });
  test("retained TRAIN cannot be relabelled as independent VALID; correction conflicts fail", () => {
    const retained = snapshot(), repeated = copy(retained.groups.new_training[0]!); repeated.id = "different-id"; repeated.purpose = "validation";
    expect(() => consolidateMaterialGroups([repeated], retained)).toThrow(/冲突/);
    const changed = copy(retained.groups.new_training[0]!); changed.answer = "replacement";
    expect(() => consolidateMaterialGroups([changed], retained)).toThrow(/冲突/);
    const boundary = row("boundary-training", "training", "new", "林浩"); boundary.prompt = retained.regression_validation[0]!.prompt;
    expect(() => consolidateMaterialGroups([boundary], retained)).toThrow(/生成保护/);
  });
  test("full groups exactly match Python consolidation for retained and current records", () => {
    const retained = snapshot(), samples = [row("fresh"), row("fresh-valid", "validation"), ...Object.values(retained.groups).flat()];
    const native = consolidateMaterialGroups(samples, retained);
    const expected = python(`import json,sys
from types import SimpleNamespace
import luban.retention as r
v=json.load(sys.stdin)
r.read_core_retention=lambda runtime:v['snapshot']
g=r.consolidation_groups(SimpleNamespace(memory=SimpleNamespace(samples=lambda:v['samples'])))
for rows in g.values():
 for row in rows: row['provenance']=r.provenance(row)
print(json.dumps(g,ensure_ascii=False))`, { snapshot: retained, samples });
    for (const rows of Object.values(native)) for (const r of rows) r.provenance = provenance(r);
    expect(native).toEqual(expected);
  });
});

describe("authorization, split and experience guards", () => {
  test("identity facts cannot authorize a user's name or lose positioning coverage", () => {
    const wrong = copy(directed); wrong.directed_teaching.facts.user_name = { subject: "user", value: "林浩", training_ids: [], validation_ids: [] };
    expect(() => validateNativeProductRecords(wrong)).toThrow(/不包含用户身份/);
    const omitted = copy(directed); delete omitted.directed_teaching.facts.model_positioning;
    expect(() => validateNativeProductRecords(omitted)).toThrow(/全部NEW/);
    const auth = copy(directed); auth.directed_teaching.authorization.explicit = false;
    expect(() => validateNativeProductRecords(auth)).toThrow(/授权/);
    const unbound = copy(directed); unbound.groups.new_training[0].provenance.authorization_reference = "other";
    expect(() => validateNativeProductRecords(unbound)).toThrow(/绑定授权/);
    const user = copy(directed); user.groups.new_training[0].answer = "林浩";
    expect(() => validateNativeProductRecords(user)).toThrow(/短答案/);
  });
  test("context and family split leakage rejected; unverified and revoked data excluded", () => {
    const g = groups(); g.new_validation[0]!.prompt = g.new_training[0]!.prompt;
    expect(() => validateMaterialGroups(g)).toThrow(/上下文/);
    g.new_validation[0]!.prompt = "different"; g.new_validation[0]!.provenance.split_family = g.new_training[0]!.provenance.split_family;
    expect(() => validateMaterialGroups(g)).toThrow(/家族/);
    const raw = row("unverified"); raw.source = "teacher";
    const revoked = row("revoked"); revoked.training_allowed = false;
    expect(consolidateMaterialGroups([raw, revoked], null).new_training).toEqual([]);
    const one = row("one"); one.prompt = "Ａ\u001cＢ";
    const two = row("two"); two.prompt = "A B";
    expect(materialContext(one)).toBe(materialContext(two));
  });
  test("generation-only user boundaries cannot become optimizer materials or conflict by ID", () => {
    const unsafe = copy(directed); unsafe.retention_regressions[0].training_allowed = true;
    expect(() => validateNativeProductRecords(unsafe)).toThrow(/许可/);
    const collision = copy(directed); collision.retention_regressions[0].id = collision.groups.new_training[0].id;
    expect(() => validateNativeProductRecords(collision)).toThrow(/同一ID/);
    const r = row("duplicate"); expect(uniqueNativeEvolutionRecords({ ...groups(), new_training: [r] }, [copy(r)]).length).toBe(4);
    const altered = copy(r); altered.provenance.generation_check.expected = "different";
    expect(() => uniqueNativeEvolutionRecords({ ...groups(), new_training: [r] }, [altered])).toThrow(/同一ID/);
  });
  test("ordinary 350 pool stays separate, bounds57, carries pending pool across rounds", () => {
    const samples = ordinarySamples(), first = collectNativeEvolutionMaterial({ ...base(), samples });
    expect((first.records as any).experience.length).toBe(350); expect(first.records.groups.new_training.length).toBe(57);
    expect(first.unspent_count).toBe(350); expect(first.records.groups.replay_training.length).toBe(1);
    const run = { kind: "evolution", scope: "product", state: "rejected", records: first.records, binding: { identity, records_sha256: materialContentSHA(first.records) } };
    const repeat = collectNativeEvolutionMaterial({ ...base(), samples, runs: [run] });
    expect(repeat.unspent_count).toBe(0); expect(repeat.records).toEqual(first.records);
    expect(python("import json,sys; from luban.evolution_progress import validate_product_records; r=json.load(sys.stdin); validate_product_records(r); print('true')", first.records)).toBe(true);
    expect(() => collectNativeEvolutionMaterial({ ...base(), samples: samples.slice(1) })).toThrow(/350/);
    expect(() => collectNativeEvolutionMaterial({ ...base(), samples, runs: [{ ...run, binding: { ...run.binding, records_sha256: "bad" } }] })).toThrow(/账本/);
  });
  test("metadata edits do not mint another350; research questions cannot unlock ordinary pool", () => {
    const samples = ordinarySamples(), first = collectNativeEvolutionMaterial({ ...base(), samples });
    const changed = copy(samples); for (const r of changed) r.provenance.unrelated_note = "new metadata";
    const run = { kind: "evolution", records: first.records, binding: { identity, records_sha256: materialContentSHA(first.records) } };
    expect(() => collectNativeEvolutionMaterial({ ...base(), samples: changed, runs: [run] })).toThrow(/350/);
    for (const r of samples) r.provenance.protocol_id = "research";
    expect(() => collectNativeEvolutionMaterial({ ...base(), samples })).toThrow(/350/);
    const ambiguous = { ...directed, experience: [] }; expect(() => validateNativeProductRecords(ambiguous)).toThrow(/互斥/);
  });
});
