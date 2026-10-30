import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { selectIndependentNativeEvolutionCohort, withNativeEvolutionMaterial } from "../backend/native-evolution-cohort";
import { materialContentSHA } from "../backend/native-evolution-material";

const dirs: string[] = [];
afterEach(() => { for (const path of dirs.splice(0)) rmSync(path, { recursive: true, force: true }); });
const identity = "c".repeat(64), tensor = "a".repeat(64), base = { identity, original_down_sha256: tensor };
const materialResult = spawnSync(resolve(".venv/bin/python"), ["-c", "import json; from scripts.current_q4_identity_material import records; print(json.dumps(records(),ensure_ascii=False))"], { env: { ...process.env, PYTHONPATH: "src:." }, encoding: "utf8", timeout: 30000 });
if (materialResult.status !== 0) throw new Error(materialResult.stderr);
const directed = JSON.parse(materialResult.stdout);
const material = () => ({ samples: Object.values(directed.groups).flat() as Record<string, any>[], directedTeaching: directed.directed_teaching, retentionRegressions: directed.retention_regressions });
function setup() {
  const store = realpathSync(mkdtempSync(join(tmpdir(), "luban-native-cohort-"))); dirs.push(store);
  const cases = Array.from({ length: 8 }, (_, i) => ({ id: `product-ordinary-${i}`, prompt: `ordinary ${i}`, answer: String(i), source: "human", purpose: "validation", approved: true, training_allowed: true, provenance: { generation_check: { type: "exact", expected: String(i) } } }));
  writeFileSync(join(store, "original-core-protection.json"), JSON.stringify({ schema_version: 1, protocol: "original-core-protection-v1", identity, passed: true, original_tensor_sha256: tensor, cases, generation: cases.map(r => ({ id: r.id, answer: r.answer, passed: true, complete: true, branch_enabled: false })) }));
  let reads = 0;
  const memory = { model_identity: identity, samples() { reads++; return []; }, connect<T>(_fn: (db: any) => T): T { throw new Error("Directed teaching must not query ordinary experience"); } };
  return { store, memory, get reads() { return reads; } };
}
test("one immutable directed cohort per round, no repeated ledger or baseline reads", async () => {
  const fixture = setup(), input = material(); let held: ReturnType<typeof selectIndependentNativeEvolutionCohort>;
  await withNativeEvolutionMaterial(fixture.memory, input, async () => {
    held = selectIndependentNativeEvolutionCohort(fixture.memory, fixture.store, base);
    expect(held.records.protection.length).toBe(8); expect(held.records.retention_regressions.length).toBe(3);
    expect(held.records.groups.new_training.length).toBe(20);
    const before = readFileSync(join(fixture.store, "original-core-protection.json"), "utf8");
    // Changing disk after freezing cannot silently replace the round's already-bound materials.
    writeFileSync(join(fixture.store, "original-core-protection.json"), "invalid after freeze");
    expect(selectIndependentNativeEvolutionCohort(fixture.memory, fixture.store, base)).toBe(held);
    for (let i = 0; i < 100; i++) held.assertUnchanged();
    expect(fixture.reads).toBe(0); expect(Object.isFrozen(held.records.protection)).toBe(true);
    writeFileSync(join(fixture.store, "original-core-protection.json"), before);
  });
  expect(() => held!.assertUnchanged()).toThrow(/作用域/);
  expect(() => selectIndependentNativeEvolutionCohort(fixture.memory, fixture.store, base)).toThrow(/整轮/);
});
test("scope cannot change base or nest; failure releases it for next round", async () => {
  const { memory, store } = setup();
  await expect(withNativeEvolutionMaterial(memory, material(), async () => {
    selectIndependentNativeEvolutionCohort(memory, store, base);
    expect(() => selectIndependentNativeEvolutionCohort(memory, store, { ...base, original_down_sha256: "b".repeat(64) })).toThrow(/切换/);
    await expect(withNativeEvolutionMaterial(memory, material(), async () => null)).rejects.toThrow(/活动/);
    throw new Error("cancelled");
  })).rejects.toThrow("cancelled");
  await withNativeEvolutionMaterial(memory, material(), async () => { expect(selectIndependentNativeEvolutionCohort(memory, store, base).records.protection.length).toBe(8); });
});
test("accepted current12-gate protection is fully loaded once; legacy without actual weights cannot fall back", async () => {
  const { memory, store } = setup(), version = "core-123456789abc", sha = "d".repeat(64);
  const old = structuredClone(directed.groups);
  for (const rows of Object.values(old) as any[][]) for (const row of rows) { row.id = "old-" + row.id; row.prompt = "old " + row.prompt; row.provenance.messages = [{ role: "user", content: row.prompt }]; row.provenance.split_family = row.id; }
  const body = { schema_version: 1, version, identity, groups: old, regression_validation: [] };
  const raw = JSON.stringify({ ...body, content_sha256: materialContentSHA(body, true) });
  writeFileSync(join(store, version + ".retention.json"), raw);
  writeFileSync(join(store, "active-core.json"), JSON.stringify({ identity, version, sha256: sha }));
  writeFileSync(join(store, version + ".json"), JSON.stringify({ identity, version, sha256: sha, retention_sha256: Bun.CryptoHasher.hash("sha256", raw, "hex"), report: "evolution-123456789abc.json" }));
  const reportPath = join(store, "evolution-123456789abc.json"); writeFileSync(reportPath, JSON.stringify({ gates: {} }));
  await expect(withNativeEvolutionMaterial(memory, material(), async () => selectIndependentNativeEvolutionCohort(memory, store, base))).rejects.toThrow(/safetensors/);
  writeFileSync(reportPath, JSON.stringify({ gates: Object.fromEntries(["protected_core_retention_generation", "protected_core_retention_loss", "protected_replay_generation", "protected_replay_loss_retained"].map(k => [k, true])), core_retention_loss: { passed: true }, core_retention_generation: [] }));
  await withNativeEvolutionMaterial(memory, material(), async () => {
    const c = selectIndependentNativeEvolutionCohort(memory, store, base);
    expect(c.records.protection.length).toBe(14);
    expect(c.records.groups.replay_training.length).toBe(52);
    expect(c.core.version).toBe(version);
    expect(c.accepted_files_sha256[version + ".safetensors"]).toBe(sha);
  });
});
