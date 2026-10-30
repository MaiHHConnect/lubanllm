import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, realpathSync, writeFileSync, readFileSync, rmSync, symlinkSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { withNativeEvolutionMaterial, selectIndependentNativeEvolutionCohort } from "../backend/native-evolution-cohort.ts";
import { readNativeEffectiveCore, assertNativeEffectiveCoreReload, nativeTrialSourceReason } from "../backend/native-effective-core.ts";
import { materialContentSHA } from "../backend/native-evolution-material.ts";

const roots: string[] = [];
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }); });
const hash = (v: Uint8Array | string) => Bun.CryptoHasher.hash("sha256", v, "hex");
function fixture() {
  const store = realpathSync(mkdtempSync(join(tmpdir(), "luban-effective-core-"))); roots.push(store);
  const base = { identity: "a".repeat(64), model_manifest_sha256: "b".repeat(64), config_sha256: "c".repeat(64),
    checkpoint_sha256: "d".repeat(64), tokenizer_sha256: "e".repeat(64), original_down_sha256: "f".repeat(64),
    engine_binary_sha256: "0".repeat(64), core_parameter: "model.layers.23.mlp.down_proj.weight",
    core_shape: [2, 3] as [number, number], dtype: "BF16" as const, device: "mps" as const };
  const version = "core-123456789abc", reportName = "evolution-123456789abc.json";
  const header = Buffer.from(JSON.stringify({ [base.core_parameter]: { dtype: "BF16", shape: base.core_shape, data_offsets: [0, 12] } }));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  const weights = Buffer.concat([length, header, Buffer.alloc(12)]);
  const retained = { schema_version: 1, identity: base.identity, version,
    groups: { new_training: [], new_validation: [], replay_training: [], replay_validation: [] }, regression_validation: [] };
  const retention = JSON.stringify({ ...retained, content_sha256: materialContentSHA(retained, true) });
  const pointer = { identity: base.identity, version, sha256: hash(weights) };
  const meta = { ...pointer, parameter: base.core_parameter, shape: base.core_shape, dtype: "torch.bfloat16",
    format: "cumulative core tensor replacement", trained_parameters: 6, retention_sha256: hash(retention), report: reportName };
  const report = { identity: base.identity, version, promoted: true, publication_state: "published", checkpoint: { sha256: meta.sha256, retention_sha256: meta.retention_sha256 },
    gates: {} as Record<string, boolean>, core_retention_loss: { passed: true }, core_retention_generation: [] };
  const save = () => {
    writeFileSync(join(store, "active-core.json"), JSON.stringify(pointer)); writeFileSync(join(store, version + ".json"), JSON.stringify(meta));
    writeFileSync(join(store, version + ".safetensors"), weights); writeFileSync(join(store, version + ".retention.json"), retention);
    writeFileSync(join(store, reportName), JSON.stringify(report));
  }; save(); return { store, base, pointer, meta, report, weights, save };
}

test("valid physical old core is preserved on disk but every effective selector and rollback receipt sees original None", () => {
  const f = fixture(), before = readFileSync(join(f.store, "active-core.json"));
  const value = readNativeEffectiveCore(f.store, f.base);
  expect(value.state.core).toBeNull(); expect(value.physical).toEqual(f.pointer);
  expect(value.core_requires_review?.reason).toBe("核心报告没有当前逐条保护门");
  expect(readFileSync(join(f.store, "active-core.json"))).toEqual(before);
  expect(() => assertNativeEffectiveCoreReload(f.pointer, { loaded: { core_loaded: false } }, value.state)).not.toThrow();
  expect(() => assertNativeEffectiveCoreReload(f.pointer, { loaded: { core_loaded: true, core_file_sha256: f.meta.sha256 } }, value.state)).toThrow();
  expect(() => assertNativeEffectiveCoreReload(f.pointer, { loaded: { core_loaded: false } }, { core: null })).toThrow();
});

test("current qualified core remains active and requires its actual file/tensor reload binding", () => {
  const f = fixture();
  for (const key of ["protected_core_retention_generation", "protected_core_retention_loss", "protected_replay_generation", "protected_replay_loss_retained"]) f.report.gates[key] = true;
  f.save(); const value = readNativeEffectiveCore(f.store, f.base);
  expect(value.state.core.version).toBe(f.pointer.version); expect(value.core_requires_review).toBeNull();
  const loaded = { core_loaded: true, core_file_sha256: f.meta.sha256, core_tensor_sha256: value.state.core.tensor_sha256 };
  expect(() => assertNativeEffectiveCoreReload(f.pointer, { loaded }, value.state)).not.toThrow();
  expect(() => assertNativeEffectiveCoreReload(f.pointer, { loaded: { ...loaded, core_tensor_sha256: "0".repeat(64) } }, value.state)).toThrow();
});

test("material selection uses the same validated None and the real original eight cases", async () => {
  const f = fixture();
  const generated = spawnSync(".venv/bin/python", ["-c", "import json;from scripts.current_q4_identity_material import records;print(json.dumps(records(),ensure_ascii=False))"],
    { cwd: process.cwd(), env: { ...process.env, PYTHONPATH: "src:." }, encoding: "utf8", timeout: 30000 });
  expect(generated.status, generated.stderr).toBe(0); const source = JSON.parse(generated.stdout);
  const cases = Array.from({ length: 8 }, (_, i) => ({ id: "product-ordinary-" + i, prompt: "original " + i, answer: String(i),
    source: "human", purpose: "validation", approved: true, training_allowed: true,
    provenance: { generation_check: { type: "exact", expected: String(i) } } }));
  const original = { schema_version: 1, protocol: "original-core-protection-v1", identity: f.base.identity,
    original_tensor_sha256: f.base.original_down_sha256, passed: true, cases,
    generation: cases.map(row => ({ id: row.id, answer: row.answer, passed: true, complete: true, branch_enabled: false })) };
  writeFileSync(join(f.store, "original-core-protection.json"), JSON.stringify(original));
  const memory = { model_identity: f.base.identity, samples: () => [], connect: () => { throw new Error("No ordinary database access"); } };
  const material = { samples: Object.values(source.groups).flat() as Record<string, any>[], directedTeaching: source.directed_teaching,
    retentionRegressions: source.retention_regressions };
  await withNativeEvolutionMaterial(memory, material, async () => {
    const selected = selectIndependentNativeEvolutionCohort(memory, f.store, f.base);
    expect(selected.core).toBeNull(); expect(selected.physical_core).toEqual(f.pointer);
    expect(selected.core_requires_review?.version).toBe(f.pointer.version);
    expect(selected.records.protection.map(row => row.id)).toEqual(cases.map(row => row.id));
  });
  unlinkSync(join(f.store, "original-core-protection.json"));
  await expect(withNativeEvolutionMaterial(memory, material, async () => selectIndependentNativeEvolutionCohort(memory, f.store, f.base))).rejects.toThrow();
});

test("old report never hides bad identity, corrupted/nonfinite weights, retention damage or symlinks", () => {
  for (const kind of ["identity", "bytes", "nonfinite", "retention", "symlink"]) {
    const f = fixture();
    if (kind === "identity") { f.pointer.identity = "0".repeat(64); f.save(); }
    if (kind === "bytes") { f.weights[f.weights.length - 1] = 1; writeFileSync(join(f.store, f.pointer.version + ".safetensors"), f.weights); }
    if (kind === "nonfinite") {
      f.weights.writeUInt16LE(0x7fc0, f.weights.length - 2); f.pointer.sha256 = f.meta.sha256 = hash(f.weights);
      f.report.checkpoint.sha256 = f.meta.sha256; f.save();
    }
    if (kind === "retention") writeFileSync(join(f.store, f.pointer.version + ".retention.json"), "{}");
    if (kind === "symlink") {
      const path = join(f.store, f.pointer.version + ".safetensors"); writeFileSync(join(f.store, "saved-weight"), f.weights);
      unlinkSync(path); symlinkSync(join(f.store, "saved-weight"), path);
    }
    expect(() => readNativeEffectiveCore(f.store, f.base), kind).toThrow();
  }
});

test("trial pointer and metadata are refused before full source or tensor reads", () => {
  const f=fixture();
  writeFileSync(join(f.store,"active.json"),JSON.stringify({id:"learned-000000000001",execution_scope:"local-trial"}));
  expect(nativeTrialSourceReason(f.store)).toContain("本地试用");
  expect(()=>readNativeEffectiveCore(f.store,f.base)).toThrow("本地试用");
  unlinkSync(join(f.store,"active.json"));
  writeFileSync(join(f.store,"active-router.json"),JSON.stringify({id:"router-000000000001"}));
  writeFileSync(join(f.store,"router-000000000001.json"),JSON.stringify({quality_admitted:false}));
  expect(nativeTrialSourceReason(f.store)).toContain("本地试用");
});
