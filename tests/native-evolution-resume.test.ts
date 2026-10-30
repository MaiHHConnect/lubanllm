import { test, expect } from "bun:test";
import { mkdtemp, writeFile, rm, symlink, rename, realpath } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readdirSync } from "node:fs";
import { materialContentSHA, collectNativeEvolutionMaterial, DIRECTED_TEACHING } from "../backend/native-evolution-material.ts";
import { NativeCore450Owner } from "../.native-migration/research/native-core450-owner-controller/owner.ts";
import { reopenNativeEvolutionPreparation, nativeEvolutionPreparationBinding, nativeEvolutionResumeSupported, nativeEvolutionResumeMaterial } from "../backend/native-evolution-resume.ts";
import { Database } from "bun:sqlite";
const sha = (v: string | Buffer) => Bun.CryptoHasher.hash("sha256", v, "hex");
async function fixture() {
  const directory = await realpath(await mkdtemp(join(tmpdir(), "native-resume-"))), id = "123456789abc", identity = "a".repeat(64);
  const groups: Record<string, any[]> = { new_training: [], new_validation: [], replay_training: [], replay_validation: [] }, facts: any = {};
  const row = (id: string, purpose: string, role: string, answer: string, extra = {}) => ({ id, prompt: id, answer, purpose,
    source: "human", approved: true, training_allowed: true, provenance: { evolution_role: role, split_family: id,
      generation_check: { type: "exact", expected: answer }, ...extra } });
  for (const [key, value] of Object.entries({ model_name: "luban0", model_fine_tuner: "麦好火" })) {
    facts[key] = { subject: "model", value, training_ids: [key + "-train"], validation_ids: [key + "-valid"] };
    for (const purpose of ["training", "validation"]) groups["new_" + purpose]!.push(row(key + (purpose === "training" ? "-train" : "-valid"), purpose, "new", value,
      { protocol_id: DIRECTED_TEACHING, authorization_reference: "human", teaching_fact: key }));
  }
  groups.replay_training!.push(row("old-train", "training", "replay", "ok")); groups.replay_validation!.push(row("old-valid", "validation", "replay", "ok"));
  const records = { groups, protection: [], retention_regressions: [], directed_teaching: { protocol: DIRECTED_TEACHING,
    authorization: { source: "human", explicit: true, training_allowed: true, reference: "human" }, facts } };
  const base = { identity, core_shape: [2, 3] }, tensors: any = {}, rows: any[] = [];
  async function tensor(name: string, dtype: string, shape: number[]) {
    const bytes = shape.reduce((a, b) => a * b, 1) * ({ bf16: 2, f32: 4, i64: 8, u8: 1 }[dtype]!);
    const raw = Buffer.alloc(bytes); if (dtype === "i64") { raw.writeBigInt64LE(name.endsWith("labels") ? -100n : 0n); raw.writeBigInt64LE(1n, 8); }
    tensors[name] = { dtype, shape, file: name + ".raw", bytes, sha256: sha(raw) }; await writeFile(join(directory, name + ".raw"), raw);
  }
  for (const name of ["candidate", "master", "reference", "exp_avg", "exp_avg_sq"]) await tensor(name, name === "candidate" ? "bf16" : "f32", [2, 3]);
  await tensor("basis", "f32", [3, 1]); await tensor("adam_step", "f32", []); await tensor("norm", "bf16", [2]); await tensor("head", "bf16", [4, 2]);
  await tensor("cpu_rng", "u8", [5056]); await tensor("mps_rng", "u8", [44]);
  const training = [...groups.new_training!, ...groups.replay_training!];
  for (const [i, sample] of training.entries()) {
    const key = "r" + String(i).padStart(3, "0");
    rows.push({ key, role: i < 2 ? "new" : "replay", sample_id: sample.id, sample_sha256: materialContentSHA(sample, true), length: 2, gold: 1, input_tokens_sha256: "a".repeat(64) });
    await tensor(key + "_x", "bf16", [1, 2, 3]); await tensor(key + "_residual", "bf16", [1, 2, 2]);
    await tensor(key + "_ids", "i64", [1, 2]); await tensor(key + "_labels", "i64", [1, 2]);
  }
  const schedule = Array.from({ length: 450 }, (_, i) => ({ ordinal: i + 1, role: i % 2 ? "replay" : "new", ce: i % 2 ? "r002" : "r00" + (Math.floor(i / 2) % 2), replay: "r002" }));
  const config = { identity, device: "mps", steps: "450", norm_eps: "1e-6", train_eos: "1", checkpoint_steps: "225,450", rng_mode: "fresh", training_offset: "0", learning_rate: "1e-05", distillation_weight: "5.0" };
  const manifest = ["LUBAN_NATIVE_TRAINING_V3", ...Object.entries(config).map(([k, v]) => `config\t${k}\t${v}`),
    ...Object.entries(tensors).map(([k, t]: any) => `tensor\t${k}\t${t.dtype}\t${t.shape.join(",") || "-"}\t${t.file}\t${t.bytes}\t${t.sha256}`),
    ...rows.map(r => `row\t${r.key}\t${r.role}\t${r.sample_sha256}\t${r.length}\t${r.gold}`),
    ...schedule.map(r => `step\t${r.ordinal}\t${r.role}\t${r.ce}\t${r.replay}`)].join("\n") + "\n";
  await writeFile(join(directory, "manifest.tsv"), manifest); await writeFile(join(directory, "fixture.json"), "{}");
  const graphBinding = { base, identity, records_sha256: materialContentSHA(records), accepted_tensor_sha256: tensors.candidate.sha256,
    accepted_core_version: null, fixture_only: true, core_last_down_outside_prefix: true, branch_enabled: false, expert: null, use_cache: true, cache_fresh_per_complete_sample: true,
    product_admission: false, quality_checked: false, basis: { sha256: tensors.basis.sha256 } };
  const content = { schema_version: 1, identity, version: "core-" + id, groups, regression_validation: [] };
  const qualification = Buffer.from(JSON.stringify({ ...content, content_sha256: materialContentSHA(content, true) }));
  const binding = nativeEvolutionPreparationBinding({ id, base, records, graphBinding, qualification,
    fixture: { directory, manifest_sha256: sha(manifest), config, tensors, rows, schedule } });
  const owners = new NativeCore450Owner();
  return { binding, base, expected: { id, kind: "evolution", binding: { identity, arguments: { steps: 450 }, core_version: null, records_sha256: materialContentSHA(records) } },
    cohort: { records, records_sha256: materialContentSHA(records), assertUnchanged() {} },
    acceptedTensorSHA: tensors.candidate.sha256, trainingBinarySHA: "c".repeat(64), owners, guard: { assertQuiescent() {} },
    async cleanup() { await owners.close(); await rm(directory, { recursive: true, force: true }); } };
}

test("resume reopens original full450 inputs and owner FDs without new baseline, seed, or model state", async () => {
  const r = await fixture();
  try {
    const before = readdirSync("/dev/fd").length, value = await reopenNativeEvolutionPreparation(r);
    expect(value.prepared.id).toBe(r.expected.id); expect(value.prepared.resume).toBe(true);
    expect(value.prepared.fixture.schedule.length).toBe(450); expect(value.accepted.sha256).toBe(r.acceptedTensorSHA);
    expect(value.prepared.fixture.tensorOwner("candidate").handle).toBe(value.accepted.handle);
    expect(() => value.prepared.fixture.tensorOwner("missing")).toThrow();
    expect(Object.hasOwn(value.prepared,"previousVersion")).toBe(false); expect(value.prepared.cohort).toBe(r.cohort);
    expect(readdirSync("/dev/fd").length).toBeGreaterThan(before);
    await value.prepared.assertUnchanged(); await r.owners.release(value.owner);
    expect(value.closed()).toBe(true); expect(value.accepted.handle.fd).toBe(-1);
    expect(readdirSync("/dev/fd").length).toBe(before); await expect(value.prepared.assertUnchanged()).rejects.toThrow();
  } finally { await r.cleanup(); }
});

test("resume refuses mismatched journal/material/previous source, changed tensors, symlinks and cancellation without leaking FDs", async () => {
  for (const bad of ["id", "records", "previous", "schedule", "tensor", "symlink", "cancel"]) {
    const r = await fixture(), path = join(r.binding.fixture.directory, "head.raw");
    try {
      if (bad === "id") r.expected.id = "abcdef123456";
      if (bad === "records") r.binding.records.groups.new_training[0].answer = "林浩";
      if (bad === "previous") r.acceptedTensorSHA = "d".repeat(64);
      if (bad === "schedule") r.binding.fixture.schedule[0]!.ce = "r002";
      if (bad === "tensor") await writeFile(path, Buffer.alloc(16, 1));
      if (bad === "symlink") { await rename(path, path + ".old"); await symlink(path + ".old", path); }
      const before = readdirSync("/dev/fd").length;
      await expect(reopenNativeEvolutionPreparation({ ...r, ...(bad === "cancel" ? { signal: AbortSignal.abort() } : {}) })).rejects.toThrow();
      expect(readdirSync("/dev/fd").length).toBe(before);
    } finally { await r.cleanup(); }
  }
});

test("resume cleanup failure retains the same file owner until a successful retry", async () => {
  const r = await fixture();
  try {
    const value = await reopenNativeEvolutionPreparation(r), handle = value.accepted.handle;
    const close = handle.close.bind(handle); let attempts = 0;
    handle.close = async () => { if (++attempts === 1) throw new Error("injected close failure"); await close(); };
    await expect(r.owners.release(value.owner)).rejects.toThrow(/FD/);
    expect(value.closed()).toBe(false); expect(handle.fd).not.toBe(-1);
    expect(r.owners.status().retained_owners).toBe(1);
    await r.owners.release(value.owner);
    expect(value.closed()).toBe(true); expect(handle.fd).toBe(-1); expect(attempts).toBe(2);
  } finally { await r.cleanup(); }
});


test("resume preview recognizes the actual SQLite journal fields and rejects old or different rounds without opening files", async () => {
  const r = await fixture(), db = new Database(":memory:");
  try {
    db.exec("CREATE TABLE training_runs (id TEXT, identity TEXT, kind TEXT, state TEXT, body TEXT, updated REAL)");
    const implementation = { graph: "a".repeat(64), trainer: "b".repeat(64) };
    const original = { id: r.expected.id, kind: "evolution", scope: "product", records: r.cohort.records,
      arguments: { steps: 450 }, binding: { ...r.expected.binding, scope: "product", implementation_sha256: implementation },
      checkpoint: { name: "checkpoint-225.pt", sha256: "c".repeat(64) }, coordinator: { native_preparation: r.binding } };
    const current = { identity: r.base.identity, records: r.cohort.records, core_version: null, implementation };
    for (const scenario of ["paused", "running", "queued", "committed", "old16", "source", "core", "material", "preparation", "checkpoint", "identity", "expert"]) {
      const body = structuredClone(original); let state = ["paused", "running", "queued", "committed"].includes(scenario) ? scenario : "paused";
      if (scenario === "old16") body.arguments.steps = body.binding.arguments.steps = 16;
      if (scenario === "source") body.binding.implementation_sha256.trainer = "d".repeat(64);
      if (scenario === "core") (body.binding as any).core_version = "old-core";
      if (scenario === "material") body.records.groups.new_training[0].answer = "wrong";
      if (scenario === "preparation") delete (body.coordinator as any).native_preparation;
      if (scenario === "checkpoint") (body as any).checkpoint = null;
      db.exec("DELETE FROM training_runs");
      db.query("INSERT INTO training_runs VALUES (?,?,?,?,?,?)").run(body.id, scenario === "identity" ? "e".repeat(64) : current.identity,
        scenario === "expert" ? "expert" : "evolution", state, JSON.stringify(body), 1);
      const row = db.query("SELECT * FROM training_runs").get();
      expect(nativeEvolutionResumeSupported(row, current)).toBe(["paused", "running", "queued"].includes(scenario));
    }
    expect(nativeEvolutionResumeSupported(null, current)).toBe(false);
  } finally { db.close(); await r.cleanup(); }
});


test("directed resume reconstructs persisted authorized material with independent validation and original protection", async () => {
  const r = await fixture();
  try {
    const cases = Array.from({ length: 8 }, (_, i) => ({ ...structuredClone(r.cohort.records.groups.replay_validation[0]), id: "product-ordinary-" + i, prompt: "protection " + i }));
    const original = { schema_version: 1, protocol: "original-core-protection-v1", identity: r.base.identity, passed: true,
      original_tensor_sha256: r.acceptedTensorSHA, cases, generation: cases.map(row => ({ id: row.id, answer: row.answer, passed: true, complete: true, branch_enabled: false })) };
    const source = { identity: r.base.identity, activeCore: null, originalTensorSHA256: r.acceptedTensorSHA, originalProtection: original };
    const initial = collectNativeEvolutionMaterial({ ...source, samples: Object.values(r.cohort.records.groups).flat() as any[], directedTeaching: r.cohort.records.directed_teaching, retentionRegressions: [] }).records;
    const body = { id: r.expected.id, kind: "evolution", scope: "product", binding: { ...r.expected.binding, scope: "product", records_sha256: materialContentSHA(initial) },
      checkpoint: { name: "checkpoint-225.pt" }, records: initial, coordinator: { native_preparation: { ...r.binding, records: initial } } };
    const journal = { id: body.id, identity: r.base.identity, kind: "evolution", state: "paused", body: JSON.stringify(body) };
    const material = nativeEvolutionResumeMaterial(journal, r.base.identity)!;
    expect(collectNativeEvolutionMaterial({ ...source, ...material }).records).toEqual(initial);
    expect(material.samples.filter(row => row.purpose === "validation").map(row => row.id)).toEqual(Object.values(initial.groups).flat().filter(row => row.purpose === "validation").map(row => row.id));
    material.samples[0]!.answer = "changed";
    expect(nativeEvolutionResumeMaterial(journal, r.base.identity)!.samples[0]!.answer).not.toBe("changed");
    const bad = structuredClone(body); bad.records.directed_teaching.authorization.explicit = false;
    expect(() => nativeEvolutionResumeMaterial({ ...journal, body: JSON.stringify(bad) }, r.base.identity)).toThrow();
    const changed = structuredClone(body); changed.records.groups.new_training[0]!.answer = "林浩";
    expect(() => nativeEvolutionResumeMaterial({ ...journal, body: JSON.stringify(changed) }, r.base.identity)).toThrow();
  } finally { await r.cleanup(); }
});
