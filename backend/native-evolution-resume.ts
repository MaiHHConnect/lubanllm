/** Reopen the original round's durable inputs. Never bootstrap, recapture tails, seed, or save another previous version. */
import assert from "node:assert/strict";
import { lstat, realpath, readdir } from "node:fs/promises";
import { isAbsolute, join, normalize } from "node:path";
import { encoded } from "./canonical-json.ts";
import { ledgerParse } from "./memory.ts";
import { materialContentSHA, validateNativeProductRecords, validateNativeCoreRetention } from "./native-evolution-material.ts";
import { parseV3Fixture } from "../.native-migration/research/native-evolution-fixture/parser-readback.ts";
import { openNativeFile, unchangedNativeFile, readNativeSmallFile } from "../.native-migration/backend/native-features.ts";
import { checkpointRawSHA } from "../.native-migration/backend/native-checkpoint.ts";
import type { NativeCore450Owner } from "../.native-migration/research/native-core450-owner-controller/owner.ts";
import type { NativeFixtureRaw } from "../.native-migration/research/native-evolution-fixture/exporter.ts";

import type { NativeEvolutionMaterial } from "./native-evolution-cohort.ts";

type Row = Record<string, any>;
export type NativeEvolutionPreparationBinding = {
  schema: "luban-native-preparation-v1"; id: string; base: Row; records: Row; graphBinding: Row; qualification: string;
  fixture: { directory: string; manifest_sha256: string; config: Record<string, string>; tensors: Row; rows: Row[]; schedule: Row[] };
};
const sha = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const same = (a: unknown, b: unknown) => assert.equal(encoded(a), encoded(b));
const clone = <T>(value: T): T => ledgerParse(encoded(value));
const states = ["candidate", "master", "reference", "basis", "exp_avg", "exp_avg_sq", "adam_step"] as const;

/** Store this value in the original journal CAS that binds its existing previous snapshot. */
export function nativeEvolutionPreparationBinding(prepared: {
  id: string; base: Row; records: Row; graphBinding: Row; qualification: Buffer;
  fixture: NativeEvolutionPreparationBinding["fixture"];
}): NativeEvolutionPreparationBinding {
  const f = prepared.fixture;
  return clone({ schema: "luban-native-preparation-v1" as const, id: prepared.id, base: prepared.base, records: prepared.records,
    graphBinding: prepared.graphBinding, qualification: prepared.qualification.toString("base64"),
    fixture: { directory: f.directory, manifest_sha256: f.manifest_sha256, config: f.config, tensors: f.tensors, rows: f.rows, schedule: f.schedule } });
}

/** Reuse the exact durable human-authorized material; never draw replacement TRAIN/VALID from today's pool. */
export function nativeEvolutionResumeMaterial(row: unknown, identity: string): NativeEvolutionMaterial | null {
  assert(row && typeof row === "object", "原恢复任务不存在"); const journal = row as Row;
  assert(journal.identity === identity && journal.kind === "evolution" && ["paused", "running", "queued"].includes(journal.state));
  const body = typeof journal.body === "string" ? ledgerParse(journal.body) : journal.body;
  assert(body?.id === journal.id && /^[a-f0-9]{12}$/.test(body.id) && body.kind === "evolution" && body.scope === "product"
    && body.binding?.identity === identity && body.binding.scope === "product" && body.binding.arguments?.steps === 450
    && body.checkpoint && body.coordinator?.native_preparation?.schema === "luban-native-preparation-v1"
    && body.coordinator.native_preparation.id === body.id, "原任务不是可恢复的原生450进化");
  validateNativeProductRecords(body.records);
  assert(materialContentSHA(body.records) === body.binding.records_sha256, "原恢复材料与持久绑定不一致");
  same(body.coordinator.native_preparation.records, body.records);
  if (!body.records.directed_teaching) return null;
  return clone({ samples: Object.values(body.records.groups).flat() as Row[],
    directedTeaching: body.records.directed_teaching, retentionRegressions: body.records.retention_regressions });
}

/** Metadata-only UI eligibility. Actual resume still claims and validates the original checkpoint. */
export function nativeEvolutionResumeSupported(row: unknown, current: {
  identity: string; records: Row; core_version: string | null; implementation: Record<string, string>;
}): boolean {
  try {
    assert(row && typeof row === "object"); const journal = row as Row;
    assert(journal.identity === current.identity && journal.kind === "evolution" && ["paused", "running", "queued"].includes(journal.state));
    const body = typeof journal.body === "string" ? ledgerParse(journal.body) : journal.body;
    assert(body && /^[a-f0-9]{12}$/.test(body.id) && body.id === journal.id && body.kind === "evolution" && body.scope === "product");
    assert(body.checkpoint && typeof body.checkpoint.name === "string" && sha(body.checkpoint.sha256));
    const binding = body.binding, preparation = body.coordinator?.native_preparation;
    assert(binding?.identity === current.identity && binding.scope === "product" && binding.arguments?.steps === 450
      && body.arguments?.steps === 450 && binding.core_version === current.core_version);
    same(binding.implementation_sha256, current.implementation); same(body.records, current.records);
    assert(binding.records_sha256 === materialContentSHA(current.records));
    assert(preparation?.schema === "luban-native-preparation-v1" && preparation.id === body.id);
    same(preparation.records, body.records);
    assert(preparation.base?.identity === current.identity && preparation.graphBinding?.identity === current.identity
      && preparation.graphBinding.records_sha256 === binding.records_sha256
      && preparation.graphBinding.accepted_core_version === current.core_version);
    return true;
  } catch { return false; }
}

/** The producer must compare binding with its actual claimed journal before registering its private preparation capability. */
export async function reopenNativeEvolutionPreparation(request: {
  binding: NativeEvolutionPreparationBinding; base: Row; expected: { id: string; kind: string; binding: Row };
  cohort: { records: Row; records_sha256: string; assertUnchanged(): void };
  acceptedTensorSHA: string; trainingBinarySHA: string; owners: NativeCore450Owner;
  guard: { assertQuiescent(): void }; signal?: AbortSignal;
}) {
  const b = clone(request.binding), { base, expected, cohort, guard, signal } = request;
  assert(b.schema === "luban-native-preparation-v1" && /^[a-f0-9]{12}$/.test(b.id) && b.id === expected.id && expected.kind === "evolution");
  same(b.base, base); assert(base.identity === expected.binding.identity && expected.binding.arguments.steps === 450);
  assert(sha(request.acceptedTensorSHA) && sha(request.trainingBinarySHA));
  validateNativeProductRecords(b.records); same(b.records, cohort.records);
  assert(materialContentSHA(b.records) === expected.binding.records_sha256 && cohort.records_sha256 === expected.binding.records_sha256);
  const graph = b.graphBinding;
  same(graph.base, base); assert(graph.identity === base.identity && graph.records_sha256 === cohort.records_sha256);
  assert(graph.accepted_tensor_sha256 === request.acceptedTensorSHA && graph.accepted_core_version === expected.binding.core_version
    && graph.fixture_only === true && graph.core_last_down_outside_prefix === true && graph.branch_enabled === false
    && graph.expert === null && graph.use_cache === true && graph.cache_fresh_per_complete_sample === true && graph.product_admission === false && graph.quality_checked === false);
  assert(typeof b.qualification === "string" && Buffer.from(b.qualification, "base64").toString("base64") === b.qualification);
  const qualification = Buffer.from(b.qualification, "base64"), retained = ledgerParse(qualification.toString());
  validateNativeCoreRetention(retained, base.identity, "core-" + b.id);
  same(retained.groups, b.records.groups); same(retained.regression_validation, b.records.retention_regressions);
  const f = b.fixture, directory = f.directory;
  assert(isAbsolute(directory) && normalize(directory) === directory && !directory.includes("\0") && await realpath(directory) === directory);
  const directoryStat = await lstat(directory, { bigint: true }); assert(directoryStat.isDirectory() && !directoryStat.isSymbolicLink());
  assert(sha(f.manifest_sha256));
  const files: Array<{ path: string } & Awaited<ReturnType<typeof openNativeFile>>> = [];
  let closed = false;
  const close = async () => {
    if (closed) return;
    const results = await Promise.allSettled(files.map(async file => { if (file.handle.fd !== -1) await file.handle.close(); }));
    const failures = results.filter((r): r is PromiseRejectedResult => r.status === "rejected");
    if (failures.length) throw new AggregateError(failures.map(r => r.reason), "原轮次输入FD尚未关闭，保留所有者");
    closed = true;
  };
  const owner = { close }; request.owners.own(owner, close);
  const scope = () => { assert(!closed); signal?.throwIfAborted(); guard.assertQuiescent(); cohort.assertUnchanged(); };
  async function bind(name: string, bytes: number) {
    scope(); const path = join(directory, name), held = await openNativeFile(path, bytes); files.push({ path, ...held });
    assert(held.stat.nlink === 1n); return { path, ...held };
  }
  try {
    scope();
    const manifestSize = Number((await lstat(join(directory, "manifest.tsv"))).size);
    assert(manifestSize > 0 && manifestSize <= (16 << 20));
    const manifestOwner = await bind("manifest.tsv", manifestSize), manifest = (await readNativeSmallFile(manifestOwner.handle, manifestSize)).toString();
    assert.equal(Bun.CryptoHasher.hash("sha256", manifest, "hex"), f.manifest_sha256);
    const fixture = parseV3Fixture(manifest, { input: directory, output: join(directory, "never-execute"),
      manifest_sha256: f.manifest_sha256, binary_sha256: request.trainingBinarySHA, identity: base.identity, device: "mps",
      core_shape: base.core_shape, reference_sha256: f.tensors.reference.sha256, basis_sha256: graph.basis.sha256 },
      { cpu_sha256: f.tensors.cpu_rng.sha256, mps_sha256: f.tensors.mps_rng.sha256 });
    same(fixture.config, f.config); same(fixture.steps, f.schedule);
    assert(fixture.steps.length === 450 && Number(fixture.config.training_offset ?? "0") === 0);
    assert(fixture.tensors.candidate!.sha256 === request.acceptedTensorSHA);
    const training = [...b.records.groups.new_training, ...b.records.groups.replay_training];
    assert(f.rows.length === training.length && Object.keys(fixture.rows).length === training.length);
    for (const [index, sample] of training.entries()) {
      const row = f.rows[index]!, role = index < b.records.groups.new_training.length ? "new" : "replay";
      assert(row.sample_id === sample.id && row.role === role && row.sample_sha256 === materialContentSHA(sample, true));
      same(fixture.rows[row.key], { role, sample_sha256: row.sample_sha256, length: row.length, gold: row.gold });
    }
    same(Object.keys(fixture.tensors).sort(), Object.keys(f.tensors).sort());
    const raw: Record<string, NativeFixtureRaw> = {}, tensorOwners = new Map<string, Awaited<ReturnType<typeof bind>> & { sha256: string }>();
    for (const [name, spec] of Object.entries(fixture.tensors)) {
      same(f.tensors[name], { dtype: spec.dtype, shape: spec.shape, file: spec.file, bytes: spec.bytes, sha256: spec.sha256 });
      const held = await bind(spec.file, spec.bytes);
      assert.equal(await checkpointRawSHA(held.handle, spec.bytes, spec.dtype), spec.sha256);
      await unchangedNativeFile(held.path, held.handle, held.stat);
      tensorOwners.set(name, { ...held, sha256: spec.sha256 });
      raw[name] = { path: held.path, handle: held.handle, dtype: spec.dtype, shape: [...spec.shape], bytes: spec.bytes,
        sha256: spec.sha256, assertUnchanged: async () => { scope(); } };
    }
    const inventory = ["fixture.json", "manifest.tsv", ...Object.values(fixture.tensors).map(t => t.file)].sort();
    same((await readdir(directory)).sort(), inventory);
    await unchangedNativeFile(manifestOwner.path, manifestOwner.handle, manifestOwner.stat); scope();
    const prepared = { id: b.id, base: b.base, records: b.records, graphBinding: graph, qualification,
      cohort, state: Object.fromEntries(states.map(name => [name, raw[name]!])),
      rng: { cpu_rng: raw.cpu_rng!, mps_rng: raw.mps_rng!, receipt: graph.preparation_rng ?? graph.fresh_rng },
      fixture: { ...f, dataset_sha256: materialContentSHA({ new: b.records.groups.new_training, replay: b.records.groups.replay_training }, true),
        tensorOwner(name: string) { scope(); const value = tensorOwners.get(name); assert(value, "原轮次缺少指定张量所有者"); return value; },
        assertUnchanged: async () => { scope(); }, close },
      resume: true as const, assertUnchanged: async () => { scope(); }, close,
      product_admission: false as const, quality_admitted: false as const };
    return { prepared, accepted: raw.candidate!, basis: raw.basis!, owner, closed: () => closed };
  } catch (error) {
    try { await request.owners.release(owner); } catch (cleanup) { throw new AggregateError([error, cleanup], "原轮次输入恢复及关闭均失败"); }
    throw error;
  }
}
