import { afterAll, beforeAll, expect, test } from "bun:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, open, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { patchLegacyExpertCollector, patchLegacyExpertValidation } from "../scripts/native-evolution-legacy-expert-patches";

const root = resolve(import.meta.dir, ".."), research = join(root, ".native-migration/research");
const sha = (bytes: Uint8Array | string) => createHash("sha256").update(bytes).digest("hex");
const id = "8ec555e81b20", expert = "learned-" + id, reportName = "learning-" + id + ".json";
let work: string, api: any;
const base = { identity: "a".repeat(64), model_manifest_sha256: "b".repeat(64), config_sha256: "c".repeat(64),
  checkpoint_sha256: "d".repeat(64), tokenizer_sha256: "e".repeat(64), original_down_sha256: "f".repeat(64),
  engine_binary_sha256: "0".repeat(64), core_parameter: "model.layers.23.mlp.down_proj.weight", core_shape: [2, 4], dtype: "BF16", device: "mps" };
const json = (value: unknown) => Buffer.from(JSON.stringify(value));
function tensors(shapes: Record<string, number[]>, nonfinite = false) {
  let offset = 0; const header: Record<string, unknown> = {};
  for (const [name, shape] of Object.entries(shapes)) {
    const bytes = shape.reduce((a, b) => a * b, 2);
    header[name] = { dtype: "BF16", shape, data_offsets: [offset, offset + bytes] }; offset += bytes;
  }
  const head = json(header), prefix = Buffer.alloc(8), data = Buffer.alloc(offset);
  prefix.writeBigUInt64LE(BigInt(head.length)); if (nonfinite) data.writeUInt16LE(0x7f80, 0);
  return Buffer.concat([prefix, head, data]);
}
function fixture(nonfinite = false) {
  const weights = tensors({ "gate.weight": [32, 2], "up.weight": [32, 2], "down.weight": [2, 32] }, nonfinite);
  const routerWeights = tensors({ "projection.weight": [3, 2], "projection.bias": [3] }), routerID = "router-112233445566";
  const routerMeta = { id: routerID, identity: base.identity, sha256: sha(routerWeights), core_version: null,
    calibrated: true, gates_expert: true, expert_id: expert, expert_sha256: sha(weights), supported_actions: ["local_core", "neural_expert", "teacher"] };
  return { "active.json": json({ id: expert, width: 32, identity: base.identity, core_version: null }),
    [`layer-23/${expert}.json`]: json({ identity: base.identity + ":layer-23", hidden: 2, width: 32, sha256: sha(weights), requires_gate: true, core_version: null }),
    [`layer-23/${expert}.safetensors`]: weights, [`${routerID}.safetensors`]: routerWeights, [`${routerID}.json`]: json(routerMeta),
    "active-router.json": json({ id: routerID, identity: base.identity, sha256: routerMeta.sha256, metadata_sha256: sha(json(routerMeta)), core_version: null }) };
}
async function store(files: Record<string, Buffer>) {
  const directory = await mkdtemp(join(work, "store-"));
  await mkdir(join(directory, "layer-23"));
  for (const [name, bytes] of Object.entries(files)) await writeFile(join(directory, name), bytes);
  return directory;
}
beforeAll(async () => {
  work = await realpath(await mkdtemp(join(tmpdir(), "luban-legacy-expert-CPU-")));
  base.identity = sha(work);
  const entry = join(work, "entry.ts"), product = join(research, "native-product-runtime/backend");
  await writeFile(entry, `export {collectLearningModelFiles} from ${JSON.stringify(join(research, "native-learning-product/model-files.ts"))};
export {validateModelFiles} from ${JSON.stringify(join(product, "evolution-snapshot-files.ts"))};
export {prepareLearningCommitStorage,nativeAcceptedHashes} from ${JSON.stringify(join(research, "native-learning-product/commit-storage.ts"))};
export {Memory} from ${JSON.stringify(join(product, "memory.ts"))};
export {encoded} from ${JSON.stringify(join(product, "canonical-json.ts"))};`);
  const replace = (body: string, before: string, after: string) => { assert.equal(body.split(before).length, 2); return body.replace(before, after); };
  const built = await Bun.build({ entrypoints: [entry], target: "bun", plugins: [{ name: "legacy-expert-compatibility", setup(builder) {
    builder.onLoad({ filter: /\/(model-files|evolution-snapshot-files)\.ts$/ }, async args => {
      const body = await readFile(args.path, "utf8");
      return { loader: "ts", contents: args.path.endsWith("/model-files.ts") ? patchLegacyExpertCollector(body, replace) : patchLegacyExpertValidation(body, replace) };
    });
  } }] });
  expect(built.success, built.logs.map(String).join("\n")).toBe(true);
  const path = join(work, "fixture.js"); await writeFile(path, Buffer.from(await built.outputs[0]!.arrayBuffer())); api = await import(path);
});
afterAll(async () => { if (work) await rm(work, { recursive: true, force: true }); });

test("persisted gated expert and calibrated router remain physically present without a historical report", async () => {
  const files = fixture(), directory = await store(files), collected = api.collectLearningModelFiles(directory);
  expect(Object.keys(collected).sort()).toEqual(Object.keys(files).sort());
  const result = api.validateModelFiles(collected, base);
  expect(result.expert.id).toBe(expert); expect(result.expert.file_sha256).toBe(sha(files[`layer-23/${expert}.safetensors`]!));
  expect(result.router.id).toBe("router-112233445566");
});
test("present reports still require valid JSON, unique promotion and the accepted expert binding", async () => {
  for (const report of [Buffer.from("{"), json({ promoted: false, active_expert: expert }), json({ promoted: true, active_expert: "learned-000000000000" })]) {
    const files = api.collectLearningModelFiles(await store({ ...fixture(), [reportName]: report }));
    expect(() => api.validateModelFiles(files, base)).toThrow();
  }
  const files: Record<string, Buffer> = { ...fixture(), [reportName]: json({ promoted: true, active_expert: expert }) };
  expect(api.validateModelFiles(files, base).expert.id).toBe(expert);
  files["learning-000000000000.json"] = files[reportName]!;
  expect(() => api.validateModelFiles(files, base)).toThrow("不唯一");
});
test("optional report only tolerates ENOENT, not links or wrong file types", async () => {
  const directory = await store(fixture()); await mkdir(join(directory, reportName));
  expect(() => api.collectLearningModelFiles(directory)).toThrow();
  await rm(join(directory, reportName), { recursive: true }); await symlink("active.json", join(directory, reportName));
  expect(() => api.collectLearningModelFiles(directory)).toThrow("符号链接");
});
test("absent report does not weaken expert identity, bytes, shape, finite values or router bindings", () => {
  const files = fixture(), metaName = `layer-23/${expert}.json`, weightName = `layer-23/${expert}.safetensors`;
  const changed = (name: string, patch: Record<string, unknown>) => ({ ...files, [name]: json({ ...JSON.parse(files[name]!.toString()), ...patch }) });
  for (const invalid of [changed("active.json", { identity: "b".repeat(64) }), changed(metaName, { identity: "wrong:layer-23" }),
    changed(metaName, { hidden: 3 }), { ...files, [weightName]: Buffer.alloc(files[weightName]!.length) }, fixture(true),
    changed("active-router.json", { metadata_sha256: "a".repeat(64) })]) expect(() => api.validateModelFiles(invalid, base)).toThrow();
});
test("new Learning commit still refuses absent report artifacts before any pointer or journal change", async () => {
  const files = fixture(), directory = await store(files), memory = new api.Memory(work), lockDirectory = `/tmp/luban-model-locks-${process.getuid!()}`;
  const localID = "123456789abc"; memory.model_identity = base.identity;
  await mkdir(lockDirectory, { recursive: true, mode: 0o700 });
  const actualLockPath = join(lockDirectory, `${base.identity}-cpu.lock`);
  // This fixture identity has its own exclusive CPU lock and never opens a model.
  const lock = await open(actualLockPath, "wx+", 0o600);
  try {
    const records = {}, binding = { identity: base.identity, scope: "product", device: "cpu", arguments: {}, core_version: null,
      active_files_sha256: api.nativeAcceptedHashes(directory), records_sha256: sha(api.encoded(records)) };
    const body = { id: localID, kind: "learning", scope: "product", binding, arguments: {}, records, started: Date.now() / 1000 };
    memory.connect((db: any) => {
      db.exec("CREATE TABLE training_runs(id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)");
      db.query("INSERT INTO training_runs VALUES(?,?,?,?,?,?)").run(localID, base.identity, "learning", "running", api.encoded(body), 1);
    });
    const request = { memory, store: directory, base, guard: { modelLockFd: lock.fd, assertQuiescent() {} }, expected: { id: localID, binding } };
    const next = "learned-" + localID, pointer = { id: next, identity: base.identity, width: 32, core_version: null };
    const nextNames = [`layer-23/${next}.json`, `layer-23/${next}.safetensors`];
    await writeFile(join(directory, nextNames[0]!), files[`layer-23/${expert}.json`]!); await writeFile(join(directory, nextNames[1]!), files[`layer-23/${expert}.safetensors`]!);
    const report = { promoted: true, id: localID, active_expert: next, base_identity: base.identity, records_sha256: binding.records_sha256, chosen_width: 32 };
    expect(() => api.prepareLearningCommitStorage(request, { report, pointer, artifacts: nextNames })).toThrow("three complete Learning artifacts");
    expect(() => api.prepareLearningCommitStorage(request, { report, pointer, artifacts: [...nextNames, `learning-${localID}.json`] })).toThrow("ENOENT");
    expect(JSON.parse((await readFile(join(directory, "active.json"))).toString()).id).toBe(expert);
    expect(memory.connect((db: any) => db.query("SELECT state FROM training_runs WHERE id=?").get(localID)).state).toBe("running");
  } finally { memory.close(); await lock.close(); await rm(actualLockPath); }
});
