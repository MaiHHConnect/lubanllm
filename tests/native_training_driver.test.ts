import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cleanEnvironment, validateFixtureManifest } from "../scripts/native-training";

let directory: string;
const hash = (bytes: Uint8Array | string) => new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
const raw = new Uint8Array([0, 0, 128, 63]);
const record = (file = "master.raw") => `tensor\tmaster\tf32\t1\t${file}\t4\t${hash(raw)}`;
async function fixture(lines = [record()]) {
  const text = ["LUBAN_NATIVE_TRAINING_V1", ...lines].join("\n") + "\n";
  await writeFile(join(directory, "manifest.tsv"), text);
  return hash(text);
}
beforeEach(async () => {
  directory = await realpath(await mkdtemp(join(tmpdir(), "luban-native-input-")));
  await writeFile(join(directory, "master.raw"), raw);
});
afterEach(async () => { await rm(directory, { recursive: true, force: true }); });

test("validates actual fixture bytes without importing Torch or starting training", async () => {
  const result = await validateFixtureManifest(directory, await fixture());
  expect(result.tensors["master.raw"]).toEqual({ sha256: hash(raw), bytes: 4 });
  expect(await readFile(join(directory, "master.raw"))).toEqual(Buffer.from(raw));
});
test("rejects same-size modified tensor bytes", async () => {
  const approved = await fixture();
  await writeFile(join(directory, "master.raw"), new Uint8Array([0, 0, 0, 0]));
  await expect(validateFixtureManifest(directory, approved)).rejects.toThrow("SHA/bytes");
});
test("rejects an unapproved manifest", async () => {
  await fixture();
  await expect(validateFixtureManifest(directory, "0".repeat(64))).rejects.toThrow("manifest SHA");
});
test("rejects path traversal and repeated tensor files", async () => {
  await expect(validateFixtureManifest(directory, await fixture([record("../master.raw")]))).rejects.toThrow("raw fixture");
  await expect(validateFixtureManifest(directory, await fixture([record(), record()]))).rejects.toThrow("Repeated");
});
test("rejects symlink tensor inputs", async () => {
  await writeFile(join(directory, "outside.raw"), raw);
  await rm(join(directory, "master.raw"));
  await symlink(join(directory, "outside.raw"), join(directory, "master.raw"));
  await expect(validateFixtureManifest(directory, await fixture())).rejects.toThrow("Symlink");
});
test("removes interpreter configuration and disables silent MPS CPU fallback", () => {
  const keys = ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTORCH_ENABLE_MPS_FALLBACK"];
  const previous = keys.map(key => process.env[key]);
  try {
    for (const key of keys) process.env[key] = "fixture-only";
    const env = cleanEnvironment();
    for (const key of keys.slice(0, 4)) expect(env[key]).toBeUndefined();
    expect(env.PYTORCH_ENABLE_MPS_FALLBACK).toBe("0");
  } finally {
    keys.forEach((key, index) => { if (previous[index] === undefined) delete process.env[key]; else process.env[key] = previous[index]; });
  }
});
