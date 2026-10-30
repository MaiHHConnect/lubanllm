import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cleanRuntimeEnvironment, dependencyClosure, digest, freshOutput, parseImage, verifyBuild } from "../scripts/package-native-runtime";

let directory: string;
const binding = { sha256: "a".repeat(64), bytes: 4 };
beforeEach(async () => { directory = await realpath(await mkdtemp(join(tmpdir(), "luban-runtime-package-"))); });
afterEach(async () => { await rm(directory, { recursive: true, force: true }); });

test("fresh output accepts missing child and refuses repeated output without changing it", async () => {
  const path = join(directory, "candidate");
  expect(await freshOutput(path)).toBe(path);
  await writeFile(path, "keep");
  await expect(freshOutput(path)).rejects.toThrow("Fresh output");
  expect(await readFile(path, "utf8")).toBe("keep");
});
test("rejects escaping, relative and symlink output", async () => {
  await expect(freshOutput(directory + "/../escape")).rejects.toThrow("Escaping");
  await expect(freshOutput("candidate")).rejects.toThrow("absolute");
  await symlink(directory, join(directory, "alias"));
  await expect(freshOutput(join(directory, "alias", "candidate"))).rejects.toThrow("noncanonical");
});
test("parses dylib self ID separately from recursive dependencies and rpaths", () => {
  const image = parseImage("x:\n\t@rpath/libtorch.dylib (compatibility version 0.0.0, current version 0.0.0)\n\t@rpath/libc10.dylib (compatibility version 0.0.0, current version 0.0.0)\n",
    "cmd LC_ID_DYLIB\n cmdsize 48\n name @rpath/libtorch.dylib (offset 24)\ncmd LC_RPATH\n cmdsize 32\n path @loader_path (offset 12)\n");
  expect(image.dependencies).toEqual(["@rpath/libc10.dylib"]);
  expect(image.rpaths).toEqual(["@loader_path"]);
});
test("recursively closes dependencies using library and executable rpaths, with cycles", async () => {
  const executable = "/candidate/luban-training", torch = "/candidate/lib/libtorch.dylib", c10 = "/candidate/lib/libc10.dylib";
  const images = {
    [executable]: { id: null, rpaths: ["@executable_path/lib"], dependencies: ["@rpath/libtorch.dylib", "/usr/lib/libSystem.B.dylib"] },
    [torch]: { id: "@rpath/libtorch.dylib", rpaths: [], dependencies: ["@rpath/libc10.dylib"] },
    [c10]: { id: "@rpath/libc10.dylib", rpaths: ["@loader_path"], dependencies: ["@rpath/libtorch.dylib"] },
  };
  const closed = await dependencyClosure(executable, { [torch]: binding, [c10]: binding }, async path => images[path]);
  expect(Object.keys(closed)).toEqual([executable, torch, c10]);
});
test("unresolved/unbound non-system dylib rejects rather than relying on host fallback", async () => {
  for (const dependency of ["@rpath/libmissing.dylib", "/opt/homebrew/lib/libc10.dylib", "@loader_path/libmissing.dylib"]) {
    await expect(dependencyClosure("/candidate/luban-training", {}, async () => ({ id: null, rpaths: [], dependencies: [dependency] }))).rejects.toThrow("Unresolved/unbound");
  }
});
test("binding alone cannot make an unresolvable rpath valid", async () => {
  await expect(dependencyClosure("/candidate/luban-training", { "/foreign/libc10.dylib": binding }, async () => ({ id: null, rpaths: ["@executable_path/lib"], dependencies: ["@rpath/libc10.dylib"] }))).rejects.toThrow("Unresolved/unbound");
});
test("rejects escaping relative dependency, Python image and disguised system path", async () => {
  for (const dependency of ["@rpath/../libc10.dylib", "@rpath/libtorch_python.dylib", "/usr/lib/../../foreign.dylib"]) {
    await expect(dependencyClosure("/candidate/luban-training", {}, async () => ({ id: null, rpaths: [], dependencies: [dependency] }))).rejects.toThrow();
  }
});
async function fixtureReceipt() {
  const binary = join(directory, "luban-training"), source = join(directory, "Training.cpp"), library = join(directory, "libtorch.dylib"), receipt = join(directory, "build.json");
  await writeFile(binary, "binary"); await writeFile(source, "source"); await writeFile(library, "library");
  const value = { binary_sha256: await digest(binary), source_sha256: await digest(source), variant: "release-ce-graph",
    no_python_linkage: true, release_ce_graph: true, reuse_replay_forward: true, model_lock_inheritance: true,
    actual_compile: { exit_code: 0, argv: ["-DLUBAN_RELEASE_CE_GRAPH=1", "-DLUBAN_REUSE_REPLAY_FORWARD=1"] },
    torch_library_bindings: { [library]: { sha256: await digest(library), bytes: 7 } } };
  await writeFile(receipt, JSON.stringify(value));
  return { binary, source, library, receipt, value };
}
test("verifies binary, C++ source and every native library against original build receipt", async () => {
  const f = await fixtureReceipt();
  const result = await verifyBuild(f.binary, f.receipt, f.source);
  expect(result.bindings[f.library].bytes).toBe(7);
  await writeFile(f.library, "changed");
  await expect(verifyBuild(f.binary, f.receipt, f.source)).rejects.toThrow("library receipt");
});
test("rejects corrupted binary, C++ source and falsely approved build flags", async () => {
  const f = await fixtureReceipt();
  await writeFile(f.binary, "broken");
  await expect(verifyBuild(f.binary, f.receipt, f.source)).rejects.toThrow("binary receipt");
  await writeFile(f.binary, "binary"); await writeFile(f.source, "changed");
  await expect(verifyBuild(f.binary, f.receipt, f.source)).rejects.toThrow("C++ receipt");
  await writeFile(f.source, "source"); f.value.actual_compile.argv = [];
  await writeFile(f.receipt, JSON.stringify(f.value));
  await expect(verifyBuild(f.binary, f.receipt, f.source)).rejects.toThrow("flags mismatch");
});
test("refuses symlink library bound by a source receipt", async () => {
  const f = await fixtureReceipt();
  await rm(f.library); await symlink(f.binary, f.library);
  await expect(verifyBuild(f.binary, f.receipt, f.source)).rejects.toThrow("Symlink");
});
test("removes Python and dyld override variables before actual candidate launch", () => {
  const keys = ["PYTHONPATH", "VIRTUAL_ENV", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "LUBAN_MODEL_LOCK_FD"];
  const prior = keys.map(key => process.env[key]);
  try {
    for (const key of keys) process.env[key] = "do-not-use";
    const env = cleanRuntimeEnvironment();
    for (const key of keys) expect(env[key]).toBeUndefined();
    expect(env.PYTORCH_ENABLE_MPS_FALLBACK).toBe("0");
  } finally { keys.forEach((key, index) => { if (prior[index] === undefined) delete process.env[key]; else process.env[key] = prior[index]; }); }
});
