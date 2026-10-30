/** Relocate verified C++/LibTorch copies; never edit or recompile their sources. */
import { createReadStream } from "node:fs";
import { chmod, copyFile, lstat, mkdir, readFile, realpath, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve, sep } from "node:path";

import { NAME, PYTHON, parseImage, system, relativeName, dependencyClosure as boundDependencyClosure, type Binding, type Image } from "../backend/native-runtime-macho.ts";
export { parseImage } from "../backend/native-runtime-macho.ts";

const ROOT = resolve(import.meta.dir, "..");
const SHA = /^[a-f0-9]{64}$/;
function need(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }

export async function digest(path: string) {
  const hash = new Bun.CryptoHasher("sha256");
  for await (const block of createReadStream(path)) hash.update(block as Buffer);
  return hash.digest("hex");
}
async function canonical(path: string) {
  const value = resolve(path);
  need(await realpath(value) === value && (await lstat(value)).isFile(), "Symlink/non-file source rejected");
  return value;
}
export async function freshOutput(path: string) {
  need(isAbsolute(path), "Output must be absolute");
  const value = resolve(path);
  need(path === value, "Escaping/noncanonical output rejected");
  need(await realpath(dirname(value)) === dirname(value), "Output parent is noncanonical");
  try { await lstat(value); } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return value;
    throw error;
  }
  throw new Error("Fresh output required; existing files are never overwritten");
}
export function cleanRuntimeEnvironment() {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined && !/^(PYTHON|DYLD_)/.test(key) && !["VIRTUAL_ENV", "CONDA_PREFIX", "LUBAN_MODEL_LOCK_FD"].includes(key)) env[key] = value;
  }
  env.PYTORCH_ENABLE_MPS_FALLBACK = "0";
  return env;
}
async function command(argv: string[], env = cleanRuntimeEnvironment()) {
  const start = performance.now();
  const process = Bun.spawn(argv, { cwd: ROOT, env, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exit_code] = await Promise.all([
    new Response(process.stdout).text(), new Response(process.stderr).text(), process.exited,
  ]);
  return { argv, pid: process.pid, stdout, stderr, exit_code, wall_seconds: (performance.now() - start) / 1000 };
}
async function must(argv: string[]) {
  const value = await command(argv);
  need(value.exit_code === 0, `Native tool failed: ${argv[0]} ${value.stderr}`);
  return value;
}
async function inspect(path: string) {
  const [link, loads] = await Promise.all([must(["/usr/bin/otool", "-L", path]), must(["/usr/bin/otool", "-l", path])]);
  return { ...parseImage(link.stdout, loads.stdout), linkage: link.stdout, loads: loads.stdout };
}
export function dependencyClosure(binary: string, bindings: Record<string, Binding>, inspectSource: (path: string) => Promise<Image> = inspect) {
  return boundDependencyClosure(binary, bindings, inspectSource);
}

export async function verifyBuild(binaryInput: string, receiptInput: string, sourceInput: string) {
  const binary = await canonical(binaryInput), receiptPath = await canonical(receiptInput), source = await canonical(sourceInput);
  const receipt = JSON.parse(await readFile(receiptPath, "utf8"));
  need(SHA.test(receipt.binary_sha256) && await digest(binary) === receipt.binary_sha256, "Source binary receipt SHA mismatch");
  need(SHA.test(receipt.source_sha256) && await digest(source) === receipt.source_sha256, "Source C++ receipt SHA mismatch");
  need(receipt.no_python_linkage === true && receipt.release_ce_graph === true && receipt.reuse_replay_forward === true
    && receipt.model_lock_inheritance === true && receipt.variant === "release-ce-graph", "Approved CE-release build receipt required");
  need(receipt.actual_compile?.exit_code === 0 && Array.isArray(receipt.actual_compile?.argv)
    && receipt.actual_compile.argv.includes("-DLUBAN_RELEASE_CE_GRAPH=1") && receipt.actual_compile.argv.includes("-DLUBAN_REUSE_REPLAY_FORWARD=1"), "Compile receipt flags mismatch");
  const bindings = receipt.torch_library_bindings as Record<string, Binding>;
  need(bindings && typeof bindings === "object" && !Array.isArray(bindings) && Object.keys(bindings).length > 0, "Native source library bindings required");
  for (const [path, binding] of Object.entries(bindings)) {
    need(isAbsolute(path) && resolve(path) === path && NAME.test(basename(path)) && !PYTHON.test(path), "Invalid native library binding path");
    await canonical(path);
    need(SHA.test(binding.sha256) && Number.isSafeInteger(binding.bytes) && binding.bytes > 0
      && (await stat(path)).size === binding.bytes && await digest(path) === binding.sha256, "Source library receipt SHA/bytes mismatch");
  }
  return { binary, source, receiptPath, receipt, bindings };
}

export async function packageNativeRuntime(outputInput: string, fixture?: string) {
  const output = await freshOutput(outputInput);
  const verified = await verifyBuild(join(ROOT, "build/native/luban-training"), join(ROOT, "build/native/luban-training.build.json"), join(ROOT, "native/Training.cpp"));
  need(!output.startsWith(ROOT + sep), "Runtime candidate cannot overwrite project payload");
  const images = await dependencyClosure(verified.binary, verified.bindings);
  const torchRoot = dirname(dirname(Object.keys(verified.bindings)[0]));
  const metadataRoot = join(dirname(torchRoot), "torch-2.8.0.dist-info", "licenses");
  const licenseSources = [join(metadataRoot, "LICENSE"), join(metadataRoot, "NOTICE")];
  for (const path of licenseSources) await canonical(path);
  await mkdir(output, { mode: 0o700 });
  const mutations = [], copied: Record<string, { source: string; original: Binding; final: Binding }> = {};
  try {
    await mkdir(join(output, "lib")); await mkdir(join(output, "licenses"));
    await copyFile(verified.receiptPath, join(output, "source-build-receipt.json"));
    for (const source of Object.keys(images)) {
      const relative = source === verified.binary ? "luban-training" : "lib/" + relativeName(basename(source));
      const target = join(output, relative);
      const original = { sha256: await digest(source), bytes: (await stat(source)).size };
      await copyFile(source, target); await chmod(target, 0o755);
      need(await digest(target) === original.sha256, "Copied native bytes differ");
      const image = images[source];
      const options: string[] = [];
      for (const dependency of image.dependencies) if (!system(dependency)) options.push("-change", dependency, "@rpath/" + basename(dependency));
      if (image.id) options.push("-id", "@rpath/" + basename(source));
      const desiredRpath = source === verified.binary ? "@executable_path/lib" : "@loader_path";
      for (const path of image.rpaths) if (path !== desiredRpath) options.push("-delete_rpath", path);
      if (!image.rpaths.includes(desiredRpath)) options.push("-add_rpath", desiredRpath);
      mutations.push(await must(["/usr/bin/install_name_tool", ...options, target]));
      mutations.push(await must(["/usr/bin/codesign", "--force", "--sign", "-", target]));
      await must(["/usr/bin/codesign", "--verify", "--strict", target]);
      copied[relative] = { source, original, final: { sha256: await digest(target), bytes: (await stat(target)).size } };
    }
    const licenses: Record<string, Binding> = {};
    for (const source of licenseSources) {
      const relative = "licenses/" + basename(source);
      await copyFile(source, join(output, relative));
      licenses[relative] = { sha256: await digest(source), bytes: (await stat(source)).size };
      need(await digest(join(output, relative)) === licenses[relative].sha256, "License readback failed");
    }
    const relocatedBindings = Object.fromEntries(Object.entries(copied).filter(([path]) => path.startsWith("lib/")).map(([path, item]) => [join(output, path), item.final]));
    const relocated = await dependencyClosure(join(output, "luban-training"), relocatedBindings);
    for (const [path, image] of Object.entries(relocated)) {
      need(image.rpaths.every(value => value === (path.endsWith("/luban-training") ? "@executable_path/lib" : "@loader_path")), "Absolute/unexpected relocated rpath");
      need(image.dependencies.every(value => system(value) || /^@rpath\/lib[a-zA-Z0-9_+.-]+\.dylib$/.test(value)), "Absolute/unexpected relocated dependency");
    }
    const probe = await command([join(output, "luban-training"), "--help"], { ...cleanRuntimeEnvironment(), DYLD_PRINT_LIBRARIES: "1" });
    await writeFile(join(output, "load-probe.json"), JSON.stringify(probe, null, 2) + "\n");
    need(probe.exit_code === 1 && probe.stderr.includes("Native training rejected: Usage:"), "Relocated binary did not reach its native argument validator");
    need(!PYTHON.test(probe.stderr), "Python image dynamically loaded");
    for (const path of Object.keys(relocatedBindings)) need(probe.stderr.includes(path), `Candidate native library not actually loaded: ${path}`);
    need(!probe.stderr.includes("site-packages") && !probe.stderr.includes(".venv/"), "Dynamic loader escaped relocated candidate");
    let cpu: Awaited<ReturnType<typeof cpuCheck>> | undefined;
    if (fixture) cpu = await cpuCheck(output, verified.binary, fixture);
    await verifyBuild(verified.binary, verified.receiptPath, verified.source);
    for (const [path, item] of Object.entries(copied)) need(await digest(join(output, path)) === item.final.sha256, "Runtime payload changed after execution");
    const result = { schema: "luban-native-runtime-package-v1", created_at: new Date().toISOString(), output,
      source_build_receipt_sha256: await digest(verified.receiptPath), source_build_receipt: "source-build-receipt.json",
      source_cpp_sha256: verified.receipt.source_sha256, payload: copied, licenses,
      payload_bytes: Object.values(copied).reduce((sum, value) => sum + value.final.bytes, 0),
      license_bytes: Object.values(licenses).reduce((sum, value) => sum + value.bytes, 0),
      relocation_commands: mutations, recursive_linkage: relocated, no_python_linkage: true,
      dynamic_load_probe: probe, python_process_invocations: 0, gpu_training_executed: false,
      source_preserved: true, cpu, product_installed: false,
      scope: "Relocated C++ optimization executor only; Python preparation, router/expert/ASR/quality/adoption not migrated by this package" };
    await writeFile(join(output, "runtime-receipt.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
    return result;
  } catch (error) {
    await writeFile(join(output, "package-failure.json"), JSON.stringify({ error: String(error), product_installed: false, source_modified: false }) + "\n", { flag: "wx" });
    throw error;
  }
}

async function cpuCheck(output: string, sourceBinary: string, fixtureInput: string) {
  const fixture = await realpath(fixtureInput), manifestPath = join(fixture, "manifest.tsv");
  const manifest = await readFile(manifestPath, "utf8");
  need(/^config\tdevice\tcpu$/m.test(manifest) && /^config\tsteps\t2$/m.test(manifest), "Only existing CPU two-step fixture may be exercised");
  const originalSHA = await digest(manifestPath);
  const inputHashes: Record<string, string> = {};
  for (const line of manifest.split("\n")) {
    const fields = line.split("\t");
    if (fields[0] !== "tensor") continue;
    need(/^[a-z][a-z0-9_]*\.raw$/.test(fields[4]), "CPU fixture path escapes");
    const path = await canonical(join(fixture, fields[4]));
    need(await digest(path) === fields[6] && (await stat(path)).size === Number(fields[5]), "CPU fixture SHA/bytes mismatch");
    inputHashes[path] = fields[6];
  }
  const commands = [], arms: Record<string, unknown> = {};
  const binaries = { source: sourceBinary, packaged: join(output, "luban-training") };
  await mkdir(join(output, "cpu-checks"));
  for (const [arm, binary] of Object.entries(binaries)) {
    const parent = join(output, "cpu-checks", arm); await mkdir(parent);
    const first = join(parent, "first2");
    const firstRun = await command([binary, "--input", fixture, "--output", first, "--manifest-sha", originalSHA]);
    commands.push(firstRun); need(firstRun.exit_code === 0, `CPU first2 failed: ${firstRun.stderr}`);
    const continuation = join(parent, "continuation-input"); await mkdir(continuation);
    const files = new Map<string, string>([["candidate", "final_candidate.raw"], ["master", "final_master.raw"], ["exp_avg", "final_exp_avg.raw"], ["exp_avg_sq", "final_exp_avg_sq.raw"], ["adam_step", "final_adam_step.raw"]]);
    const lines = [];
    for (const line of manifest.trimEnd().split("\n")) {
      const fields = line.split("\t");
      if (fields[0] === "tensor") {
        const source = files.has(fields[1]) ? join(first, files.get(fields[1])!) : join(fixture, fields[4]);
        await copyFile(source, join(continuation, fields[4]));
        fields[6] = await digest(source);
      }
      lines.push(fields.join("\t"));
    }
    lines.splice(1, 0, "config\ttraining_offset\t2");
    await writeFile(join(continuation, "manifest.tsv"), lines.join("\n") + "\n");
    const second = join(parent, "next2");
    const secondRun = await command([binary, "--input", continuation, "--output", second, "--manifest-sha", await digest(join(continuation, "manifest.tsv"))]);
    commands.push(secondRun); need(secondRun.exit_code === 0, `CPU next2 failed: ${secondRun.stderr}`);
    const firstResult = JSON.parse(await readFile(join(first, "result.json"), "utf8"));
    const secondResult = JSON.parse(await readFile(join(second, "result.json"), "utf8"));
    need(firstResult.final_adam_step === 2 && secondResult.entry_adam_step === 2 && secondResult.final_adam_step === 4
      && firstResult.device === "cpu" && secondResult.device === "cpu", "CPU Adam continuation/device differs");
    arms[arm] = { first: firstResult, continuation: secondResult };
  }
  const comparisons = [];
  for (const phase of ["first2", "next2"]) {
    const text = await readFile(join(output, "cpu-checks/source", phase, "tensors.tsv"), "utf8");
    const other = await readFile(join(output, "cpu-checks/packaged", phase, "tensors.tsv"), "utf8");
    need(text === other, "Relocated CPU output tensor manifest differs");
    for (const line of text.trimEnd().split("\n").slice(1)) {
      const fields = line.split("\t");
      const name = fields.find(value => /^[a-z][a-z0-9_]*\.raw$/.test(value));
      need(name, "CPU output raw tensor name missing");
      const sourceSHA = await digest(join(output, "cpu-checks/source", phase, name));
      const candidateSHA = await digest(join(output, "cpu-checks/packaged", phase, name));
      need(sourceSHA === candidateSHA, "Relocated CPU raw tensor differs");
      comparisons.push({ phase, file: name, sha256: sourceSHA, raw_equal: true });
    }
  }
  need(await digest(manifestPath) === originalSHA, "Original CPU fixture manifest changed");
  for (const [path, hash] of Object.entries(inputHashes)) need(await digest(path) === hash, "Original CPU fixture raw changed");
  return { scope: "Same C++ binary before/after relocation, tiny CPU FP32/BF16 fixture; no real 2B quality/equivalence or GPU claim", fixture, manifest_sha256: originalSHA,
    input_hashes_preserved: true, cpu2plus2_passed: true, source_vs_package_all_raw_equal: true,
    comparisons, commands, arms, python_process_invocations: 0, teacher_calls: 0, product_adopted: false };
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  need(args.length === 2 || args.length === 4, "Usage: --output fresh-external-dir [--fixture existing-cpu2-dir]");
  need(args[0] === "--output" && (args.length === 2 || args[2] === "--fixture"), "Unexpected runtime packaging arguments");
  need(args[1].startsWith("/Volumes/移动数据库/lubanllm/experiments/"), "Candidate must be a fresh external experiment directory");
  need((await stat("/Volumes/移动数据库")).isDirectory(), "External model volume unavailable");
  const result = await packageNativeRuntime(args[1], args[3]);
  console.log(JSON.stringify({ output: result.output, payload_bytes: result.payload_bytes, license_bytes: result.license_bytes, cpu2plus2_passed: result.cpu?.cpu2plus2_passed ?? false }));
}
