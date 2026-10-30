/** Research-only compiler/launcher for actual C++ LibTorch training; never Python. */
import { createReadStream } from "node:fs";
import { lstat, mkdir, open, readFile, realpath, rename, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const SOURCE = join(ROOT, "native/Training.cpp");
const TORCH = join(ROOT, ".venv/lib/python3.12/site-packages/torch");
const ABI = "LUBAN_NATIVE_TRAINING_V1";
const SHA = /^[0-9a-f]{64}$/;
const fail = (message: string): never => { throw new Error(message); };
function need(value: unknown, message: string): asserts value { if (!value) fail(message); }
async function digest(path: string) {
  const hash = new Bun.CryptoHasher("sha256");
  for await (const block of createReadStream(path)) hash.update(block as Buffer);
  return hash.digest("hex");
}
async function canonical(path: string) {
  const absolute = resolve(path);
  need(await realpath(absolute) === absolute, "Symlink/noncanonical input rejected");
  return absolute;
}
function cleanEnvironment() {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined && !key.startsWith("PYTHON") && !["VIRTUAL_ENV", "CONDA_PREFIX"].includes(key)) env[key] = value;
  }
  // A missing unsupported MPS kernel must reject, rather than silently run CPU.
  env.PYTORCH_ENABLE_MPS_FALLBACK = "0";
  return env;
}
async function captured(argv: string[], env = cleanEnvironment()) {
  const began = performance.now();
  const child = Bun.spawn(argv, { cwd: ROOT, env, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exit] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return { argv, stdout, stderr, exit_code: exit, pid: child.pid, wall_seconds: (performance.now() - began) / 1000 };
}
async function dependencies(binary: string) {
  const actual = await captured(["/usr/bin/otool", "-L", binary]);
  need(actual.exit_code === 0, "otool binary inspection failed");
  const texts = [actual.stdout]; const libraries: Record<string, { sha256: string; bytes: number }> = {};
  for (const name of ["libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib", "libomp.dylib"]) {
    const library = await canonical(join(TORCH, "lib", name));
    const inspected = await captured(["/usr/bin/otool", "-L", library]);
    need(inspected.exit_code === 0, "LibTorch dependency inspection failed"); texts.push(inspected.stdout);
    libraries[library] = { sha256: await digest(library), bytes: (await stat(library)).size };
  }
  need(!texts.some(text => /libtorch_python|libpython|Python\.framework/i.test(text)), "Python interpreter dependency forbidden");
  need(actual.stdout.includes("libtorch.dylib") && actual.stdout.includes("libtorch_cpu.dylib") && actual.stdout.includes("libc10.dylib"), "Expected standalone LibTorch linkage missing");
  return { otool: texts, no_python_linkage: true, torch_library_bindings: libraries };
}
function args() {
  const values = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) {
    need(/^--(compile|run|output|binary|binary-sha|manifest-sha|reuse-replay-forward|release-ce-graph)$/.test(process.argv[i]) && process.argv[i + 1] && !process.argv[i + 1].startsWith("--"), "Explicit option/value pairs required");
    need(!values.has(process.argv[i]), "Repeated option"); values.set(process.argv[i], process.argv[i + 1]);
  }
  return values;
}
async function freshParent(path: string) {
  const output = resolve(path); await canonical(dirname(output));
  try { await lstat(output); fail("Fresh output already exists"); } catch (error) { need((error as NodeJS.ErrnoException).code === "ENOENT", "Fresh output required"); }
  // Research artifacts stay outside application/source/model roots.
  need(![ROOT, join("/Volumes/移动数据库/lubanllm", "app"), join("/Volumes/移动数据库/lubanllm", "modules"), join("/Volumes/移动数据库/lubanllm", "models")].some(root => output === root || output.startsWith(root + "/")), "Protected output root");
  return output;
}
async function compile(output: string, reuseReplayForward = false, releaseCeGraph = false) {
  const binary = await freshParent(output); const temporary = binary + ".compiling";
  const exclusive = await open(temporary, "wx", 0o700); await exclusive.close();
  const sourceSHA = await digest(SOURCE);
  const variant = releaseCeGraph ? "release-ce-graph" : reuseReplayForward ? "reuse-replay-forward" : "baseline";
  const flags = ["/usr/bin/clang++", "-std=c++17", "-O1", "-g0", "-Wno-deprecated-declarations", ...(reuseReplayForward ? ["-DLUBAN_REUSE_REPLAY_FORWARD=1"] : []), ...(releaseCeGraph ? ["-DLUBAN_RELEASE_CE_GRAPH=1"] : []), "-I" + join(TORCH, "include"), "-I" + join(TORCH, "include/torch/csrc/api/include"), SOURCE,
    "-L" + join(TORCH, "lib"), "-Wl,-rpath," + join(TORCH, "lib"), "-ltorch", "-ltorch_cpu", "-lc10", "-o", temporary];
  const actual = await captured(flags);
  await writeFile(binary + ".compile.json", JSON.stringify({ ...actual, source_sha256: sourceSHA, variant, reuse_replay_forward: reuseReplayForward, release_ce_graph: releaseCeGraph, parallel_compilers: 1, python_invocations: 0 }, null, 2) + "\n", { flag: "wx" });
  need(actual.exit_code === 0, "Native clang compile failed; original log retained");
  need(await digest(SOURCE) === sourceSHA, "Source changed during build");
  const linkage = await dependencies(temporary); await rename(temporary, binary);
  const report = { binary, binary_sha256: await digest(binary), source_sha256: sourceSHA, variant, reuse_replay_forward: reuseReplayForward, release_ce_graph: releaseCeGraph, model_lock_inheritance: true, engine_name: "Luban 训练引擎", ...linkage, actual_compile: actual, model_started: false };
  await writeFile(binary + ".build.json", JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(report));
}
async function manifest(fixture: string, approved: string) {
  const directory = await canonical(fixture); const path = await canonical(join(directory, "manifest.tsv"));
  need(SHA.test(approved) && await digest(path) === approved, "Exact fixture manifest SHA required");
  const lines = (await readFile(path, "utf8")).trimEnd().split("\n"); need(lines.shift() === ABI, "Wrong fixture ABI");
  const files: Record<string, { sha256: string; bytes: number }> = {}; const names = new Set<string>();
  for (const line of lines) {
    const fields = line.split("\t"); need(["config", "tensor", "row", "step"].includes(fields[0]), "Unsupported manifest record");
    if (fields[0] !== "tensor") continue;
    need(fields.length === 7 && /^[a-z][a-z0-9_]*\.raw$/.test(fields[4]) && SHA.test(fields[6]) && /^(0|[1-9][0-9]*)$/.test(fields[5]), "Malformed raw fixture reference");
    need(!names.has(fields[1]) && !files[fields[4]], "Repeated tensor/file"); names.add(fields[1]);
    const file = await canonical(join(directory, fields[4])); const bytes = Number(fields[5]);
    need(Number.isSafeInteger(bytes) && bytes > 0 && (await stat(file)).size === bytes && await digest(file) === fields[6], "Actual fixture SHA/bytes differ");
    files[fields[4]] = { sha256: fields[6], bytes };
  }
  return { directory, path, manifest_sha256: approved, tensors: files };
}
async function run(input: string, output: string, binaryInput: string, binarySHA: string, manifestSHA: string) {
  const began = performance.now(); const fixture = await manifest(input, manifestSHA);
  const binary = await canonical(binaryInput); need(SHA.test(binarySHA) && await digest(binary) === binarySHA, "Exact native binary SHA required");
  const build = JSON.parse(await readFile(binary + ".build.json", "utf8"));
  need(build.binary_sha256 === binarySHA && build.source_sha256 === await digest(SOURCE), "Binary build/source receipt mismatch");
  need(typeof build.reuse_replay_forward === "boolean" && typeof build.release_ce_graph === "boolean" && (!build.release_ce_graph || build.reuse_replay_forward) && build.variant === (build.release_ce_graph ? "release-ce-graph" : build.reuse_replay_forward ? "reuse-replay-forward" : "baseline") && Array.isArray(build.actual_compile?.argv)
    && build.actual_compile.argv.includes("-DLUBAN_REUSE_REPLAY_FORWARD=1") === build.reuse_replay_forward && build.actual_compile.argv.includes("-DLUBAN_RELEASE_CE_GRAPH=1") === build.release_ce_graph, "Binary build variant/compile flags mismatch");
  const linkage = await dependencies(binary);
  need(JSON.stringify(linkage.torch_library_bindings) === JSON.stringify(build.torch_library_bindings), "Actual LibTorch library bytes changed since binary build");
  const target = await freshParent(output);
  need(!target.startsWith(fixture.directory + "/") && !fixture.directory.startsWith(target + "/"), "Independent fixture/output required");
  const marker = join(dirname(target), basename(target) + ".native-started.json");
  const fd = await open(marker, "wx", 0o600);
  const binding = { fixture, binary, binary_sha256: binarySHA, worker_source_sha256: await digest(SOURCE), driver_source_sha256: await digest(import.meta.path), variant: build.variant, reuse_replay_forward: build.reuse_replay_forward, release_ce_graph: build.release_ce_graph, ...linkage };
  await fd.writeFile(JSON.stringify(binding, null, 2) + "\n"); await fd.sync(); await fd.close();
  const actual = await captured([binary, "--input", fixture.directory, "--output", target, "--manifest-sha", manifestSHA]);
  await writeFile(marker + ".stdout.log", actual.stdout, { flag: "wx" });
  await writeFile(marker + ".stderr.log", actual.stderr, { flag: "wx" });
  let preserved = true;
  try { await manifest(input, manifestSHA); preserved = await digest(binary) === binarySHA && await digest(SOURCE) === binding.worker_source_sha256 && await digest(import.meta.path) === binding.driver_source_sha256;
    for (const [path, expected] of Object.entries(linkage.torch_library_bindings)) preserved = preserved && await digest(path) === expected.sha256; } catch { preserved = false; }
  const report = { ...binding, actual_execution: actual, fixture_binary_sources_preserved: preserved, whole_driver_seconds: (performance.now() - began) / 1000,
    removed_interpreter_environment: ["PYTHON*", "VIRTUAL_ENV", "CONDA_PREFIX"], child_python_invocations: 0, publication: false,
    scope: "Native cached-tail audit/training only; successful execution is not equivalence, mastery or global speed evidence" };
  await writeFile(marker + ".result.json", JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(report)); need(actual.exit_code === 0 && preserved, "Native run rejected/preservation failed; no retry");
}
export { manifest as validateFixtureManifest, cleanEnvironment };
if (import.meta.main) {
const options = args();
if (!options.size) console.log(JSON.stringify({ stage: "SOURCE_ONLY", abi: ABI, worker: SOURCE, linked_libraries: ["libtorch", "libtorch_cpu", "libc10"], no_python_training: true, model_started: false }));
else if (options.has("--compile")) {
  need([...options.keys()].every(key => ["--compile", "--reuse-replay-forward", "--release-ce-graph"].includes(key)) && (!options.has("--reuse-replay-forward") || options.get("--reuse-replay-forward") === "1") && (!options.has("--release-ce-graph") || options.get("--release-ce-graph") === "1" && options.has("--reuse-replay-forward")), "Compile requires fixed variant flags; CE release requires replay reuse");
  await compile(options.get("--compile")!, options.has("--reuse-replay-forward"), options.has("--release-ce-graph"));
} else {
  need(options.size === 5 && ["--run", "--output", "--binary", "--binary-sha", "--manifest-sha"].every(key => options.has(key)), "Run requires fixed fixture/output/binary/exactSHAs");
  await run(options.get("--run")!, options.get("--output")!, options.get("--binary")!, options.get("--binary-sha")!, options.get("--manifest-sha")!);
}
}
