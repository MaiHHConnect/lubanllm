import { mkdir, copyFile, readFile, writeFile, access } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve, join } from "node:path";

// Research/runtime worker only: application packaging consumes the explicit receipt assets.
const root = resolve(import.meta.dir, "..");
const installedSwift = join(homedir(), "Library/Developer/Toolchains/swift-6.3.3-RELEASE.xctoolchain/usr/bin/swift");
const swift = process.env.LUBAN_SWIFT ?? (await Bun.file(installedSwift).exists() ? installedSwift : "swift");
const dependencyRoot = join(root, "build/native-mlx-deps");
const dependencies = [
  { repo: "ml-explore/mlx-swift-lm", ref: "3.32.3", directory: "mlx-swift-lm-3.32.3", marker: "Libraries/MLXLLM/Models/Qwen35.swift" },
  { repo: "ml-explore/mlx-swift", ref: "0.32.3", directory: "mlx-swift-0.32.3", marker: "Source/MLX/MLXArray.swift" },
  { repo: "ml-explore/mlx", ref: "1f8e74e3f12f31365464a6867c6579f0e9b29d85", directory: "mlx-swift-0.32.3/Source/Cmlx/mlx", marker: "mlx/version.h" },
  { repo: "ml-explore/mlx-c", ref: "ebc88f10caa1b625e6b581437a8dea6df8a70085", directory: "mlx-swift-0.32.3/Source/Cmlx/mlx-c", marker: "mlx/c/array.cpp" },
  { repo: "apple/swift-numerics", ref: "1.1.1", directory: "swift-numerics-1.1.1", marker: "Package.swift" },
];
for (const dep of dependencies) {
  const directory = join(dependencyRoot, dep.directory);
  let exists = true;
  try { await access(join(directory, dep.marker)); } catch { exists = false; }
  if (exists) continue;
  await mkdir(directory, { recursive: true });
  const archive = join(dependencyRoot, dep.repo.split("/")[1] + "-" + dep.ref + ".tar.gz");
  const download = Bun.spawn(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20",
    "https://codeload.github.com/" + dep.repo + "/tar.gz/" + dep.ref, "--output", archive], { stdout: "inherit", stderr: "inherit" });
  if (await download.exited !== 0) throw new Error("Official source download failed: " + dep.repo);
  const extract = Bun.spawn(["tar", "-xzf", archive, "-C", directory, "--strip-components=1"], { stdout: "inherit", stderr: "inherit" });
  if (await extract.exited !== 0) throw new Error("Source extraction failed: " + dep.repo);
}
const routingSource = join(dependencyRoot, "mlx-swift-lm-3.32.3/Libraries/MLXLLM/Models/Qwen35.swift");
const routingPatch = Bun.spawn(["bun", join(root, "scripts/apply-mlx-routing-patch.ts"), routingSource],
  { cwd: root, stdout: "inherit", stderr: "inherit" });
if (await routingPatch.exited !== 0) throw new Error("MLX routing API patch failed");
// Build-only manifests select existing official targets and local pinned sources, without editing inference code.
await copyFile(join(root, "native/mlx/LanguagePackage.swift.txt"), join(dependencyRoot, "mlx-swift-lm-3.32.3/Package.swift"));
const mlxManifestPath = join(dependencyRoot, "mlx-swift-0.32.3/Package.swift");
const mlxManifest = await readFile(mlxManifestPath, "utf8");
await writeFile(mlxManifestPath, mlxManifest.replace('.package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")', '.package(name: "swift-numerics", path: "../swift-numerics-1.1.1")'));
const packagePath = join(root, "native/mlx");
const scratch = process.env.LUBAN_MLX_SCRATCH ?? join(root, "build/native-mlx");
const output = join(root, "build/native");
await mkdir(output, { recursive: true });
const build = Bun.spawn([swift, "build", "--package-path", packagePath, "--scratch-path", scratch,
  "--build-system", "native", "--configuration", "release", "--product", "luban-mlx-worker", "--jobs", "4"],
  { cwd: root, stdout: "inherit", stderr: "inherit", env: process.env });
if (await build.exited !== 0) throw new Error("Native Swift MLX build failed; product was not replaced");
const pathRun = Bun.spawn([swift, "build", "--package-path", packagePath, "--scratch-path", scratch,
  "--build-system", "native", "--configuration", "release", "--show-bin-path"], { cwd: root, stdout: "pipe", stderr: "inherit" });
const release = (await new Response(pathRun.stdout).text()).trim();
if (await pathRun.exited !== 0 || !release.startsWith(scratch)) throw new Error("Unexpected Swift binary output");
const executable = join(output, "luban-mlx-worker");
await copyFile(join(release, "luban-mlx-worker"), executable);
// The matching official mlx-metal 0.32.2 distribution provides this standalone GPU library.
// It is a data resource; neither Python nor the Python extension is linked or packaged.
const metalSource = process.env.LUBAN_MLX_METALLIB ?? join(root, "build/community-mlx-runtime/mlx/lib/mlx.metallib");
await copyFile(metalSource, join(output, "mlx.metallib"));
const assets: string[] = ["luban-mlx-worker", "mlx.metallib"];
const smoke = Bun.spawn([executable, "--help"], { stdout: "pipe", stderr: "inherit" });
const help = await new Response(smoke.stdout).text();
if (await smoke.exited !== 0 || !help.includes("luban-worker-jsonl-v1")) throw new Error("Native worker CPU smoke failed");
const dependencyRun = Bun.spawn(["otool", "-L", executable], { stdout: "pipe", stderr: "inherit" });
const dynamicDependencies = await new Response(dependencyRun.stdout).text();
if (await dependencyRun.exited !== 0) throw new Error("Cannot inspect runtime dependencies");
const metallibData = await readFile(join(output, "mlx.metallib"));
const metallibDigest = new Bun.CryptoHasher("sha256").update(metallibData).digest("hex");
const digest = new Bun.CryptoHasher("sha256").update(await readFile(executable)).digest("hex");
const sourceHasher = new Bun.CryptoHasher("sha256");
const sourceFiles = ["native/mlx/Package.swift", "native/mlx/LanguagePackage.swift.txt", "native/mlx/Sources/LubanMLXWorker/main.swift", "native/mlx/Sources/LubanMLXWorker/PrefixCache.swift", "native/mlx/Sources/LubanMLXWorker/DiskPrefixCache.swift", "native/mlx/Sources/LubanMLXWorker/RoutingFeatures.swift", "native/mlx/Sources/LubanMLXWorker/Weights.swift", "native/mlx/Sources/LubanMLXWorker/ExpertCheck.swift", "native/mlx/Sources/LubanMLXWorker/Training.swift", "native/mlx/Sources/LubanMLXWorker/TrainingCheck.swift", "native/mlx/Sources/LubanMLXWorker/VisionSupport.swift", "native/Tokenizer.swift", "scripts/build-mlx-worker.ts", "scripts/apply-mlx-routing-patch.ts"];
for (const file of sourceFiles) {
  sourceHasher.update(file + "\n"); sourceHasher.update(await readFile(join(root, file)));
}
const receipt = { schema: "luban-mlx-worker-build-v1", timestamp: new Date().toISOString(),
  protocol: "luban-worker-jsonl-v1", language: "Swift", mlx_swift_lm: "3.32.3", mlx_swift: "0.32.3",
  binary: executable, binary_sha256: digest, source_sha256: sourceHasher.digest("hex"), source_files: sourceFiles, assets, dynamic_dependencies: dynamicDependencies,
  build: "swift build --build-system native -c release --product luban-mlx-worker --jobs 4", cpu_smoke: true,
  gpu_model_test: false, quality_qualification: false,
  pinned_dependencies: dependencies, metallib_source: metalSource, metallib_version: "0.32.2", metallib_sha256: metallibDigest, metallib_bytes: metallibData.length,
  manifest_changes: ["MLXLMCommon/MLXLLM/MLXVLM target subset; unrelated HF macro targets omitted", "Pinned local dependencies; upstream Swift 6.3 tools requirement preserved"],
};
await writeFile(join(output, "luban-mlx-worker.json"), JSON.stringify(receipt, null, 2) + "\n");
console.log(JSON.stringify({ binary: executable, assets, receipt: join(output, "luban-mlx-worker.json") }));
