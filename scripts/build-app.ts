import { cp, mkdir, mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { readRelease } from "./release";
import { assertReleasePayload, readImageTrainingBuild, readEvolutionObserverBuild, readMLXWorkerBuild, binaryFiles, licenseFiles, runtimeFiles,
  evolutionPayload, evolutionPrefix, evolutionCapsulePrefix } from "./release-payload";
import { checkReleaseMachO } from "./release-macho";
import { assertComponentVersions } from "./component-versions";
import { packagePythonRuntime, pythonRuntimePrefix } from "./package-python-runtime";

const root = resolve(import.meta.dir, "..");
const args = Bun.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== "--output" || !args[1].endsWith(".app"))) throw new Error("用法：bun scripts/build-app.ts [--output /new/path/Luban.app]");
const app = args.length ? resolve(args[1]) : resolve(root, "build/Luban.app");
if (args.length && existsSync(app)) throw new Error("候选输出必须是全新应用目录");
const bundle = resolve(app, "Contents");
const mlxBuild = Bun.spawn([process.execPath, "scripts/build-mlx-worker.ts"], { cwd: root, stdout: "inherit", stderr: "inherit" });
if (await mlxBuild.exited !== 0) throw new Error("原生 MLX 推理程序构建失败");
const evolutionBundle = Bun.spawn([process.execPath, "scripts/build-native-evolution-worker.ts"], { cwd: root, stdout: "inherit", stderr: "inherit" });
if (await evolutionBundle.exited !== 0) throw new Error("原生进化运行程序构建失败");
const frontendBuild = Bun.spawn([process.execPath, "frontend/build.ts"], { cwd: root, stdout: "inherit", stderr: "inherit" });
if (await frontendBuild.exited !== 0) throw new Error("前端与组件版本构建失败");
await assertComponentVersions(root);
const release = await readRelease(root);
const worker = resolve(root, "build/native/luban-worker");
const mlxWorker = resolve(root, "build/native/luban-mlx-worker");
const mlxReceipt = await readMLXWorkerBuild(mlxWorker, root);
const receipt = JSON.parse(await readFile(worker + ".json", "utf8"));
const sha = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
if (sha(await readFile(worker)) !== receipt.binary_sha256) throw new Error("原生推理二进制SHA不匹配，请重新构建");
for (const [path, hash] of Object.entries(receipt.source_sha256)) {
  if (sha(await readFile(resolve(root, "native/luban/overlay", path))) !== hash) throw new Error(`原生源码已变化：${path}，请重新构建`);
}
const trainingWorker = resolve(root, "build/native/luban-training");
const trainingReceipt = JSON.parse(await readFile(trainingWorker + ".build.json", "utf8"));
if (sha(await readFile(trainingWorker)) !== trainingReceipt.binary_sha256 || trainingReceipt.source_sha256 !== sha(await readFile(resolve(root, "native/Training.cpp"))) || !trainingReceipt.release_ce_graph || !trainingReceipt.model_lock_inheritance) {
  throw new Error("Luban训练引擎源码/二进制回执必须匹配经过核验的CE释放版，请重新构建训练引擎");
}
const asrWorkers = await Promise.all(["luban-asr-training", "luban-asr-features", "luban-asr-tokenizer"].map(async name => {
  const source = resolve(root, "build/native", name), original = JSON.parse(await readFile(source + ".build.json", "utf8"));
  if (original.schema !== "luban-native-asr-build-v1" || original.production_shape !== true || !original.no_python_linkage ||
      sha(await readFile(source)) !== original.binary_sha256 || !Array.isArray(original.sources) || !original.sources.length ||
      original.sources.some((path: unknown) => typeof path !== "string" || !path.startsWith("native/") || path.includes(".."))) {
    throw new Error(`原生声音组件构建回执无效：${name}`);
  }
  const bytes: Uint8Array[] = [];
  for (const path of original.sources) bytes.push(Buffer.from(path + "\0"), await readFile(resolve(root, path)), Buffer.from("\0"));
  if (sha(Buffer.concat(bytes)) !== original.source_bundle_sha256) throw new Error(`原生声音组件源码已改变：${name}`);
  return { source, original };
}));
const imageSource = resolve(root, "build/native/libluban_image_training.dylib");
const imageReceipt = await readImageTrainingBuild(imageSource, imageSource + ".build.json", root);
const imageWorker = {source: imageSource, original: imageReceipt};
const evolutionSource = resolve(root, "build/native-evolution"), observer = resolve(root, "build/native/luban-evolution-observe");
await readEvolutionObserverBuild(observer, observer + ".build.json", root);
const authority = JSON.parse(await readFile(resolve(evolutionSource, "runtime-authority.json"), "utf8"));
const capsuleBytes = await readFile(resolve(evolutionSource, "Resources/native/runtime-release.json"));
const capsule = JSON.parse(capsuleBytes.toString("utf8"));
if (JSON.stringify([...authority.payload].sort()) !== JSON.stringify(evolutionPayload)
    || authority.authority.manifest_relative !== "native/runtime-release.json"
    || sha(capsuleBytes) !== authority.authority.manifest_sha256) {
  throw new Error("原生进化胶囊绑定或明确载荷清单不匹配，请重新构建");
}
for (const [name, role] of Object.entries(authority.roles) as [string, any][]) {
  if (capsule.programs[name]?.path !== role.path || capsule.payload[role.path]?.final?.sha256 !== role.sha256
      || sha(Buffer.from(JSON.stringify(capsule.programs[name].proof))) !== authority.authority.sources[name]) {
    throw new Error("原生进化角色来源与最终程序绑定不匹配：" + name);
  }
}
const evolutionBuild = JSON.parse(await readFile(resolve(evolutionSource, "build.json"), "utf8"));
const evolutionWorker = await readFile(resolve(evolutionSource, "worker.js"));
if (evolutionBuild.worker_sha256 !== sha(evolutionWorker)
    || evolutionBuild.runtime_manifest_sha256 !== authority.authority.manifest_sha256) throw new Error("原生进化worker与运行胶囊构建绑定不一致");
await rm(app, { recursive: true, force: true });
await mkdir(`${bundle}/MacOS`, { recursive: true });
await mkdir(`${bundle}/Resources`, { recursive: true });
await packagePythonRuntime(resolve(app, pythonRuntimePrefix));
// Bun 1.3.5 can report success but emit an empty binary on an external HFS+ volume.
// Compile on the macOS system filesystem, validate it, then copy into the candidate.
const serviceTemporary = await mkdtemp("/private/tmp/luban-service-build-");
try {
  const compiled = resolve(serviceTemporary, "luban-service");
  const serviceBuild = Bun.spawn([process.execPath, "build", "backend/main.ts", "--compile", "--minify",
    "--no-compile-autoload-dotenv", "--no-compile-autoload-bunfig", "--outfile", compiled],
    { cwd: root, stdout: "inherit", stderr: "inherit" });
  if (await serviceBuild.exited !== 0) throw new Error("Bun本地服务编译失败");
  const bytes = await readFile(compiled);
  if (bytes.length < 32 || bytes.subarray(0, 4).toString("hex") !== "cffaedfe") throw new Error("Bun编译未生成有效macOS程序");
  await cp(compiled, `${bundle}/MacOS/luban-service`);
} finally { await rm(serviceTemporary, { recursive: true, force: true }); }
const build = Bun.spawn(["/Library/Developer/CommandLineTools/usr/bin/swiftc", "native/Luban.swift", "native/Updater.swift", "native/UpdateArchive.swift", "native/OperationRecorder.swift",
  "-sdk", (await Bun.$`xcrun --show-sdk-path`.text()).trim(),
  "-O", "-framework", "AppKit", "-framework", "WebKit", "-o", `${bundle}/MacOS/Luban`],
  { cwd: root, stdout: "inherit", stderr: "inherit" });
if (await build.exited !== 0) throw new Error("Mac 应用编译失败");
const updateHelper = Bun.spawn(["/Library/Developer/CommandLineTools/usr/bin/swiftc", "native/Updater.swift", "native/UpdateArchive.swift", "native/UpdateService.swift",
  "-sdk", (await Bun.$`xcrun --show-sdk-path`.text()).trim(), "-O", "-o", `${bundle}/MacOS/luban-update-service`],
  { cwd: root, stdout: "inherit", stderr: "inherit" });
if (await updateHelper.exited !== 0) throw new Error("原生更新检查程序编译失败");
await writeFile(`${bundle}/Info.plist`, `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>Luban</string>
<key>CFBundleIdentifier</key><string>local.luban.desktop</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>LSMinimumSystemVersion</key><string>26.0</string>
<key>CFBundleName</key><string>鲁班零号</string>
<key>CFBundleDisplayName</key><string>鲁班零号</string>
<key>CFBundleIconFile</key><string>LubanIcon</string>
<key>CFBundleVersion</key><string>${release.build}</string>
<key>CFBundleShortVersionString</key><string>${release.version}</string>
<key>NSHighResolutionCapable</key><true/>
<key>NSDownloadsFolderUsageDescription</key><string>导入模型包和学习资料。</string>
<key>NSRemovableVolumesUsageDescription</key><string>读取约 2B 本地模型并保存学习材料和神经权重。</string>
<key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>`);
await writeFile(`${bundle}/Resources/launcher.json`, JSON.stringify({ mode: "bundled" }));
await cp(resolve(root, "native/assets/LubanIcon.icns"), `${bundle}/Resources/LubanIcon.icns`);
for (const file of runtimeFiles) {
  const destination = resolve(bundle, "Resources/backend/src/luban", file);
  await mkdir(resolve(destination, ".."), { recursive: true });
  await cp(resolve(root, "src/luban", file), destination, { dereference: false });
}
for (const [source, file] of Object.entries(licenseFiles)) {
  const destination = resolve(app, file);
  await mkdir(resolve(destination, ".."), { recursive: true });
  await cp(resolve(root, source), destination, { dereference: false });
}
await mkdir(`${bundle}/Resources/backend/native`, { recursive: true });
const videoFrames = Bun.spawn(["/Library/Developer/CommandLineTools/usr/bin/swiftc", resolve(root, "native/VideoFrames.swift"),
  "-sdk", (await Bun.$`xcrun --show-sdk-path`.text()).trim(), "-O", "-framework", "AVFoundation",
  "-framework", "ImageIO", "-framework", "UniformTypeIdentifiers", "-o", `${bundle}/Resources/backend/native/luban-video-frames`],
  { stdout: "inherit", stderr: "inherit" });
if (await videoFrames.exited !== 0) throw new Error("视频画面解码程序编译失败");
const documentText = Bun.spawn(["/Library/Developer/CommandLineTools/usr/bin/swiftc", resolve(root, "native/DocumentText.swift"),
  "-sdk", (await Bun.$`xcrun --show-sdk-path`.text()).trim(), "-O", "-framework", "AppKit", "-framework", "PDFKit", "-o", `${bundle}/Resources/backend/native/luban-document-text`],
  { stdout: "inherit", stderr: "inherit" });
if (await documentText.exited !== 0) throw new Error("文档解析程序编译失败");
await cp(worker, `${bundle}/Resources/backend/native/luban-worker`);
await cp(worker + ".json", `${bundle}/Resources/backend/native/luban-worker.json`);
await cp(mlxWorker, `${bundle}/Resources/backend/native/luban-mlx-worker`);
await cp(mlxWorker + ".json", `${bundle}/Resources/backend/native/luban-mlx-worker.json`);
await cp(resolve(root, "build/native/mlx.metallib"), `${bundle}/Resources/backend/native/mlx.metallib`);
await cp(trainingWorker, `${bundle}/Resources/backend/native/luban-training`);
await cp(trainingWorker + ".build.json", `${bundle}/Resources/backend/native/luban-training.build.json`);
for (const {source} of [...asrWorkers, imageWorker]) {
  const destination = resolve(bundle, "Resources/backend/native", source.split("/").at(-1)!);
  await cp(source, destination);
  await cp(source + ".build.json", destination + ".build.json");
}
await mkdir(resolve(app, evolutionPrefix), { recursive: true });
await writeFile(resolve(app, evolutionPrefix, "worker.js"), evolutionWorker);
for (const name of evolutionPayload) {
  const destination = resolve(app, evolutionCapsulePrefix, name);
  await mkdir(resolve(destination, ".."), { recursive: true });
  await cp(resolve(evolutionSource, "Resources", name), destination, { dereference: false });
}
if (sha(await readFile(resolve(app, evolutionCapsulePrefix, "native/runtime-release.json"))) !== authority.authority.manifest_sha256) {
  throw new Error("原生进化胶囊在打包期间发生变化，请重新构建");
}
await cp(observer, `${bundle}/Resources/backend/native/luban-evolution-observe`);
await cp(observer + ".build.json", `${bundle}/Resources/backend/native/luban-evolution-observe.build.json`);
// Strip only package copies; retain original research binaries and build receipts.
for (const file of binaryFiles) {
  const binary = resolve(app, file);
  // These final bytes are already signed and bound by their runtime/observer receipts.
  // Re-signing or stripping here would invalidate the compiled native authority.
  if (file.startsWith(evolutionCapsulePrefix) || file.endsWith("/luban-evolution-observe")) {
    checkReleaseMachO(await readFile(binary), file);
    const verify = Bun.spawn(["/usr/bin/codesign", "--verify", "--strict", binary], { stdout: "inherit", stderr: "inherit" });
    if (await verify.exited !== 0) throw new Error(`原生进化发行文件签名无效：${file}`);
    continue;
  }
  const strip = Bun.spawn(["/usr/bin/strip", "-S", "-x", binary], { stdout: "inherit", stderr: "inherit" });
  if (await strip.exited !== 0) throw new Error(`发行程序调试符号清理失败：${file}`);
  // Signing the main executable seals its enclosing bundle. Sign all nested code first.
  if (file === "Contents/MacOS/Luban") continue;
  const sign = Bun.spawn(["/usr/bin/codesign", "--force", "--sign", "-", binary], { stdout: "inherit", stderr: "inherit" });
  if (await sign.exited !== 0) throw new Error(`发行程序签名失败：${file}`);
}
for (const [source, original, suffix] of [[worker, receipt, ".json"], [mlxWorker, mlxReceipt, ".json"], [trainingWorker, trainingReceipt, ".build.json"],
  ...[...asrWorkers, imageWorker].map(({source, original}) => [source, original, ".build.json"] as const)] as const) {
  const binary = resolve(bundle, "Resources/backend/native", source.split("/").at(-1)!);
  await writeFile(binary + suffix, JSON.stringify({ ...original, binary_sha256: sha(await readFile(binary)),
    packaging: { source_binary_sha256: original.binary_sha256,
      source_receipt_sha256: sha(await readFile(source + suffix)), operations: ["strip -S -x", "codesign --force --sign -"] } }, null, 2) + "\n");
}
await Promise.all([`${bundle}/Resources/release.json`, `${bundle}/Resources/backend/release.json`]
  .map(path => writeFile(path, JSON.stringify(release, null, 2) + "\n")));
const signing = Bun.spawn(["/usr/bin/codesign", "--force", "--sign", "-", app],
  { stdout: "inherit", stderr: "inherit" });
if (await signing.exited !== 0) throw new Error("本机应用签名失败");
await assertReleasePayload(app, root);
await assertComponentVersions(root, `${bundle}/Resources/backend/src/luban/web`);
console.log(`Mac 应用已构建：${app}`);
