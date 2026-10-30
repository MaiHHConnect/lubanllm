import factoryExpert from "../src/luban/factory-expert.json";
import {createHash} from "node:crypto";
import {createReadStream,existsSync} from "node:fs";
import {lstat,mkdtemp,readFile,readdir,realpath,rm,writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {dirname,extname,join,resolve} from "node:path";
import {assertNoInternalFiles} from "./internal-only";
import {checkReleaseMachO} from "./release-macho";
import {assertPythonRuntime,pythonRuntimePlan,pythonRuntimePrefix} from "./package-python-runtime";
// Only files required by the current product. New modules need an explicit review.
export const runtimeFiles = [
  "__init__.py", "agreement.py", "automatic.py", "backbone.py", "backups.py", "benchmark.py",
  "benchmark_service.py", "bun_bridge.py", "calibration.py", "coldpaging.py", "context.py",
  "core_publication.py", "decision.py", "dynamic.py", "evolution.py", "evolution_progress.py",
  "documents.py", "learning_history.py", "external_brain.py", "feeding.py", "foundation.py", "foundation_install.py", "factory_expert.py", "factory-expert.json", "factory_engine.py", "factory-engine-qualifications.json", "generation_checks.py", "goal_topics.py", "growth.py",
  "growth_service.py", "model_quiz.py", "teacher_practice.py", "leaderboard.py", "local_review.py", "luban_training.py", "memory.py",
  "model_packages.py", "mlx_model_packages.py", "name_filter.py", "name_words.LICENSE", "name_words.json", "native_core_training.py",
  "native_engine.py", "native_expert.py", "native_mlx.py", "native_mlx_routing.py", "native_mlx_vision.py", "mlx_learning.py", "native_q4.py", "native_evolution_lease.py", "original_protection.py", "package_api.py", "process_lock.py", "product_inference.py", "retention.py",
  "review.py", "runtime.py", "server.py", "settings.py", "speech.py", "teacher.py", "thinking.py", "tool_protocol.py", "native_q4_clip_profile.txt",
  "media_learning.py", "image_learning.py", "native_image_training.py", "operation_learning.py", "native_operation_training.py", "native_video_training.py", "speech_learning.py", "video_learning.py",
  "chat_context.py", "training_checkpoint.py", "upgrades.py", "vision.py", "web/app.js", "web/index.html", "web/style.css", "web/component-versions.json",
  "web/assets/templates/source.txt", "web/assets/templates/qa.csv", "web/assets/templates/qa.jsonl",
  "web/assets/brand/luban-zero-logo.png", "web/assets/brand/favicon.svg", "web/assets/brand/luban-intro-v1.png",
  "web/assets/brand/home-scene.webp", "web/assets/brand/home-play.svg", "web/assets/brand/home-pencil.png",
  "web/assets/brand/home-study.webp", "web/assets/brand/home-logo.png",
  "web/assets/brand/home-relax.webp", "web/assets/brand/home-inspiration.webp", "web/assets/brand/home-sunset.webp",
  "web/assets/brand/home-music.webp",
  "web/assets/brand/home-game.webp",
  "web/assets/brand/home-night.webp",
  "web/assets/brand/home-ideas.webp",
  "web/assets/brand/home-castle.webp",
  "web/assets/brand/home-painting.webp",
  "web/assets/brand/home-camp.webp",
  "web/assets/brand/home-team.webp",
  "web/assets/brand/home-journey.webp",
  "web/assets/brand/home-overlook.webp",
  "web/assets/brand/home-valley.webp",
  "web/assets/brand/home-hiking.webp",
  "web/assets/brand/home-pixel-sunset.webp",
  "web/assets/brand/home-pixel-music.webp",
  "web/assets/brand/home-pixel-night.webp",
  "web/assets/brand/home-fishing.webp",
  "web/assets/brand/home-seaside.webp",
  "web/assets/brand/home-map.webp",
  "web/assets/brand/home-summit.webp",
  "web/assets/brand/home-departure.webp",
  "web/assets/brand/home-victory.webp",
  "web/assets/brand/home-flight.webp",
  "web/assets/brand/home-coffee.webp",
  "web/assets/brand/home-balcony.webp", "web/assets/brand/home-workshop.webp",
  "web/assets/brand/home-stroll.webp", "web/assets/brand/home-comfort.webp",
  "web/assets/brand/home-milestone-ten.webp", "web/assets/brand/home-milestone-hundred.webp",
  "web/assets/brand/home-milestone-two-hundred.webp",
  "web/assets/brand/home-free-summit.webp",
  "web/assets/brand/home-free-nap.webp",
  "web/assets/brand/home-free-fishing.webp",
  "web/assets/brand/home-free-marbles.webp",
  "web/assets/brand/home-free-cards.webp",
  "web/assets/brand/home-free-warehouse.webp",
  "web/assets/brand/home-free-faces.webp",
  "web/assets/brand/home-free-memories.webp",
  "web/assets/brand/home-free-stickers.webp",
  "web/assets/share/evolution-success.png",
  "web/assets/integrations/chatbox.png", "web/assets/integrations/cherry-studio.png",
  "web/assets/integrations/codebuddy.svg", "web/assets/integrations/openclaw.svg",
  "web/assets/integrations/codex.png", "web/assets/integrations/workbuddy.svg",
  "web/assets/integrations/qwen-office.png", "web/assets/integrations/wukong.png",
  "web/assets/integrations/hermes.png", "web/assets/integrations/qoder.svg", "web/assets/integrations/zcode.png",
  "web/assets/integrations/pi.svg", "web/assets/integrations/trae.png",
] as const;
export const licenseFiles: Record<string, string> = {
  "LICENSE": "Contents/Resources/licenses/LICENSE",
  "THIRD_PARTY_NOTICES.md": "Contents/Resources/licenses/THIRD_PARTY_NOTICES.md",
  "native/luban/licenses/LICENSE": "Contents/Resources/licenses/TensorFold-LICENSE",
  "native/luban/licenses/NOTICE": "Contents/Resources/licenses/TensorFold-NOTICE",
  "native/luban/licenses/THIRD_PARTY_NOTICES.md": "Contents/Resources/licenses/TensorFold-THIRD_PARTY_NOTICES.md",
  "native/luban/licenses/LICENSES/Apache-2.0.txt": "Contents/Resources/licenses/TensorFold-Apache-2.0.txt",
  "native/luban/licenses/LICENSES/MIT.txt": "Contents/Resources/licenses/TensorFold-MIT.txt",
  "native/luban/licenses/LICENSES/MiaAI-Lab-MIT.txt": "Contents/Resources/licenses/TensorFold-MiaAI-Lab-MIT.txt",
  "src/luban/name_words.LICENSE": "Contents/Resources/backend/src/luban/name_words.LICENSE",
  ...Object.fromEntries(["fmt-LICENSE", "metal-cpp-LICENSE", "mlx-ACKNOWLEDGMENTS.md", "mlx-LICENSE",
    "mlx-c-ACKNOWLEDGMENTS.md", "mlx-c-LICENSE", "mlx-swift-LICENSE", "mlx-swift-lm-LICENSE",
    "nlohmann-json-LICENSE", "swift-numerics-LICENSE"].map(name =>
      ["native/mlx/licenses/" + name, "Contents/Resources/licenses/MLX/" + name])),
};
export const evolutionPrefix = "Contents/Resources/native-evolution/";
/** Explicit model assets; factory expert includes only reviewed runtime provenance. */
export const modelReleaseFiles = {
  expert: factoryExpert.asset.files,
  mlx: ["README.md", "config.json", "model.safetensors", "model.safetensors.index.json",
    "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json", "LICENSE"],
  speech: ["README.md", "config.json", "configuration.json", "model.safetensors",
    "chat_template.jinja", "generation_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "LICENSE-2.0.txt"],
} as const;
export const foundationPrefix = "Contents/Resources/foundation/";
export const foundationArchives = [
  {name:"Luban-MLX-Q4-model.zip",bytes:1728259246,prefix:"Luban-MLX-Q4",expanded:1749093508,files:modelReleaseFiles.mlx},
  {name:"Luban-Speech-model.zip",bytes:1567082641,prefix:"Luban-Speech",expanded:1576391182,files:modelReleaseFiles.speech},
  {name:"Luban-Model-Metadata.zip",bytes:7041228,prefix:"Luban-Model-Metadata",expanded:22928412,
    files:["LICENSE","chat_template.jinja","config.json","merges.txt","preprocessor_config.json","tokenizer.json","tokenizer_config.json","upstream-file-manifest.json","video_preprocessor_config.json","vocab.json"]},
  {name:factoryExpert.asset.name,bytes:factoryExpert.asset.bytes,prefix:factoryExpert.asset.prefix,
    expanded:factoryExpert.asset.expanded_bytes,files:factoryExpert.asset.files},
] as const;
export const evolutionCapsulePrefix = evolutionPrefix + "Resources/";
export const evolutionBinaryNames = [
  "native/bin/asr-decoder", "native/bin/asr-features", "native/bin/asr-tokenizer", "native/bin/cached-core", "native/bin/cached-none",
  "native/bin/checkpoint-reader", "native/bin/checkpoint-writer", "native/bin/fresh-rng", "native/bin/lib-generation-fd.dylib",
  "native/bin/lib-process-fd.dylib", "native/bin/luban-text", "native/bin/luban-worker", "native/bin/persistent-expert",
  "native/bin/prefix-worker", "native/bin/replay-basis", "native/bin/router-io", "native/bin/router-metrics",
  "native/bin/router-temperature", "native/bin/router-trainer", "native/bin/syntax-parser", "native/bin/training-v3",
  "native/lib/libc10.dylib", "native/lib/libomp.dylib", "native/lib/libtorch.dylib", "native/lib/libtorch_cpu.dylib",
] as const;
export const evolutionRoleNames = ["asrDecoder", "asrFeatures", "asrTokenizer", "cachedCore", "cachedNone", "reader", "writer", "rng",
  "generationFD", "processFD", "bridge", "worker", "persistentExpert", "prefix", "replayBasis", "routerIO", "routerMetrics",
  "routerTemperature", "routerTrainer", "syntaxParser", "trainer"] as const;
export const evolutionLicenseNames = ["LUBAN_LICENSE", "PYTORCH_LICENSE", "PYTORCH_NOTICE", "TREE_SITTER_LICENSE",
  "TREE_SITTER_PYTHON_LICENSE", "TREE_SITTER_UNICODE_LICENSE", "UNICODE_DATA_LICENSE", "worker-LICENSE", "worker-NOTICE",
  "worker-THIRD_PARTY_NOTICES.md"].map(name => "native/licenses/" + name);
export const evolutionPayload = [...evolutionBinaryNames, ...evolutionLicenseNames, "native/bin/luban-worker.json", "native/runtime-release.json"].sort();
export const binaryFiles = ["Contents/MacOS/Luban", "Contents/MacOS/luban-service",
  "Contents/MacOS/luban-update-service", "Contents/Resources/backend/native/luban-worker",
  "Contents/Resources/backend/native/luban-mlx-worker",
  "Contents/Resources/backend/native/luban-training",
  "Contents/Resources/backend/native/luban-asr-training", "Contents/Resources/backend/native/luban-asr-features",
  "Contents/Resources/backend/native/luban-asr-tokenizer",
  "Contents/Resources/backend/native/luban-video-frames",
  "Contents/Resources/backend/native/luban-document-text",
  "Contents/Resources/backend/native/libluban_image_training.dylib",
  "Contents/Resources/backend/native/luban-evolution-observe",
  ...evolutionBinaryNames.map(name => evolutionCapsulePrefix + name)] as const;
export const requiredPayload = [...binaryFiles, "Contents/Info.plist", "Contents/Resources/launcher.json",
  "Contents/Resources/LubanIcon.icns",
  "Contents/Resources/release.json", "Contents/Resources/backend/release.json",
  "Contents/Resources/backend/native/luban-worker.json",
  "Contents/Resources/backend/native/luban-mlx-worker.json", "Contents/Resources/backend/native/mlx.metallib",
  "Contents/Resources/backend/native/luban-training.build.json",
  "Contents/Resources/backend/native/luban-asr-training.build.json",
  "Contents/Resources/backend/native/luban-asr-features.build.json",
  "Contents/Resources/backend/native/luban-asr-tokenizer.build.json",
  "Contents/Resources/backend/native/libluban_image_training.dylib.build.json",
  "Contents/Resources/backend/native/luban-evolution-observe.build.json",
  evolutionPrefix + "worker.js", ...evolutionPayload.map(name => evolutionCapsulePrefix + name),
  ...runtimeFiles.map(file => "Contents/Resources/backend/src/luban/" + file), ...Object.values(licenseFiles)];

const need = (ok:unknown,message:string) => {if(!ok) throw Error("发行载荷门禁失败："+message);};
async function sha(path:string) {const hash=createHash("sha256");for await(const data of createReadStream(path))hash.update(data);return hash.digest("hex");}
export const mlxSourceFiles = ["native/mlx/Package.swift", "native/mlx/LanguagePackage.swift.txt",
  "native/mlx/Sources/LubanMLXWorker/main.swift", "native/mlx/Sources/LubanMLXWorker/PrefixCache.swift", "native/mlx/Sources/LubanMLXWorker/DiskPrefixCache.swift", "native/mlx/Sources/LubanMLXWorker/RoutingFeatures.swift", "native/mlx/Sources/LubanMLXWorker/Weights.swift",
  "native/mlx/Sources/LubanMLXWorker/ExpertCheck.swift", "native/mlx/Sources/LubanMLXWorker/Training.swift",
  "native/mlx/Sources/LubanMLXWorker/TrainingCheck.swift", "native/mlx/Sources/LubanMLXWorker/VisionSupport.swift",
  "native/Tokenizer.swift", "scripts/build-mlx-worker.ts", "scripts/apply-mlx-routing-patch.ts"] as const;
export async function readMLXWorkerBuild(binary: string, projectRoot: string) {
  const receipt = JSON.parse(await readFile(binary + ".json", "utf8"));
  const source = createHash("sha256");
  for (const file of mlxSourceFiles) { source.update(file + "\n"); source.update(await readFile(join(projectRoot, file))); }
  const metal = join(dirname(binary), "mlx.metallib");
  need(receipt.schema === "luban-mlx-worker-build-v1" && receipt.language === "Swift"
    && receipt.protocol === "luban-worker-jsonl-v1" && receipt.cpu_smoke === true
    && receipt.mlx_swift_lm === "3.32.3" && receipt.mlx_swift === "0.32.3"
    && JSON.stringify(receipt.assets) === JSON.stringify(["luban-mlx-worker", "mlx.metallib"])
    && receipt.source_sha256 === source.digest("hex") && receipt.binary_sha256 === await sha(binary)
    && receipt.metallib_bytes === (await lstat(metal)).size && receipt.metallib_sha256 === await sha(metal),
    "MLX 原生程序、源码或Metal资源构建绑定不匹配");
  return receipt;
}
export async function readEvolutionObserverBuild(binary: string, receiptPath: string, projectRoot: string) {
  const receipt = JSON.parse(await readFile(receiptPath, "utf8"));
  const sources = ["native/EvolutionObserve.swift", "native/Worker.swift", "native/Tokenizer.swift"];
  const bytes: Buffer[] = [];
  for (const source of sources) bytes.push(Buffer.from(source + "\0"), await readFile(join(projectRoot, source)), Buffer.from("\0"));
  need(receipt.schema === "luban-native-evolution-observer-build-v1" && receipt.compiled === true && receipt.signed === true
    && receipt.precision === "mixed-q4-g16" && receipt.layers === 24 && receipt.seed === 42 && receipt.temperature === 0
    && receipt.python_invocations === 0 && JSON.stringify(receipt.sources) === JSON.stringify(sources)
    && receipt.source_bundle_sha256 === createHash("sha256").update(Buffer.concat(bytes)).digest("hex")
    && receipt.binary_sha256 === await sha(binary), "原生Q4观察程序源码/二进制构建回执不匹配");
  return receipt;
}
// Runtime capsule members are fixed here, never accepted from an untrusted manifest's path list.
export async function checkEvolutionCapsule(root: string, projectRoot: string) {
  const manifest = JSON.parse(await readFile(join(root, "native/runtime-release.json"), "utf8"));
  function metadata(value: unknown) {
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      need(!["stdout", "stderr", "edits", "source_text", "patch", "diff"].includes(key), "原生进化运行收据包含研究日志或源码变换");
      metadata(child);
    }
  }
  metadata(manifest);
  const equalNames = (value: unknown, expected: readonly string[]) => value && typeof value === "object"
    && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...expected].sort());
  need(manifest.schema === "luban-native-unified-service-role-capsule-v6" && equalNames(manifest.payload, evolutionBinaryNames)
    && equalNames(manifest.libraries, evolutionBinaryNames.filter(name => name.startsWith("native/lib/")))
    && equalNames(manifest.licenses, evolutionLicenseNames) && equalNames(manifest.assets, ["native/bin/luban-worker.json"]), "原生进化运行载荷清单无效");
  for (const name of [...evolutionBinaryNames, ...evolutionLicenseNames, "native/bin/luban-worker.json"]) {
    const binding = manifest.payload[name]?.final ?? manifest.licenses[name] ?? manifest.assets[name];
    const file = join(root, name), stat = await lstat(file);
    need(binding && /^[a-f0-9]{64}$/.test(binding.sha256) && Number.isSafeInteger(binding.bytes) && binding.bytes > 0
      && stat.isFile() && !stat.isSymbolicLink() && stat.size === binding.bytes && await sha(file) === binding.sha256,
      "原生进化最终文件绑定不匹配：" + name);
    if (evolutionLicenseNames.includes(name)) need(await sha(join(projectRoot, "build/native-evolution/Resources", name)) === binding.sha256,
      "原生进化许可证内容与原件不一致：" + name);
    if (name.startsWith("native/lib/")) need(manifest.libraries[name]?.sha256 === binding.sha256 && manifest.libraries[name]?.bytes === binding.bytes,
      "原生进化动态库绑定不匹配：" + name);
  }
  const binaries = new Set<string>(evolutionBinaryNames.filter(name => name.startsWith("native/bin/")));
  need(equalNames(manifest.programs, evolutionRoleNames), "原生进化角色缺失或未知");
  for (const [index, name] of evolutionRoleNames.entries()) {
    const program = manifest.programs[name];
    need(program.path === evolutionBinaryNames[index] && binaries.delete(program.path) && program.proof
      && /^[a-f0-9]{64}$/.test(program.proof.source_build_sha256), "原生进化角色来源绑定无效");
  }
  need(binaries.size === 0, "原生进化角色重复或缺失");
  return manifest;
}
// Shared build/package boundary for the image library; never used during learning.
export async function readImageTrainingBuild(binary:string, receiptPath:string, projectRoot:string) {
  const [file, receiptFile] = await Promise.all([lstat(binary), lstat(receiptPath)]);
  need(file.isFile() && !file.isSymbolicLink() && receiptFile.isFile() && !receiptFile.isSymbolicLink(), "图片训练库与回执必须是普通文件");
  const receipt = JSON.parse(await readFile(receiptPath, "utf8"));
  const source = Buffer.concat([Buffer.from("native/ImageTraining.cpp\0"), await readFile(join(projectRoot, "native/ImageTraining.cpp")), Buffer.from("\0")]);
  need(receipt.schema === "luban-native-image-training-build-v1" && receipt.production_shape === true &&
    receipt.no_python_linkage === true && receipt.actual_compile?.exit_code === 0 &&
    JSON.stringify(receipt.sources) === JSON.stringify(["native/ImageTraining.cpp"]) &&
    receipt.source_bundle_sha256 === createHash("sha256").update(source).digest("hex") &&
    receipt.binary_sha256 === await sha(binary), "原生图片训练库源码/二进制构建回执不匹配");
  return receipt;
}
// The archive adapter compiles only private copies of the existing native validator.
// No Python fallback; an unavailable platform compiler fails the release check.
export async function assertReleasePayload(path:string, projectRoot=resolve(import.meta.dir,"..")) {
  path = resolve(path); const entry = await lstat(path);
  need(!entry.isSymbolicLink(), "发行载荷禁止符号链接");
  const licenses = Object.fromEntries(await Promise.all(Object.entries(licenseFiles).map(async ([source, target]) => [target, await sha(join(projectRoot, source))])));
  async function walk(app: string) {
    const allowed = new Set<string>([...requiredPayload, "Contents/_CodeSignature/CodeResources"]), files = new Set<string>(), directories = new Set([""]);
    const approvedArchives = new Set<string>();
    if(existsSync(join(app,foundationPrefix))) {
      const code = `import json,pathlib,stat,sys,zipfile
root=pathlib.Path(sys.argv[1])
assets=json.load(sys.stdin)
assert root.is_dir() and not root.is_symlink() and {p.name for p in root.iterdir()}=={a['name'] for a in assets}
for a in assets:
 p=root/a['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==a['bytes']
 with zipfile.ZipFile(p) as z:
  rows=z.infolist();assert len(rows)==len(a['files']) and {r.filename for r in rows}=={a['prefix']+'/'+n for n in a['files']}
  assert sum(r.file_size for r in rows)==a['expanded']
  assert all(not r.is_dir() and not r.flag_bits&1 and stat.S_IFMT(r.external_attr>>16) in (0,stat.S_IFREG) and r.compress_type in (0,8) and (not r.filename.endswith('.safetensors') or r.compress_type==0) for r in rows)
print('PINNED FOUNDATION ARCHIVES PASSED')`;
      const child=Bun.spawn([join(projectRoot,".venv/bin/python"),"-c",code,join(app,foundationPrefix)],{stdin:"pipe",stdout:"pipe",stderr:"pipe"});
      child.stdin.write(JSON.stringify(foundationArchives));child.stdin.end();
      await Promise.all([new Response(child.stdout).text(),new Response(child.stderr).text()]);
      need(await child.exited===0,"内置模型归档不匹配已固定的公开发行载荷");
      for(const asset of foundationArchives){allowed.add(foundationPrefix+asset.name);approvedArchives.add(join(app,foundationPrefix,asset.name));}
    }
    const launcher = JSON.parse(await readFile(join(app, "Contents/Resources/launcher.json"), "utf8"));
    const bundledPython = existsSync(join(app, pythonRuntimePrefix));
    need(launcher.mode === "bundled" || typeof launcher.project === "string", "应用启动配置无效");
    need(launcher.mode !== "bundled" || bundledPython, "独立应用缺少便携Python运行环境");
    const python = bundledPython ? await assertPythonRuntime(join(app, pythonRuntimePrefix), await pythonRuntimePlan()) : null;
    if (python) {
      const manifest = JSON.parse(await readFile(join(app, pythonRuntimePrefix, "runtime-manifest.json"), "utf8"));
      for (const name of [...manifest.files, "runtime-manifest.json"]) allowed.add(pythonRuntimePrefix + "/" + name);
    }
    for (const name of allowed) for (let parent = dirname(name); parent !== "."; parent = dirname(parent)) directories.add(parent);
    async function visit(relative: string) {
      const file = join(app, relative), info = await lstat(file);
      need(!info.isSymbolicLink() && (info.isDirectory() || info.isFile()), "发行载荷包含链接或特殊文件：" + relative);
      need(relative.normalize("NFKC") === relative && !relative.includes("\\") && !(relative.split("/").some(part => part === "." || part === "..")), "不安全的发行路径");
      need((info.isDirectory() ? directories : allowed).has(relative), "发行载荷不在白名单：" + relative);
      if (info.isDirectory()) { for (const name of await readdir(file)) await visit(relative ? relative + "/" + name : name); return; }
      files.add(relative);
      if ((binaryFiles as readonly string[]).includes(relative)) checkReleaseMachO(await readFile(file), relative);
      if (licenses[relative]) need(info.size > 0 && await sha(file) === licenses[relative], "许可证内容与原件不一致：" + relative);
    }
    await visit("");
    need(requiredPayload.every(name => files.has(name)), "缺少必要运行文件/许可证：" + requiredPayload.filter(name => !files.has(name)).join(", "));
    await readMLXWorkerBuild(join(app, "Contents/Resources/backend/native/luban-mlx-worker"), projectRoot);
    await readImageTrainingBuild(join(app, "Contents/Resources/backend/native/libluban_image_training.dylib"),
      join(app, "Contents/Resources/backend/native/libluban_image_training.dylib.build.json"), projectRoot);
    await readEvolutionObserverBuild(join(app, "Contents/Resources/backend/native/luban-evolution-observe"),
      join(app, "Contents/Resources/backend/native/luban-evolution-observe.build.json"), projectRoot);
    await checkEvolutionCapsule(join(app, evolutionCapsulePrefix), projectRoot);
    need(await sha(join(app, evolutionPrefix, "worker.js")) === await sha(join(projectRoot, "build/native-evolution/worker.js")),
      "原生进化worker与实际构建不一致");
    await assertNoInternalFiles(app, approvedArchives);
    const { assertComponentVersions } = await import("./component-versions");
    await assertComponentVersions(projectRoot, join(app, "Contents/Resources/backend/src/luban/web"));
    return { files: files.size, binaries: binaryFiles.length + (python?.binaries.length ?? 0),
      licenses: Object.keys(licenses).length + (python?.licenses.length ?? 0), standalone: launcher.mode === "bundled" && !!python, completeModels: approvedArchives.size === foundationArchives.length };
  }
  if (extname(path).toLowerCase() !== ".zip") {
    need(entry.isDirectory() && extname(path) === ".app", "请提供实际.app目录或发行ZIP"); return walk(path);
  }
  const complete = path.endsWith("-macos-arm64-full.zip");
  need(entry.isFile() && entry.size > 0 && entry.size <= (complete ? 4 * 1024 ** 3 - 1 : 512 << 20), "ZIP类型/大小无效");

  const temporary=await realpath(await mkdtemp(join(tmpdir(),"luban-native-release-")));
  async function command(argv:string[]) {
    const child=Bun.spawn(argv,{env:{PATH:"/usr/bin:/bin",LANG:"en_US.UTF-8"},stdin:"ignore",stdout:"pipe",stderr:"pipe"});
    const [stdout,stderr,code]=await Promise.all([new Response(child.stdout).text(),new Response(child.stderr).text(),child.exited]);
    need(code===0,stderr.trim()||"原生发行检查程序失败"); return stdout;
  }
  try {
    const sourceRoot=resolve(import.meta.dir,"../native");
    for(const name of ["UpdateArchive.swift","ReleaseInspection.swift"])
      await writeFile(join(temporary,name),await readFile(join(sourceRoot,name)),{flag:"wx",mode:0o600});
    const helper=join(temporary,"inspector"), sdk=(await command(["/usr/bin/xcrun","--sdk","macosx","--show-sdk-path"])).trim();
    need(sdk.startsWith("/"),"macOS SDK不可用");
    await command(["/usr/bin/xcrun","--sdk","macosx","swiftc",join(temporary,"UpdateArchive.swift"),join(temporary,"ReleaseInspection.swift"),...(complete?["-D","COMPLETE_RELEASE_INSPECTION"]:[]),"-sdk",sdk,"-O","-o",helper]);
    await command(["/usr/bin/codesign","--force","--sign","-",helper]);
    await command(["/usr/bin/codesign","--verify","--strict",helper]);
    const digest=await sha(path), stage=join(temporary,"payload");
    const stdout=await command([helper,path,stage,digest]);
    const receipt = JSON.parse(stdout); need(receipt.schema === "luban-native-release-inspection-v1" && receipt.sha256 === digest, "原生ZIP检查绑定无效");
    need((await readdir(stage)).join() === "Luban.app", "发行ZIP包含应用之外的成员");
    const result=await walk(join(stage, "Luban.app"));
    need(!complete || result.completeModels,"完整安装包必须包含全部模型");
    return { ...result, archiveSha256: digest };

  } catch(error) {
    let message=String(error);
    if(/links/.test(message)) message="链接："+message;
    if(/Internal-only|internal archive/.test(message)) message="内部资料："+message;
    throw Error("发行载荷门禁失败："+message);
  } finally {await rm(temporary,{recursive:true,force:true});}
}
if (import.meta.main) {
  try {
    const paths = Bun.argv.slice(2);
    if (!paths.length) throw new Error("请指定实际.app目录或发行ZIP");
    for (const path of paths) console.log(`发行载荷门禁通过：${path} ${JSON.stringify(await assertReleasePayload(path))}`);
  } catch (error) { console.error((error as Error).message); process.exitCode = 1; }
}
