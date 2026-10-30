import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { runtimeFiles } from "./release-payload";
import { compareVersions, validateRelease, type Release } from "./release";

export const componentDefinitions = [
  { id: "frontend", name: "界面", patterns: ["frontend/**", "native/assets/**", "native/Luban.swift", "native/OperationRecorder.swift", "native/Updater.swift", "native/Update*.swift", "native/ReleaseInspection.swift", "scripts/build-app.ts", "scripts/build-app-icon.swift", "scripts/component-versions.ts", "scripts/release*.ts", "scripts/internal-only.ts", "package.json"] },
  { id: "service", name: "本地服务", patterns: ["backend/*.ts", ...runtimeFiles.filter(p => p.endsWith(".py")).map(p => "src/luban/" + p), "src/luban/factory-expert.json", "src/luban/factory-engine-qualifications.json", "scripts/package-python-runtime.ts", "scripts/python-runtime-lock.json", "native/python/licenses/*", "package.json", "pyproject.toml"] },
  { id: "generation", name: "生成引擎", patterns: ["native/luban/overlay/**", "native/luban/build.py", "native/mlx/Package.swift", "native/mlx/LanguagePackage.swift.txt", "native/mlx/Sources/**", "scripts/build-mlx-worker.ts", "src/luban/native_mlx.py", "src/luban/native_engine.py", "src/luban/product_inference.py", "src/luban/native_q4_clip_profile.txt", "src/luban/runtime.py", "src/luban/backbone.py", "src/luban/context.py", "src/luban/coldpaging.py", "src/luban/thinking.py", "pyproject.toml"], optional: ["build/native/luban-worker", "build/native/luban-worker.json", "build/native/luban-mlx-worker", "build/native/luban-mlx-worker.json"] },
  { id: "operators", name: "算子", patterns: ["native/*.cpp", "native/asr/**", "native/luban/overlay/zig/kernels/**", "native/luban/overlay/zig/src/families/qwen3_5/kernels.zig", "native/luban/build.py", "scripts/native-training.ts", "pyproject.toml"], optional: ["build/native/luban-worker", "build/native/luban-worker.json", "build/native/luban-training", "build/native/luban-training.build.json"] },
  { id: "training", name: "训练引擎", patterns: ["backend/mlx-learning-*.ts", "src/luban/mlx_learning.py", "native/mlx/Sources/LubanMLXWorker/Training.swift", "native/mlx/Sources/LubanMLXWorker/TrainingCheck.swift", "native/Training.cpp", "native/ImageTraining.cpp", "native/asr/**", "native/Tokenizer.swift", "native/Worker.swift", "native/EvolutionObserve.swift", "native/EvolutionTail.h", "native/EvolutionRound.h", "native/AcceptedSourcePermissions.h", "scripts/build-native-evolution*.ts", "scripts/native-evolution-*.ts", "native/ASRTokenizer/**", "scripts/build-asr-training.py", "scripts/build-image-training.ts", "scripts/native-training.ts", ...["luban_training", "native_core_training", "evolution", "dynamic", "retention", "generation_checks", "review", "local_review", "training_checkpoint", "automatic", "upgrades", "media_learning", "image_learning", "native_image_training", "operation_learning", "native_operation_training", "native_video_training", "speech_learning"].map(p => `src/luban/${p}.py`), "pyproject.toml"], optional: ["build/native-evolution/worker.js", "build/native-evolution/build.json", "build/native/luban-evolution-observe", "build/native/luban-evolution-observe.build.json", "build/native/luban-training", "build/native/luban-training.build.json", "build/native/libluban_image_training.dylib", "build/native/libluban_image_training.dylib.build.json", "build/native/luban-asr-training", "build/native/luban-asr-training.build.json", "build/native/luban-asr-features", "build/native/luban-asr-features.build.json", "build/native/luban-asr-tokenizer", "build/native/luban-asr-tokenizer.build.json"] },
  { id: "decision", name: "快脑决策", patterns: [...["decision", "calibration", "external_brain"].map(p => `src/luban/${p}.py`), "pyproject.toml"] },
  { id: "vision", name: "图片理解", patterns: [...["vision", "runtime", "backbone", "coldpaging", "context", "image_learning", "native_image_training", "operation_learning", "native_operation_training", "native_video_training", "media_learning", "video_learning"].map(p => `src/luban/${p}.py`), "native/ImageTraining.cpp", "native/VideoFrames.swift", "scripts/build-image-training.ts", "pyproject.toml"], optional: ["build/native/libluban_image_training.dylib", "build/native/libluban_image_training.dylib.build.json"] },
  { id: "speech", name: "声音识别", patterns: ["src/luban/speech.py", "src/luban/speech_learning.py", "src/luban/media_learning.py", "native/asr/**", "native/Tokenizer.swift", "native/ASRTokenizer/**", "scripts/build-asr-training.py", "pyproject.toml"], optional: ["build/native/luban-asr-training", "build/native/luban-asr-training.build.json", "build/native/luban-asr-features", "build/native/luban-asr-features.build.json", "build/native/luban-asr-tokenizer", "build/native/luban-asr-tokenizer.build.json"] },
] as const;
export type Component = { version: string; source_sha256: string };
export type ComponentRegistry = { schema: 1; app: { version: string; build: number }; notes_sha256: string; components: Record<string, Component> };
export type ComponentManifest = { schema: 1; app: ComponentRegistry["app"]; components: { id: string; name: string; version: string }[]; pending_components: { name: string; status: string }[]; artifacts: Record<string, string> };
export type VersionPlan = { registry: ComponentRegistry; release: Release; changed: string[]; previous: string | null; originalRelease: string };
const sha = (raw: string | Uint8Array) => createHash("sha256").update(raw).digest("hex");
const need = (ok: unknown, message: string) => { if (!ok) throw Error("组件版本门禁失败：" + message); };
const encode = (value: unknown) => JSON.stringify(value, null, 2) + "\n";
const validVersion = (version: string) => validateRelease({ version, build: 1, notes: ["组件版本"] }).version;
function bump(version: string) {
  const parts = validVersion(version).split(".").map(Number); parts[2]++;
  return validVersion(parts.join("."));
}
async function optionalText(path: string) {
  try { return await readFile(path, "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return null; throw error; }
}
async function hashFiles(root: string, paths: string[]) {
  const hash = createHash("sha256");
  for (const path of [...new Set(paths)].sort()) {
    const info = await lstat(join(root, path));
    need(info.isFile() && !info.isSymbolicLink(), "禁止链接或特殊源文件：" + path);
    hash.update(path + "\0"); hash.update(sha(await readFile(join(root, path))) + "\0");
  }
  return hash.digest("hex");
}
async function scan(pattern: string, root: string) {
  const files: string[] = [];
  for await (const file of new Bun.Glob(pattern).scan({ cwd: root, onlyFiles: true, followSymlinks: false })) files.push(file);
  return files;
}
export async function sourceSnapshot(root: string) {
  const hashes: Record<string, string> = {};
  for (const definition of componentDefinitions) {
    const paths: string[] = [];
    for (const pattern of definition.patterns) {
      const matches = await scan(pattern, root);
      need(matches.length > 0, "缺少组件源码：" + pattern);
      paths.push(...matches.filter(p => !p.split("/").some(part => part.startsWith(".") || part === "__pycache__") && !p.endsWith(".pyc")));
    }
    const optional = "optional" in definition ? definition.optional : [], present: string[] = [];
    for (const path of optional) {
      try { await lstat(join(root, path)); present.push(path); }
      catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
    }
    for (let i = 0; i < optional.length; i += 2) {
      need(present.includes(optional[i]) === present.includes(optional[i + 1]), "引擎程序与构建回执必须同时存在：" + definition.id);
      if (present.includes(optional[i])) {
        const receipt = JSON.parse(await readFile(join(root, optional[i + 1]), "utf8"));
        need(receipt.binary_sha256 === sha(await readFile(join(root, optional[i]))), "引擎程序与构建回执SHA不匹配：" + definition.id);
      }
    }
    paths.push(...present);
    hashes[definition.id] = await hashFiles(root, paths);
  }
  return hashes;
}
function validateRegistry(value: unknown, allowLegacy = false): ComponentRegistry {
  const item = value as ComponentRegistry;
  need(item?.schema === 1 && item.components && item.app && /^[a-f0-9]{64}$/.test(item.notes_sha256), "版本登记格式无效");
  validateRelease({ ...item.app, notes: ["组件版本"] });
  const ids = JSON.stringify(Object.keys(item.components).sort());
  const expected = componentDefinitions.map(c => c.id).sort();
  need(ids === JSON.stringify(expected) || allowLegacy && ids === JSON.stringify(expected.filter(id => id !== "vision")), "组件登记缺失或包含未知组件");
  for (const entry of Object.values(item.components)) {
    validVersion(entry.version); need(/^[a-f0-9]{64}$/.test(entry.source_sha256), "源码指纹无效");
  }
  return item;
}
export function nextVersions(previous: ComponentRegistry | null, snapshot: Record<string, string>, release: Release) {
  release = validateRelease(release); if (previous) validateRegistry(previous, true);
  const changed: string[] = [], components: Record<string, Component> = {};
  for (const { id } of componentDefinitions) {
    need(/^[a-f0-9]{64}$/.test(snapshot[id]), "组件指纹缺失：" + id);
    const prior = previous?.components[id], different = !prior || prior.source_sha256 !== snapshot[id];
    if (different) changed.push(id);
    components[id] = { version: prior ? different ? bump(prior.version) : prior.version : "1.0.0", source_sha256: snapshot[id] };
  }
  const notes = sha(JSON.stringify(release.notes));
  if (previous) {
    need(compareVersions(release.version, previous.app.version) >= 0 && release.build >= previous.app.build, "禁止客户端版本或构建号回退");
    if (changed.length || notes !== previous.notes_sha256 || release.version !== previous.app.version || release.build !== previous.app.build) {
      if (compareVersions(release.version, previous.app.version) <= 0) release = { ...release, version: bump(previous.app.version) };
      if (release.build <= previous.app.build) release = { ...release, build: previous.app.build + 1 };
    }
  }
  validateRelease(release);
  return { registry: { schema: 1 as const, app: { version: release.version, build: release.build }, notes_sha256: notes, components }, release, changed };
}
export async function prepareVersions(root: string): Promise<VersionPlan> {
  const [previous, originalRelease, snapshot] = await Promise.all([optionalText(join(root, "component-versions.json")), readFile(join(root, "release.json"), "utf8"), sourceSnapshot(root)]);
  return { ...nextVersions(previous === null ? null : validateRegistry(JSON.parse(previous), true), snapshot, JSON.parse(originalRelease)), previous, originalRelease };
}
export async function frontendArtifacts(root: string) {
  const files = ["app.js", "index.html", "style.css", ...await scan("assets/**", root)];
  return Object.fromEntries(await Promise.all([...new Set(files)].sort().map(async file => [file, sha(await readFile(join(root, file)))])));
}
export function publicVersions(registry: ComponentRegistry, artifacts: Record<string, string>): ComponentManifest {
  return { schema: 1, app: registry.app, components: componentDefinitions.map(({ id, name }) => ({ id, name, version: registry.components[id].version })), pending_components: [{ name: "深度学习", status: "操作学习实现随训练引擎版本；真实质量、采纳与执行待验收" }], artifacts };
}
async function atomicJSON(path: string, value: unknown) {
  const temporary = path + "." + randomUUID() + ".tmp";
  try { await writeFile(temporary, encode(value), { flag: "wx" }); await rename(temporary, path); }
  finally { await rm(temporary, { force: true }); }
}
export async function commitVersions(root: string, plan: VersionPlan) {
  const snapshot = await sourceSnapshot(root);
  need(componentDefinitions.every(({ id }) => snapshot[id] === plan.registry.components[id].source_sha256), "构建中源码发生变化，请重新编译");
  need(await optionalText(join(root, "component-versions.json")) === plan.previous && await readFile(join(root, "release.json"), "utf8") === plan.originalRelease, "构建中版本记录发生变化，请重新编译");
  await atomicJSON(join(root, "release.json"), plan.release);
  await atomicJSON(join(root, "component-versions.json"), plan.registry);
}
export async function assertComponentVersions(root = resolve(import.meta.dir, ".."), web = join(root, "src/luban/web")) {
  const registry = validateRegistry(JSON.parse(await readFile(join(root, "component-versions.json"), "utf8")));
  const [snapshot, release, manifest, artifacts] = await Promise.all([sourceSnapshot(root), readFile(join(root, "release.json"), "utf8").then(s => validateRelease(JSON.parse(s))), readFile(join(web, "component-versions.json"), "utf8").then(JSON.parse), frontendArtifacts(web)]);
  need(componentDefinitions.every(({ id }) => snapshot[id] === registry.components[id].source_sha256), "源码已变化，请先执行 bun run build 更新对应版本");
  need(release.version === registry.app.version && release.build === registry.app.build && sha(JSON.stringify(release.notes)) === registry.notes_sha256, "客户端版本记录过期，请重新编译前端");
  need(JSON.stringify(manifest) === JSON.stringify(publicVersions(registry, artifacts)), "前端产物或组件版本不匹配，请重新编译前端");
  return registry;
}
export async function versionBuildLock(root: string) {
  const path = join(root, ".component-versions.lock");
  try { await mkdir(path); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "EEXIST") throw Error("前端版本构建锁已占用；确认构建结束后才可清理 .component-versions.lock"); throw error; }
  return () => rm(path, { recursive: true });
}
if (import.meta.main) {
  try { await assertComponentVersions(); console.log("COMPONENT VERSION GATE PASSED"); }
  catch (error) { console.error((error as Error).message); process.exitCode = 1; }
}
