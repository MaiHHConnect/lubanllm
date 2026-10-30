import { createHash, randomUUID } from "node:crypto";
import { link, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { arch } from "node:os";
import { basename, resolve } from "node:path";
import { assertNoInternalFiles } from "./internal-only";
import { assertReleasePayload, licenseFiles } from "./release-payload";
import { assertComponentVersions } from "./component-versions";
import { createDeltaPackage, deltaArchiveName, validDeltaPath } from "./release-delta";

export type Release = { version: string; build: number; notes: string[] };
export type Delta = { from_version: string; from_build: number; url: string; sha256: string; size: number; removed: string[] };
export type Manifest = Release & {
  url: string; sha256: string; size: number; fingerprint: string;
  platform: "macos"; architecture: "arm64" | "x64";
  delta?: Delta;
};
const root = resolve(import.meta.dir, "..");
const updateOrigin = "https://ai.maihh.net";
const latestURL = `${updateOrigin}/updates/latest.json`;
const remoteUpdates = process.env.LUBAN_RELEASE_REMOTE_DIR ?? "";
const sha256 = (data: string | Uint8Array) => createHash("sha256").update(data).digest("hex");

export async function preserveArchive(temporary: string, archive: string) {
  const digest = sha256(await readFile(temporary));
  try { await link(temporary, archive); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    if (sha256(await readFile(archive)) !== digest) throw new Error("该版本更新包已存在且字节不同；请提升版本号和构建号");
  }
  return digest;
}

export function validateRelease(value: unknown): Release {
  const item = value as Release;
  if (!item || typeof item.version !== "string" || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(item.version)
      || item.version.split(".").some(n => !Number.isSafeInteger(Number(n)) || Number(n) >= 1_000_000)) throw new Error("版本必须是 major.minor.patch，每段小于 1000000");
  if (!Number.isSafeInteger(item.build) || item.build < 1) throw new Error("构建号必须是正整数");
  if (!Array.isArray(item.notes) || item.notes.length < 1 || item.notes.length > 5
      || item.notes.some(n => typeof n !== "string" || !n.trim() || /[\u0000-\u001f\u007f-\u009f]/.test(n))
      || Array.from(item.notes.join("")).length > 300) throw new Error("更新内容限 1–5 条，总计不超过 300 字");
  return { version: item.version, build: item.build, notes: [...item.notes] };
}

export function compareVersions(a: string, b: string): number {
  const left = a.split(".").map(Number), right = b.split(".").map(Number);
  for (let i = 0; i < 3; i++) if (left[i] !== right[i]) return left[i] > right[i] ? 1 : -1;
  return 0;
}

export function validateManifest(value: unknown): Manifest {
  const release = validateRelease(value);
  const item = value as Manifest;
  const url = new URL(item.url);
  if (url.origin !== updateOrigin || !/^\/updates\/Luban-[\d.]+-macos-(arm64|x64)\.zip$/.test(url.pathname)
      || url.search || url.hash || url.username || url.password) throw new Error("更新包必须位于 ai.maihh.net/updates");
  if (!/^[a-f0-9]{64}$/.test(item.sha256) || !/^[a-f0-9]{64}$/.test(item.fingerprint)
      || !Number.isSafeInteger(item.size) || item.size < 1 || item.size > 512 * 1024 * 1024 || item.platform !== "macos"
      || !["arm64", "x64"].includes(item.architecture)
      || basename(url.pathname) !== `Luban-${release.version}-macos-${item.architecture}.zip`) throw new Error("发布清单无效");
  let delta: Delta | undefined;
  if (item.delta !== undefined) {
    const patch = item.delta;
    validateRelease({ version: patch?.from_version, build: patch?.from_build, notes: ["上版"] });
    const location = new URL(patch.url);
    if (compareVersions(patch.from_version, release.version) >= 0 || patch.from_build >= release.build
      || location.origin !== updateOrigin || location.search || location.hash || location.username || location.password
      || location.pathname !== "/updates/" + deltaArchiveName(release.version, patch.from_version, item.architecture)
      || !/^[a-f0-9]{64}$/.test(patch.sha256) || !Number.isSafeInteger(patch.size) || patch.size < 1 || patch.size >= item.size
      || !Array.isArray(patch.removed) || patch.removed.length > 10_000 || patch.removed.some(path => !validDeltaPath(path))
      || new Set(patch.removed.map(path => path.toLowerCase())).size !== patch.removed.length
      || patch.removed.some(path => patch.removed.some(other => path !== other && other.startsWith(path + "/"))))
      throw new Error("模块更新清单无效");
    delta = { from_version: patch.from_version, from_build: patch.from_build, url: patch.url, sha256: patch.sha256, size: patch.size, removed: [...patch.removed] };
  }
  return { ...release, url: item.url, sha256: item.sha256, size: item.size,
    fingerprint: item.fingerprint, platform: "macos", architecture: item.architecture, ...(delta ? { delta } : {}) };
}

export function assertReleaseProgress(current: Release, fingerprint: string, previous?: Manifest, initial = false) {
  validateRelease(current);
  if (!previous) {
    if (!initial) throw new Error("需要 --previous 上次发布清单；第一次发布必须显式 --initial");
    if (compareVersions(current.version, "0.1.0") <= 0 || current.build <= 1) throw new Error("首次更新必须高于旧客户端 0.1.0 / 构建 1");
    return;
  }
  const order = compareVersions(current.version, previous.version);
  if (order < 0 || current.build < previous.build) throw new Error("禁止回退客户端版本或构建号");
  const changed = fingerprint !== previous.fingerprint;
  const update = changed || order > 0 || current.build > previous.build;
  if (update && (order <= 0 || current.build <= previous.build)) throw new Error("客户端或发布内容变更必须同时提升版本号与构建号");
}

export async function readRelease(projectRoot = root) {
  return validateRelease(JSON.parse(await readFile(resolve(projectRoot, "release.json"), "utf8")));
}

export async function payloadFingerprint(projectRoot: string, release: Release): Promise<string> {
  const hash = createHash("sha256");
  async function walk(relative: string) {
    for (const entry of (await readdir(resolve(projectRoot, relative), { withFileTypes: true })).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      if (entry.name === "__pycache__" || entry.name === ".DS_Store" || entry.name.endsWith(".pyc")) continue;
      const path = `${relative}/${entry.name}`;
      if (entry.isDirectory()) await walk(path);
      else if (entry.isFile()) { hash.update(`${path}\0`); hash.update(await readFile(resolve(projectRoot, path))); hash.update("\0"); }
    }
  }
  for (const directory of ["native", "frontend", "backend", "src/luban"]) await walk(directory);
  for (const path of ["scripts/build-app.ts", "scripts/release.ts", "scripts/release-delta.ts", "scripts/internal-only.ts", "scripts/release-payload.ts", "scripts/release-macho.ts", "scripts/component-versions.ts", "component-versions.json", "package.json", ...Object.keys(licenseFiles)]) {
    hash.update(`${path}\0`); hash.update(await readFile(resolve(projectRoot, path))); hash.update("\0");
  }
  hash.update(JSON.stringify({ notes: release.notes }));
  return hash.digest("hex");
}

async function run(command: string[], cwd = root) {
  const process = Bun.spawn(command, { cwd, stdin: "ignore", stdout: "inherit", stderr: "inherit" });
  if (await process.exited !== 0) throw new Error(`命令失败：${command[0]}`);
}

export async function reusablePackage(projectRoot: string, release: Release, fingerprint: string, architecture: "arm64" | "x64") {
  const output = resolve(projectRoot, "build/updates");
  let contents: string;
  try { contents = await readFile(resolve(output, "latest.json"), "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  const manifest = validateManifest(JSON.parse(contents));
  if (manifest.version !== release.version || manifest.build !== release.build || manifest.architecture !== architecture) return undefined;
  if (manifest.fingerprint !== fingerprint || JSON.stringify(manifest.notes) !== JSON.stringify(release.notes))
    throw new Error("已有同版本本地包，但客户端或更新内容已变更；请同时提升版本号与构建号");
  const archive = resolve(output, basename(new URL(manifest.url).pathname));
  if ((await stat(archive)).size !== manifest.size)
    throw new Error("本地更新包被修改，SHA-256 或大小校验失败；禁止复用");
  const bytes = await readFile(archive);
  if (bytes.byteLength !== manifest.size || sha256(bytes) !== manifest.sha256)
    throw new Error("本地更新包被修改，SHA-256 或大小校验失败；禁止复用");
  if (manifest.delta) {
    const patch = await readFile(resolve(output, basename(new URL(manifest.delta.url).pathname)));
    if (patch.byteLength !== manifest.delta.size || sha256(patch) !== manifest.delta.sha256)
      throw new Error("本地模块更新包被修改，SHA-256 或大小校验失败；禁止复用");
  }
  return manifest;
}

async function packageRelease(release: Release, previous?: Manifest, initial = false): Promise<Manifest> {
  await assertNoInternalFiles(resolve(root, "release.json"));
  // Frontend compilation can bump the application version; read it before naming archives.
  await run(["bun", "run", "build"]);
  await assertComponentVersions(root);
  release = await readRelease(root);
  const sourceFingerprint = await payloadFingerprint(root, release);
  assertReleaseProgress(release, sourceFingerprint, previous, initial);
  const architecture = arch();
  if (architecture !== "arm64" && architecture !== "x64") throw new Error("仅支持 macOS arm64 / x64");
  const existing = await reusablePackage(root, release, sourceFingerprint, architecture);
  if (existing) {
    await assertReleasePayload(resolve(root, "build/updates", basename(new URL(existing.url).pathname)), root);
    await assertNoInternalFiles(resolve(root, "build/updates/latest.json"));
    console.log(`复用已校验更新包：${existing.url}`); return existing;
  }
  await run(["bun", "run", "app:build"]);
  release = await readRelease(root);
  await assertComponentVersions(root);
  // The generated browser bundle is part of the actual payload and must also pass the watchdog.
  const fingerprint = await payloadFingerprint(root, release);
  assertReleaseProgress(release, fingerprint, previous, initial);
  const output = resolve(root, "build/updates");
  await mkdir(output, { recursive: true });
  const name = `Luban-${release.version}-macos-${architecture}.zip`;
  const temporary = resolve(output, `.${randomUUID()}.zip`);
  const archive = resolve(output, name);
  try {
    await assertReleasePayload(resolve(root, "build/Luban.app"), root);
    await run(["/usr/bin/ditto", "-c", "-k", "--norsrc", "--noextattr", "--keepParent", resolve(root, "build/Luban.app"), temporary]);
    const names = await Bun.$`/usr/bin/unzip -Z1 ${temporary}`.text();
    if (!names.trim() || names.trimEnd().split("\n").some(name => !name.startsWith("Luban.app/") || name.split("/").includes("..")))
      throw new Error("更新归档包含应用目录之外的文件");
    await assertReleasePayload(temporary, root);
    // link() fails if the versioned archive already exists; never replace published bytes.
    const digest = await preserveArchive(temporary, archive);
    let manifest = validateManifest({ ...release, platform: "macos", architecture,
      url: `${updateOrigin}/updates/${name}`, sha256: digest, size: (await stat(archive)).size, fingerprint });
    const delta = await createDeltaPackage(resolve(root, "build/Luban.app"), manifest, previous, output, root);
    if (delta) manifest = validateManifest({ ...manifest, delta });
    const manifestTemp = resolve(output, `.${randomUUID()}.json`);
    await writeFile(manifestTemp, JSON.stringify(manifest, null, 2) + "\n");
    try {
      await assertNoInternalFiles(manifestTemp);
      await rename(manifestTemp, resolve(output, "latest.json"));
    } finally { await rm(manifestTemp, { force: true }); }
    console.log(`更新包已生成：${archive}`);
    return manifest;
  } finally { await rm(temporary, { force: true }); }
}

const shellQuote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
async function uploadRelease(release: Release, initial: boolean, remoteDirectory: string) {
  if (!remoteDirectory || !/^\/[a-zA-Z0-9_./-]+$/.test(remoteDirectory) || remoteDirectory.includes(".."))
    throw new Error("需要 --remote-dir 指定已配置的服务器更新目录");
  const response = await fetch(latestURL, { redirect: "error", signal: AbortSignal.timeout(20_000) });
  let previous: Manifest | undefined, baselineDigest = "absent";
  if (response.ok) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    baselineDigest = sha256(bytes);
    previous = validateManifest(JSON.parse(new TextDecoder().decode(bytes)));
  } else if (!(response.status === 404 && initial)) throw new Error(`读取线上清单失败：HTTP ${response.status}；仅首次 404 可使用 --initial`);
  const manifest = await packageRelease(release, previous, initial);
  await run([resolve(root, ".venv/bin/python"), "scripts/release-secrets.py",
    resolve(root, "build/updates", basename(new URL(manifest.url).pathname)),
    ...(manifest.delta ? [resolve(root, "build/updates", basename(new URL(manifest.delta.url).pathname))] : [])]);
  // Public releases remain subject to the project's research release gate.
  await run([resolve(root, ".venv/bin/python"), "scripts/paper_gate.py", "--release"]);
  const sshOptions = ["-p", process.env.LUBAN_RELEASE_SSH_PORT ?? (() => { throw Error("LUBAN_RELEASE_SSH_PORT is required"); })(), "-o", "BatchMode=yes", "-o", "ControlPath=/tmp/luban-ssh-%r-%h-%p"];
  const host = process.env.LUBAN_RELEASE_HOST ?? (() => { throw Error("LUBAN_RELEASE_HOST is required"); })();
  const id = randomUUID(), archiveName = basename(new URL(manifest.url).pathname);
  await assertReleasePayload(resolve(root, "build/updates", archiveName), root);
  const manifestTemp = `${remoteDirectory}/.${id}.json`;
  const artifacts = [manifest, ...(manifest.delta ? [manifest.delta] : [])].map((artifact, index) => ({
    ...artifact, name: basename(new URL(artifact.url).pathname), temporary: `${remoteDirectory}/.${id}-${index}.zip`,
  }));
  const localManifest = resolve(root, `build/updates/.${id}.json`);
  const publishedManifest = JSON.stringify(manifest, null, 2) + "\n";
  await writeFile(localManifest, publishedManifest);
  try { await assertNoInternalFiles(localManifest); }
  catch (error) { await rm(localManifest, { force: true }); throw error; }
  await run(["ssh", ...sshOptions, host, `mkdir -p ${shellQuote(remoteDirectory)} && test -w ${shellQuote(remoteDirectory)}`]);
  try {
    for (const [local, remote] of [...artifacts.map(artifact => [resolve(root, "build/updates", artifact.name), artifact.temporary]),
      [localManifest, manifestTemp]])
      await run(["scp", "-P", process.env.LUBAN_RELEASE_SSH_PORT ?? (() => { throw Error("LUBAN_RELEASE_SSH_PORT is required"); })(), "-o", "BatchMode=yes", "-o", "ControlPath=/tmp/luban-ssh-%r-%h-%p", local, `${host}:${remote}`]);
    const script = `set -eu
publish_dir=$1
manifest_temp=$2
expected_baseline=$3
shift 3
lock_dir="$publish_dir/.publish-lock"
mkdir "$lock_dir" || { echo '已有发布正在进行'; exit 1; }
trap 'rmdir "$lock_dir"' EXIT
actual_baseline=absent
if [ -f "$publish_dir/latest.json" ]; then actual_baseline=$(sha256sum "$publish_dir/latest.json" | cut -d ' ' -f 1); fi
[ "$actual_baseline" = "$expected_baseline" ] || { echo '线上版本已变化，请重新发布'; exit 1; }
# Verify every artifact before publishing either package or switching the manifest.
validate_artifacts() {
  while [ "$#" -gt 0 ]; do
    [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$3" ] || { echo '服务器更新包 SHA-256 校验失败'; exit 1; }
    if [ -e "$publish_dir/$2" ]; then
      [ "$(sha256sum "$publish_dir/$2" | cut -d ' ' -f 1)" = "$3" ] || { echo '禁止覆盖同版本不同更新包'; exit 1; }
    fi
    shift 3
  done
}
validate_artifacts "$@"
while [ "$#" -gt 0 ]; do
  if [ -e "$publish_dir/$2" ]; then rm "$1"; else chmod 644 "$1"; mv "$1" "$publish_dir/$2"; fi
  shift 3
done
chmod 644 "$manifest_temp"
mv "$manifest_temp" "$publish_dir/latest.json"
`;
    const args = [remoteDirectory, manifestTemp, baselineDigest, ...artifacts.flatMap(artifact => [artifact.temporary, artifact.name, artifact.sha256])];
    const process = Bun.spawn(["ssh", ...sshOptions, host, `sh -s -- ${args.map(shellQuote).join(" ")}`],
      { cwd: root, stdin: "pipe", stdout: "inherit", stderr: "inherit" });
    process.stdin.write(script); process.stdin.end();
    if (await process.exited !== 0) throw new Error("服务器原子发布失败");
    const servedManifest = await fetch(latestURL, { redirect: "error", cache: "no-store", signal: AbortSignal.timeout(20_000) });
    if (!servedManifest.ok || sha256(new Uint8Array(await servedManifest.arrayBuffer())) !== sha256(publishedManifest))
      throw new Error("文件已上传，但 HTTPS 最新清单校验失败，请检查静态文件路由或缓存");
    for (const artifact of artifacts) {
      const servedArchive = await fetch(artifact.url, { method: "HEAD", redirect: "error", cache: "no-store", signal: AbortSignal.timeout(20_000) });
      if (!servedArchive.ok || Number(servedArchive.headers.get("content-length")) !== artifact.size)
        throw new Error("文件已上传，但 HTTPS 更新包大小校验失败");
    }
    console.log(`更新已发布：${latestURL}`);
  } finally {
    await rm(localManifest, { force: true });
    try { await run(["ssh", ...sshOptions, host, `rm -f ${[manifestTemp, ...artifacts.map(artifact => artifact.temporary)].map(shellQuote).join(" ")}`]); }
    catch { console.error("远端临时文件清理失败，请检查 SSH 连接"); }
  }
}

if (import.meta.main) {
  try {
    const [command = "check", ...args] = Bun.argv.slice(2);
    for (const name of ["--previous", "--remote-dir"]) {
      const index = args.indexOf(name);
      if (index >= 0 && (!args[index + 1] || args[index + 1].startsWith("--"))) throw new Error(`${name} 缺少参数`);
    }
    const flagValue = (name: string) => { const position = args.indexOf(name); return position < 0 ? undefined : args[position + 1]; };
    if (!args.every((arg, index) => arg === "--initial" || ["--previous", "--remote-dir"].includes(arg)
        || (index > 0 && ["--previous", "--remote-dir"].includes(args[index - 1])))) throw new Error("未知发布参数");
    const release = await readRelease();
    const previousPath = flagValue("--previous");
    const previous = previousPath ? validateManifest(JSON.parse(await readFile(resolve(previousPath), "utf8"))) : undefined;
    const initial = args.includes("--initial");
    if (previous && initial) throw new Error("--previous 与 --initial 不能同时使用");
    if (command === "upload") {
      if (previous) throw new Error("upload 自动使用 HTTPS 线上清单，不接受 --previous");
      await uploadRelease(release, initial, flagValue("--remote-dir") ?? remoteUpdates);
    }
    else if (command === "package") await packageRelease(release, previous, initial);
    else if (command === "check") {
      await assertComponentVersions(root);
      assertReleaseProgress(release, await payloadFingerprint(root, release), previous, initial);
      console.log(`版本看门狗通过：${release.version} / 构建 ${release.build}`);
    } else throw new Error("使用 check、package 或 upload");
  } catch (error) { console.error((error as Error).message); process.exitCode = 1; }
}
