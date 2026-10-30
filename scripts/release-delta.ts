import { createHash, randomUUID } from "node:crypto";
import { copyFile, link, lstat, mkdir, mkdtemp, readFile, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import type { Delta, Manifest } from "./release";

type Entry = { name: string; sha256: string; size: number; directory: boolean; mode?: number };
const sha = (data: Uint8Array) => createHash("sha256").update(data).digest("hex");
const need = (ok: unknown, message: string) => { if (!ok) throw Error("模块更新包：" + message); };
export const deltaArchiveName = (version: string, from: string, architecture: string) => `Luban-${version}-from-${from}-macos-${architecture}-delta.zip`;
export function validDeltaPath(path: string) {
  return typeof path === "string" && path.startsWith("Contents/") && path.length <= 512
    && path.normalize("NFKC") === path && !/[\\:\u0000-\u001f\u007f-\u009f]/.test(path)
    && path.split("/").every(part => part !== "" && part !== "." && part !== ".." && !part.startsWith("."));
}
export function deltaFiles(before: Entry[], after: Entry[]) {
  function files(entries: Entry[]) {
    const result = new Map<string, Entry>();
    for (const entry of entries) {
      if (entry.directory) continue;
      need(entry.name.startsWith("Luban.app/") && validDeltaPath(entry.name.slice(10))
        && /^[a-f0-9]{64}$/.test(entry.sha256) && Number.isSafeInteger(entry.size) && entry.size >= 0, "文件清单无效");
      const name = entry.name.slice(10); need(!result.has(name), "重复文件路径"); result.set(name, entry);
    }
    return result;
  }
  const old = files(before), next = files(after);
  // File/directory type transitions cannot be safely overlaid; use the complete fallback instead.
  const paths = [...new Set([...old.keys(), ...next.keys()])].sort();
  need(!paths.some((name, index) => paths[index + 1]?.startsWith(name + "/")), "文件与目录类型发生变化，需要完整更新");
  return { changed: [...next.keys()].filter(name => old.get(name)?.sha256 !== next.get(name)!.sha256
      || Boolean((old.get(name)?.mode ?? 0) & 0o111) !== Boolean((next.get(name)?.mode ?? 0) & 0o111)).sort(),
    removed: [...old.keys()].filter(name => !next.has(name)).sort() };
}

async function run(command: string[]) {
  const child = Bun.spawn(command, { stdin: "ignore", stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  need(code === 0, stderr.trim() || "系统归档检查失败"); return stdout;
}
async function inventory(app: string) {
  const entries: Entry[] = [];
  async function visit(relative: string) {
    const path = join(app, relative), info = await lstat(path);
    need(!info.isSymbolicLink(), "禁止符号链接");
    if (info.isDirectory()) { for (const name of await readdir(path)) await visit(relative ? relative + "/" + name : name); }
    else {
      need(info.isFile() && validDeltaPath(relative), "禁止未知类型或危险路径");
      entries.push({ name: "Luban.app/" + relative, size: info.size, sha256: sha(await readFile(path)), directory: false, mode: info.mode });
    }
  }
  await visit(""); return entries;
}

/** Build an exact signed-app overlay only when the approved preceding archive is already local. */
export async function createDeltaPackage(currentApp: string, current: Manifest, previous: Manifest | undefined, output: string,
  projectRoot = resolve(import.meta.dir, "..")): Promise<Delta | undefined> {
  if (!previous || previous.version === current.version || previous.architecture !== current.architecture) return;
  const baseline = join(output, basename(new URL(previous.url).pathname));
  let info;
  try { info = await lstat(baseline); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
  need(info.isFile() && !info.isSymbolicLink() && info.size === previous.size, "上版归档类型或大小不符");
  const temporary = await mkdtemp(join(tmpdir(), "luban-release-delta-"));
  const archive = join(output, `.${randomUUID()}-delta.zip`);
  try {
    const inspector = join(temporary, "inspector");
    await run(["/usr/bin/xcrun", "--sdk", "macosx", "swiftc", join(projectRoot, "native/UpdateArchive.swift"), join(projectRoot, "native/ReleaseInspection.swift"), "-O", "-o", inspector]);
    await run(["/usr/bin/codesign", "--force", "--sign", "-", inspector]);
    const baseStage = join(temporary, "base");
    const receipt = JSON.parse(await run([inspector, baseline, baseStage, previous.sha256]));
    need(receipt.schema === "luban-native-release-inspection-v1" && receipt.sha256 === previous.sha256, "上版归档绑定无效");
    const baseApp = join(baseStage, "Luban.app");
    await run(["/usr/bin/codesign", "--verify", "--deep", "--strict", baseApp]);
    const release = JSON.parse(await readFile(join(baseApp, "Contents/Resources/release.json"), "utf8"));
    need(release.version === previous.version && release.build === previous.build, "上版应用版本不匹配");
    const before = await Promise.all(receipt.entries.map(async (entry: Entry) => ({ ...entry,
      mode: entry.directory ? 0 : (await lstat(join(baseStage, entry.name))).mode })));
    const after = await inventory(currentApp);
    let plan;
    try { plan = deltaFiles(before, after); }
    catch (error) { if (String(error).includes("文件与目录类型发生变化")) return; throw error; }
    need(plan.changed.length > 0, "新版本没有变化的运行文件");
    const patchApp = join(temporary, "patch/Luban.app");
    for (const name of plan.changed) {
      await mkdir(dirname(join(patchApp, name)), { recursive: true });
      await copyFile(join(currentApp, name), join(patchApp, name));
    }
    await run(["/usr/bin/ditto", "-c", "-k", "--norsrc", "--noextattr", "--keepParent", patchApp, archive]);
    const bytes = await readFile(archive), digest = sha(bytes);
    // Never advertise a patch that costs as much as the complete download.
    if (bytes.length >= current.size || bytes.length > 512 * 1024 * 1024) return;
    const extracted = join(temporary, "patch-validated");
    const patchReceipt = JSON.parse(await run([inspector, archive, extracted, digest]));
    const actual = patchReceipt.entries.filter((entry: Entry) => !entry.directory).map((entry: Entry) => entry.name.slice(10)).sort();
    need(JSON.stringify(actual) === JSON.stringify(plan.changed), "模块归档成员不匹配");
    for (const name of plan.removed) await rm(join(baseApp, name));
    for (const name of plan.changed) {
      await mkdir(dirname(join(baseApp, name)), { recursive: true });
      await copyFile(join(extracted, "Luban.app", name), join(baseApp, name));
    }
    const reconstructed = deltaFiles(after, await inventory(baseApp));
    need(reconstructed.changed.length === 0 && reconstructed.removed.length === 0, "重建应用不是目标版本");
    await run(["/usr/bin/codesign", "--verify", "--deep", "--strict", baseApp]);
    const final = await stat(baseline);
    need(final.size === info.size && final.mtimeMs === info.mtimeMs && final.ino === info.ino, "上版归档在打包时发生变化");
    const name = deltaArchiveName(current.version, previous.version, current.architecture), destination = join(output, name);
    try { await link(archive, destination); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      need(sha(await readFile(destination)) === digest, "禁止覆盖同版本不同模块更新包");
    }
    return { from_version: previous.version, from_build: previous.build, url: `https://ai.maihh.net/updates/${name}`,
      sha256: digest, size: bytes.length, removed: plan.removed };
  } finally { await rm(archive, { force: true }); await rm(temporary, { recursive: true, force: true }); }
}
