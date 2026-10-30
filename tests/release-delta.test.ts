import { describe, expect, test } from "bun:test";
import { createHash, randomBytes } from "node:crypto";
import { chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir, arch } from "node:os";
import { join } from "node:path";
import { createDeltaPackage, deltaFiles, validDeltaPath } from "../scripts/release-delta";
import { validateManifest, reusablePackage, type Manifest } from "../scripts/release";

const digest = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const manifest: Manifest = { version: "0.3.0", build: 3, notes: ["模块更新"], platform: "macos", architecture: "arm64",
  url: "https://ai.maihh.net/updates/Luban-0.3.0-macos-arm64.zip", sha256: "a".repeat(64), fingerprint: "b".repeat(64), size: 1000 };
const delta = { from_version: "0.2.0", from_build: 2, url: "https://ai.maihh.net/updates/Luban-0.3.0-from-0.2.0-macos-arm64-delta.zip",
  sha256: "c".repeat(64), size: 300, removed: ["Contents/Resources/old.py"] };
const entry = (name: string, text: string) => ({ name: "Luban.app/Contents/" + name, sha256: digest(text), size: text.length, directory: false });

describe("精确基线模块更新", () => {
  test("旧清单兼容，新清单绑定版本、架构和不可变同域包名", () => {
    expect(validateManifest(manifest)).toEqual(manifest);
    expect(validateManifest({ ...manifest, delta }).delta).toEqual(delta);
    for (const patch of [null, { ...delta, from_version: "0.3.0" }, { ...delta, from_build: 3 }, { ...delta, size: 1000 },
      { ...delta, url: delta.url.replace("ai.maihh.net", "evil.example") }, { ...delta, url: delta.url + "?redirect=1" },
      { ...delta, url: delta.url.replace("arm64", "x64") }, { ...delta, url: delta.url.replace("from-0.2.0", "from-0.1.0") },
      { ...delta, sha256: "not-sha" }]) expect(() => validateManifest({ ...manifest, delta: patch })).toThrow();
  });
  test("删除清单拒绝危险路径、大小写别名及文件父子冲突", () => {
    for (const path of ["Contents", "Other/file", "Contents/../data", "Contents//file", "Contents/.hidden", "Contents/file/",
      "Contents/a\\b", "Contents/a:b", "Contents/a\u0000b", "Contents/Ａ.txt"])
      expect(validDeltaPath(path)).toBe(false);
    for (const removed of [["Contents/a", "Contents/a"], ["Contents/A", "Contents/a"], ["Contents/a", "Contents/a/file"], ["../data"]])
      expect(() => validateManifest({ ...manifest, delta: { ...delta, removed } })).toThrow();
    expect(validDeltaPath("Contents/_CodeSignature/CodeResources")).toBe(true);
  });
  test("只列出新增/变化文件和已删除文件，保留原始清单", () => {
    const before = [entry("same", "unchanged"), entry("changed", "old"), entry("removed", "old")];
    const after = [entry("same", "unchanged"), entry("changed", "new"), entry("added", "new")];
    const original = JSON.stringify(before);
    expect(deltaFiles(before, after)).toEqual({ changed: ["Contents/added", "Contents/changed"], removed: ["Contents/removed"] });
    expect(JSON.stringify(before)).toBe(original);
    expect(() => deltaFiles([entry("file", "old")], [entry("file/new", "new")])).toThrow("完整更新");
    expect(() => deltaFiles([], [entry("../unsafe", "new")])).toThrow();
    expect(deltaFiles([{ ...entry("mode-only", "same"), mode: 0o644 }], [{ ...entry("mode-only", "same"), mode: 0o755 }]))
      .toEqual({ changed: ["Contents/mode-only"], removed: [] });
  });
  test("缺少本地基线或架构不同不生成补丁，不下载基线", async () => {
    const directory = await mkdtemp(join(tmpdir(), "luban-delta-absent-"));
    try {
      const previous = { ...manifest, version: "0.2.0", build: 2, url: manifest.url.replace("0.3.0", "0.2.0") };
      expect(await createDeltaPackage("missing.app", manifest, previous, directory)).toBeUndefined();
      expect(await createDeltaPackage("missing.app", manifest, { ...previous, architecture: "x64" }, directory)).toBeUndefined();
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  test("复用更新包必须同时核验补丁", async () => {
    const directory = await mkdtemp(join(tmpdir(), "luban-delta-reuse-")), output = join(directory, "build/updates");
    try {
      await mkdir(output, { recursive: true });
      const bytes = "complete signed archive", patch = "patch";
      const value = { ...manifest, size: bytes.length, sha256: digest(bytes), delta: { ...delta, size: patch.length, sha256: digest(patch) } };
      await writeFile(join(output, "latest.json"), JSON.stringify(value));
      await writeFile(join(output, new URL(manifest.url).pathname.split("/").pop()!), bytes);
      const path = join(output, new URL(delta.url).pathname.split("/").pop()!);
      await writeFile(path, patch);
      expect(await reusablePackage(directory, value, value.fingerprint, "arm64")).toEqual(value);
      await writeFile(path, "other");
      await expect(reusablePackage(directory, value, value.fingerprint, "arm64")).rejects.toThrow("模块更新包被修改");
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  test.skipIf(process.platform !== "darwin")("真实签名应用差量能重建，未变资源不下载且基线不变", async () => {
    const directory = await mkdtemp(join(tmpdir(), "luban-delta-real-"));
    async function command(argv: string[]) {
      const child = Bun.spawn(argv, { stdin: "ignore", stdout: "pipe", stderr: "pipe" });
      const output = await new Response(child.stderr).text();
      if (await child.exited) throw Error(output);
    }
    const output = join(directory, "updates"), previousApp = join(directory, "old/Luban.app"), currentApp = join(directory, "new/Luban.app");
    try {
      await mkdir(output);
      const stable = randomBytes(256 * 1024);
      for (const [app, version, build] of [[previousApp, "0.2.0", 2], [currentApp, "0.3.0", 3]] as const) {
        await mkdir(join(app, "Contents/MacOS"), { recursive: true });
        await mkdir(join(app, "Contents/Resources"));
        await copyFile("/usr/bin/true", join(app, "Contents/MacOS/Luban"));
        await writeFile(join(app, "Contents/Info.plist"), `<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>net.maihh.luban.delta-test</string><key>CFBundleExecutable</key><string>Luban</string><key>CFBundlePackageType</key><string>APPL</string><key>CFBundleShortVersionString</key><string>${version}</string><key>CFBundleVersion</key><string>${build}</string></dict></plist>`);
        await writeFile(join(app, "Contents/Resources/release.json"), JSON.stringify({ version, build }));
        await writeFile(join(app, "Contents/Resources/stable.bin"), stable);
        await writeFile(join(app, "Contents/Resources/changed.txt"), "changed in " + version);
        await writeFile(join(app, "Contents/Resources/mode-only.txt"), "identical bytes");
        await chmod(join(app, "Contents/Resources/mode-only.txt"), app === previousApp ? 0o644 : 0o755);
        if (app === previousApp) await writeFile(join(app, "Contents/Resources/removed.txt"), "old resource");
        await command(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", app]);
      }
      const architecture = arch() as "arm64" | "x64";
      const previousPath = join(output, `Luban-0.2.0-macos-${architecture}.zip`), currentPath = join(output, `Luban-0.3.0-macos-${architecture}.zip`);
      for (const [app, path] of [[previousApp, previousPath], [currentApp, currentPath]])
        await command(["/usr/bin/ditto", "-c", "-k", "--norsrc", "--noextattr", "--keepParent", app, path]);
      const before = await readFile(previousPath), current = await readFile(currentPath);
      const value = { ...manifest, architecture, url: `https://ai.maihh.net/updates/Luban-0.3.0-macos-${architecture}.zip`, size: current.length, sha256: digest(current) };
      const previous = { ...value, version: "0.2.0", build: 2, url: value.url.replace("0.3.0", "0.2.0"), size: before.length, sha256: digest(before) };
      const patch = await createDeltaPackage(currentApp, value, previous, output);
      expect(patch).toBeDefined();
      expect(patch!.removed).toEqual(["Contents/Resources/removed.txt"]);
      expect(patch!.size).toBeLessThan(value.size / 2);
      const names = await Bun.$`/usr/bin/unzip -Z1 ${join(output, new URL(patch!.url).pathname.split("/").pop()!)}`.text();
      expect(names).not.toContain("stable.bin");
      expect(names).toContain("_CodeSignature/CodeResources");
      expect(names).toContain("Contents/Info.plist");
      expect(names).toContain("Contents/Resources/mode-only.txt");
      expect(await readFile(previousPath)).toEqual(before);
      expect(await readFile(join(previousApp, "Contents/Resources/removed.txt"), "utf8")).toBe("old resource");
      expect(validateManifest({ ...value, delta: patch }).delta).toEqual(patch);
    } finally { await rm(directory, { recursive: true, force: true }); }
  }, 120_000);
});
