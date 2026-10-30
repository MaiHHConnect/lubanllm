import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import { assertReleaseProgress, payloadFingerprint, preserveArchive, reusablePackage, validateManifest, validateRelease, type Manifest } from "../scripts/release";
import { licenseFiles } from "../scripts/release-payload";

const release = { version: "0.2.0", build: 2, notes: ["修复设置"] };
const previous: Manifest = { ...release, url: "https://ai.maihh.net/updates/Luban-0.2.0-macos-arm64.zip",
  sha256: "a".repeat(64), fingerprint: "b".repeat(64), size: 42, platform: "macos", architecture: "arm64" };

describe("发布版本看门狗", () => {
  test("更新内容上限按 Unicode 字符计算", () => {
    expect(validateRelease({ ...release, notes: ["😀".repeat(300)] }).notes[0]).toHaveLength(600);
    expect(() => validateRelease({ ...release, notes: ["字".repeat(301)] })).toThrow("300");
    expect(() => validateRelease({ ...release, notes: Array(6).fill("更新") })).toThrow("1–5");
    expect(() => validateRelease({ ...release, notes: [" "] })).toThrow();
    expect(() => validateRelease({ ...release, notes: ["更新\n另一条"] })).toThrow();
    expect(() => validateRelease({ ...release, notes: ["更新\u0085另一条"] })).toThrow();
  });
  test("拒绝不合法版本和构建号", () => {
    for (const version of ["01.2.0", "0.2", "0.2.0-beta", "<script>", "9007199254740992.0.0", "0.1000000.0"])
      expect(() => validateRelease({ ...release, version })).toThrow();
    for (const build of [0, -1, 2.5, Number.MAX_SAFE_INTEGER + 1])
      expect(() => validateRelease({ ...release, build })).toThrow();
  });
  test("首次发布要求显式声明且高于已有客户端", () => {
    expect(() => assertReleaseProgress(release, "c".repeat(64))).toThrow("--previous");
    expect(() => assertReleaseProgress({ ...release, version: "0.1.0" }, "c".repeat(64), undefined, true)).toThrow("旧客户端");
    expect(() => assertReleaseProgress(release, "c".repeat(64), undefined, true)).not.toThrow();
  });
  test("代码变更必须同时提升版本与构建号", () => {
    expect(() => assertReleaseProgress(release, "c".repeat(64), previous)).toThrow("同时提升");
    expect(() => assertReleaseProgress({ ...release, version: "0.3.0" }, "c".repeat(64), previous)).toThrow();
    expect(() => assertReleaseProgress({ ...release, build: 3 }, "c".repeat(64), previous)).toThrow();
    expect(() => assertReleaseProgress({ ...release, version: "0.3.0", build: 3 }, "c".repeat(64), previous)).not.toThrow();
    expect(() => assertReleaseProgress(release, previous.fingerprint, previous)).not.toThrow();
  });
  test("无代码变更也禁止版本回退或更新时复用构建号", () => {
    expect(() => assertReleaseProgress({ ...release, version: "0.1.9", build: 3 }, previous.fingerprint, previous)).toThrow("回退");
    expect(() => assertReleaseProgress({ ...release, version: "0.3.0" }, previous.fingerprint, previous)).toThrow();
  });
  test("清单限制托管域名、平台、架构和完整性字段", () => {
    expect(validateManifest(previous)).toEqual(previous);
    for (const url of ["http://ai.maihh.net/updates/Luban-0.2.0-macos-arm64.zip", "https://evil.test/updates/Luban-0.2.0-macos-arm64.zip",
      "https://ai.maihh.net/updates/Luban-0.2.0-macos-arm64.zip?q=1", "https://ai.maihh.net/updates/Luban-0.3.0-macos-arm64.zip"])
      expect(() => validateManifest({ ...previous, url })).toThrow();
    expect(() => validateManifest({ ...previous, architecture: "x64" })).toThrow();
    expect(() => validateManifest({ ...previous, sha256: "bad" })).toThrow();
    expect(() => validateManifest({ ...previous, size: 512 * 1024 * 1024 + 1 })).toThrow();
  });
  test("同版本归档只能重用相同字节，禁止覆盖", async () => {
    const directory = await mkdtemp(join(tmpdir(), "luban-release-"));
    try {
      const temporary = join(directory, "temporary.zip"), archive = join(directory, "release.zip");
      await writeFile(temporary, "original payload");
      const digest = await preserveArchive(temporary, archive);
      expect(await preserveArchive(temporary, archive)).toBe(digest);
      await rm(temporary);
      await writeFile(temporary, "changed payload");
      await expect(preserveArchive(temporary, archive)).rejects.toThrow("字节不同");
      expect(await readFile(archive, "utf8")).toBe("original payload");
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  test("指纹覆盖实际客户端和更新内容，忽略 Python 缓存", async () => {
    const directory = await mkdtemp(join(tmpdir(), "luban-fingerprint-"));
    try {
      for (const name of ["native", "frontend", "backend", "src/luban", "scripts"])
        await mkdir(join(directory, name), { recursive: true });
      for (const name of ["scripts/build-app.ts", "scripts/release.ts", "scripts/release-delta.ts", "scripts/internal-only.ts", "scripts/release-payload.ts", "scripts/release-macho.ts", "scripts/component-versions.ts", "component-versions.json", "package.json", "src/luban/server.py", ...Object.keys(licenseFiles)]) {
        await mkdir(dirname(join(directory, name)), { recursive: true });
        await writeFile(join(directory, name), "initial");
      }
      const original = await payloadFingerprint(directory, release);
      await writeFile(join(directory, "backend/server.ts"), "Bun data service");
      expect(await payloadFingerprint(directory, release)).not.toBe(original);
      await rm(join(directory, "backend/server.ts"));
      await mkdir(join(directory, "src/luban/__pycache__"));
      await writeFile(join(directory, "src/luban/__pycache__/server.pyc"), "cache");
      expect(await payloadFingerprint(directory, { ...release, version: "0.3.0", build: 3 })).toBe(original);
      expect(await payloadFingerprint(directory, { ...release, notes: ["不同更新内容"] })).not.toBe(original);
      await writeFile(join(directory, "scripts/internal-only.ts"), "changed publication guard");
      expect(await payloadFingerprint(directory, release)).not.toBe(original);
      await writeFile(join(directory, "scripts/internal-only.ts"), "initial");
      await writeFile(join(directory, "scripts/release-payload.ts"), "changed payload policy");
      expect(await payloadFingerprint(directory, release)).not.toBe(original);
      await writeFile(join(directory, "scripts/release-payload.ts"), "initial");
      await writeFile(join(directory, "scripts/release-macho.ts"), "changed native payload inspector");
      expect(await payloadFingerprint(directory, release)).not.toBe(original);
      await writeFile(join(directory, "scripts/release-macho.ts"), "initial");
      await writeFile(join(directory, "LICENSE"), "changed required license");
      expect(await payloadFingerprint(directory, release)).not.toBe(original);
      await writeFile(join(directory, "LICENSE"), "initial");
      await writeFile(join(directory, "src/luban/server.py"), "changed backend");
      expect(await payloadFingerprint(directory, release)).not.toBe(original);
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  test("已打包版本可直接复用，改源码或篡改更新包均拒绝", async () => {
    const directory = await mkdtemp(join(tmpdir(), "luban-reuse-"));
    try {
      expect(await reusablePackage(directory, release, previous.fingerprint, "arm64")).toBeUndefined();
      const output = join(directory, "build/updates"), archive = join(output, "Luban-0.2.0-macos-arm64.zip");
      await mkdir(output, { recursive: true });
      const bytes = "original signed archive";
      const manifest = { ...previous, size: Buffer.byteLength(bytes), sha256: createHash("sha256").update(bytes).digest("hex") };
      await writeFile(archive, bytes);
      await writeFile(join(output, "latest.json"), JSON.stringify(manifest));
      expect(await reusablePackage(directory, release, previous.fingerprint, "arm64")).toEqual(manifest);
      expect(await reusablePackage(directory, { ...release, version: "0.3.0", build: 3 }, previous.fingerprint, "arm64")).toBeUndefined();
      await expect(reusablePackage(directory, release, "c".repeat(64), "arm64")).rejects.toThrow("同时提升");
      await expect(reusablePackage(directory, { ...release, notes: ["改动说明"] }, previous.fingerprint, "arm64")).rejects.toThrow("同时提升");
      await writeFile(archive, "modified signed archive");
      await expect(reusablePackage(directory, release, previous.fingerprint, "arm64")).rejects.toThrow("SHA-256");
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
});
