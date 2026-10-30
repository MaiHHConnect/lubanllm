import {afterAll, beforeAll, expect, test} from "bun:test";
import {chmod, lstat, mkdir, mkdtemp, readFile, realpath, rm, symlink, truncate, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import {binaryFiles} from "../scripts/release-payload";

let root: string, decoder: string, fixture: string;
async function run(argv: string[]) {
  const child = Bun.spawn(argv, {stdout: "pipe", stderr: "pipe"});
  const [status, stdout, stderr] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()]);
  return {status, stdout, stderr};
}
beforeAll(async () => {
  root = await mkdtemp(join(await realpath(tmpdir()), "luban-video-test-"));
  decoder = join(root, "decoder"); fixture = join(root, "fixture");
  const sdk = (await run(["xcrun", "--show-sdk-path"])).stdout.trim();
  for (const [source, output] of [["native/VideoFrames.swift", decoder], ["tests/fixtures/video_frames.swift", fixture]]) {
    const build = await run(["/Library/Developer/CommandLineTools/usr/bin/swiftc", resolve(source!), "-sdk", sdk,
      "-O", "-framework", "AVFoundation", "-framework", "ImageIO", "-framework", "UniformTypeIdentifiers", "-o", output!]);
    expect(build.status, build.stderr).toBe(0);
  }
  expect((await run([fixture, join(root, "short.mov"), "3"])).status).toBe(0);
  expect((await run([fixture, join(root, "long.mov"), "31"])).status).toBe(0);
}, 60_000);
afterAll(async () => { if (root) await rm(root, {recursive: true, force: true}); });
async function output(name: string) { const path = join(root, name); await mkdir(path, {mode: 0o700}); return path; }
test("real AVFoundation video yields at most four ordered JPEG frames with actual timestamps", async () => {
  const directory = await output("frames");
  const child = await run([decoder, join(root, "short.mov"), directory]);
  expect(child.status, child.stdout + child.stderr).toBe(0);
  const result = JSON.parse(child.stdout);
  expect(result.ok).toBe(true); expect(result.audio_processed).toBe(false);
  expect(result.duration_seconds).toBe(3); expect(result.frames.length).toBe(4);
  let previous = -1;
  for (const frame of result.frames) {
    expect(frame.path.startsWith(directory + "/")).toBe(true);
    expect(frame.timestamp_seconds).toBeGreaterThan(previous);
    expect(frame.timestamp_seconds).toBeLessThanOrEqual(3);
    previous = frame.timestamp_seconds;
    const file = await lstat(frame.path), bytes = await readFile(frame.path);
    expect(file.mode & 0o777).toBe(0o600); expect(file.isFile()).toBe(true);
    expect(bytes.subarray(0, 2)).toEqual(Buffer.from([255, 216]));
  }
});
test("rejects long/invalid videos, insecure directories and linked inputs", async () => {
  const directory = await output("reject");
  const long = await run([decoder, join(root, "long.mov"), directory]);
  expect(long.status).toBe(1); expect(JSON.parse(long.stdout).error).toContain("30秒");
  const invalid = join(root, "invalid.mov"); await writeFile(invalid, "not a video");
  expect((await run([decoder, invalid, directory])).status).toBe(1);
  await chmod(directory, 0o755);
  expect((await run([decoder, join(root, "short.mov"), directory])).status).toBe(1);
  await chmod(directory, 0o700);
  const linked = join(root, "linked.mov"); await symlink(join(root, "short.mov"), linked);
  expect((await run([decoder, linked, directory])).status).toBe(1);
  expect((await run([decoder, "relative.mov", directory])).status).toBe(1);
  const linkedDirectory = join(root, "linked-directory"); await symlink(directory, linkedDirectory);
  expect((await run([decoder, join(root, "short.mov"), linkedDirectory])).status).toBe(1);
  const oversized = join(root, "oversized.mov"); await writeFile(oversized, "");
  await truncate(oversized, 64 * 1024 * 1024 + 1);
  expect((await run([decoder, oversized, directory])).status).toBe(1);
});
test("does not overwrite existing output and is included in the explicit release payload", async () => {
  const directory = await output("existing");
  await writeFile(join(directory, "frame-000.jpg"), "preserved", {mode: 0o600});
  expect((await run([decoder, join(root, "short.mov"), directory])).status).toBe(1);
  expect(await readFile(join(directory, "frame-000.jpg"), "utf8")).toBe("preserved");
  const later = await output("later-collision");
  await writeFile(join(later, "frame-001.jpg"), "preserved later", {mode: 0o600});
  expect((await run([decoder, join(root, "short.mov"), later])).status).toBe(1);
  expect(await readFile(join(later, "frame-001.jpg"), "utf8")).toBe("preserved later");
  await expect(lstat(join(later, "frame-000.jpg"))).rejects.toThrow();
  expect(binaryFiles).toContain("Contents/Resources/backend/native/luban-video-frames");
});
