import { expect, test } from "bun:test";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { run } from "../鲁班零号大模型后台管理skill/scripts/ad";

test("management helper privately authenticates one update then reads only public state", async () => {
  const root = await mkdtemp(join(tmpdir(), "luban-ad-skill-")), original = globalThis.fetch;
  const keyFile = join(root, "credentials.json"), content = join(root, "ad.json"), image = join(root, "image.png");
  const config = JSON.parse(await readFile(new URL("../鲁班零号大模型后台管理skill/广告.json", import.meta.url), "utf8"));
  const calls: { url: string; options: RequestInit }[] = [];
  try {
    await writeFile(keyFile, JSON.stringify({ origin: "https://ai.maihh.net", token: "a".repeat(64) }), { mode: 0o600 });
    await writeFile(content, JSON.stringify(config));
    await writeFile(image, await readFile(new URL("../frontend/assets/brand/luban-intro-v1.png", import.meta.url)));
    globalThis.fetch = (async (url: any, options: any) => { calls.push({ url: String(url), options }); return Response.json(config); }) as any;
    expect(await run(["apply", content, image], keyFile)).toEqual(config);
    expect(calls.map(x => [x.url, x.options.method])).toEqual([
      ["https://ai.maihh.net/api/admin/app-ad", "PUT"], ["https://ai.maihh.net/api/app-ad", "GET"],
    ]);
    expect(new Headers(calls[0].options.headers).get("authorization")).toBe("Bearer " + "a".repeat(64));
    expect(new Headers(calls[1].options.headers).has("authorization")).toBe(false);
    expect(JSON.parse(calls[0].options.body as string).png_base64).toBeTruthy();
    expect(calls.every(x => x.options.redirect === "error")).toBe(true);
    await chmod(keyFile, 0o644);
    await expect(run(["show"], keyFile)).rejects.toThrow("权限600");
    expect(calls.length).toBe(2);
    await chmod(keyFile, 0o600);
    globalThis.fetch = (async () => new Response("private message", { status: 401 })) as any;
    await expect(run(["show"], keyFile)).rejects.toThrow("尚未配对");
  } finally { globalThis.fetch = original; await rm(root, { recursive: true, force: true }); }
});

test("stats command uses private authenticated read only and never reports identifiers", async () => {
  const root = await mkdtemp(join(tmpdir(), "luban-stats-skill-")), original = globalThis.fetch;
  const keyFile = join(root, "credentials.json"), calls: { url: string; options: RequestInit }[] = [];
  const stats = { schema: 1, timezone: "Asia/Shanghai", totals: { starts: 15, unique_installations: 4 } };
  try {
    await writeFile(keyFile, JSON.stringify({ origin: "https://ai.maihh.net", token: "a".repeat(64) }), { mode: 0o600 });
    globalThis.fetch = (async (url: any, options: any) => { calls.push({ url: String(url), options }); return Response.json(stats); }) as any;
    expect(await run(["stats"], keyFile)).toEqual(stats);
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://ai.maihh.net/api/admin/app-starts");
    expect(calls[0].options.method).toBe("GET");
    expect(calls[0].options.body).toBeUndefined();
    expect(new Headers(calls[0].options.headers).get("authorization")).toBe("Bearer " + "a".repeat(64));
    await expect(run(["stats", "extra"], keyFile)).rejects.toThrow("用法");
    expect(calls).toHaveLength(1);
  } finally { globalThis.fetch = original; await rm(root, { recursive: true, force: true }); }
});
