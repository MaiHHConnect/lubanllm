import { test, expect } from "bun:test";
import { copyFile, mkdir, mkdtemp, rm, writeFile, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

test("官网仅提供指定的更新文件，清单不缓存，HEAD 返回下载大小", async () => {
  const root = await mkdtemp(join(tmpdir(), "luban-update-site-"));
  const oldPort = process.env.PORT, oldDatabase = process.env.LEADERBOARD_DB;
  let server: ReturnType<typeof Bun.serve> | undefined;
  try {
    await copyFile(new URL("../website/server.ts", import.meta.url), join(root, "server.ts"));
    await copyFile(new URL("../website/countries.ts", import.meta.url), join(root, "countries.ts"));
    for (const name of ["name-filter.ts","name_words.json","evaluation.ts","enterprise.ts","app-ad.ts","app-starts.ts","growth.ts","profile.ts","i18n.ts","language.ts","downloads.json"]) await copyFile(new URL(`../website/${name}`, import.meta.url), join(root,name));
    await writeFile(join(root,"downloads.json"), JSON.stringify({version:"0.2.0",direct:"https://ai.maihh.net/downloads/Luban-0.2.0-macos-arm64-full.zip",quark:"https://pan.quark.cn/s/fixture123",github:"https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.2.0-beta.1"}));
    await copyFile(new URL("../benchmarks/ranking-v1.json", import.meta.url), join(root, "ranking-v1.json"));
    await mkdir(join(root, "public/updates"), { recursive: true });
    await mkdir(join(root, "public/app-ad"), { recursive: true });
    await writeFile(join(root, "public/app-ad/luban-intro-v1.png"), "png-fixture");
    await writeFile(join(root, "public/updates/latest.json"), '{"version":"fixture"}');
    await writeFile(join(root, "public/updates/Luban-0.2.0-macos-arm64.zip"), "archive-fixture");
    await writeFile(join(root, "public/updates/Luban-0.3.0-from-0.2.0-macos-arm64-delta.zip"), "delta-fixture");
    await writeFile(join(root, "public/secret.txt"), "private");
    process.env.PORT = "0"; process.env.LEADERBOARD_DB = join(root, "fixture.sqlite");
    server = (await import(join(root, "server.ts"))).server;
    const origin = `http://127.0.0.1:${server!.port}`;
    for(const language of ["zh","en"]) {
      const home=await (await fetch(`${origin}/?lang=${language}`)).text();
      expect(home).toContain('href="https://ai.maihh.net/downloads/Luban-0.2.0-macos-arm64-full.zip"');
      expect(home).toContain('href="https://pan.quark.cn/s/fixture123"');
      expect(home).toContain('href="https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.2.0-beta.1"');
      expect(home).not.toContain('aria-disabled="true"');
      expect(home).toContain(language === "zh" ? "GitHub 小包（模型另下）" : "GitHub small app (models separate)");
    }
    const ad = await fetch(`${origin}/api/app-ad`);
    expect(ad.headers.get("access-control-allow-origin")).toBe("*");
    expect((await ad.json()).image).toBe("/app-ad/luban-intro-v1.png");
    const adImage = await fetch(`${origin}/app-ad/luban-intro-v1.png`, { method: "HEAD" });
    expect(adImage.status).toBe(200);
    expect(adImage.headers.get("content-type")).toBe("image/png");
    expect(adImage.headers.get("access-control-allow-origin")).toBe("*");
    expect(await adImage.text()).toBe("");
    expect((await fetch(`${origin}/api/admin/app-ad`)).status).toBe(401);
    const credential = (await readFile(join(root, "app-ad-admin-key"), "utf8")).trim();
    expect(credential).toMatch(/^[0-9a-f]{64}$/);
    expect((await stat(join(root, "app-ad-admin-key"))).mode & 0o777).toBe(0o600);
    expect((await fetch(`${origin}/api/admin/app-ad`, { headers: { authorization: `Bearer ${credential}` } })).status).toBe(200);
    expect((await fetch(`${origin}/api/admin/app-starts`)).status).toBe(401);
    const startEvent = { schema: 1, install_id: crypto.randomUUID(), event_id: crypto.randomUUID() };
    const postStart = () => fetch(`${origin}/api/app-start`, { method: "POST", headers: { origin, "content-type": "application/json" }, body: JSON.stringify(startEvent) });
    expect(await (await postStart()).json()).toEqual({ schema: 1, ok: true, duplicate: false });
    expect(await (await postStart()).json()).toEqual({ schema: 1, ok: true, duplicate: true });
    const starts = await fetch(`${origin}/api/admin/app-starts`, { headers: { authorization: `Bearer ${credential}` } });
    expect(starts.status).toBe(200);
    expect((await starts.json()).totals).toEqual({ starts: 1, unique_installations: 1 });
    expect((await fetch(`${origin}/app-ad/../secret.txt`)).status).toBe(404);
    const manifest = await fetch(`${origin}/updates/latest.json`);
    expect(manifest.status).toBe(200);
    expect(manifest.headers.get("cache-control")).toBe("no-store");
    expect(await manifest.json()).toEqual({ version: "fixture" });
    const archive = await fetch(`${origin}/updates/Luban-0.2.0-macos-arm64.zip`, { method: "HEAD" });
    expect(archive.status).toBe(200);
    expect(archive.headers.get("content-length")).toBe("15");
    expect(await archive.text()).toBe("");
    const delta = await fetch(`${origin}/updates/Luban-0.3.0-from-0.2.0-macos-arm64-delta.zip`);
    expect(delta.status).toBe(200);
    expect(delta.headers.get("cache-control")).toContain("immutable");
    expect(await delta.text()).toBe("delta-fixture");
    const deltaHead = await fetch(`${origin}/updates/Luban-0.3.0-from-0.2.0-macos-arm64-delta.zip`, { method: "HEAD" });
    expect(deltaHead.headers.get("content-length")).toBe("13");
    expect(await deltaHead.text()).toBe("");
    for (const path of ["/updates/missing.zip", "/updates/secret.txt", "/updates/../secret.txt", "/updates/Luban-0.2.1-macos-arm64.zip",
      "/updates/Luban-0.3.0-macos-arm64-delta.zip", "/updates/Luban-0.3.0-from-0.2.0-macos-arm64.zip"])
      expect((await fetch(`${origin}${path}`)).status).toBe(404);
  } finally {
    server?.stop(true);
    if (oldPort === undefined) delete process.env.PORT; else process.env.PORT = oldPort;
    if (oldDatabase === undefined) delete process.env.LEADERBOARD_DB; else process.env.LEADERBOARD_DB = oldDatabase;
    await rm(root, { recursive: true, force: true });
  }
});
