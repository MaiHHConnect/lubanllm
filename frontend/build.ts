import { mkdir, copyFile, cp, mkdtemp, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { tmpdir } from "node:os";
import { assertComponentVersions, commitVersions, frontendArtifacts, prepareVersions, publicVersions, versionBuildLock } from "../scripts/component-versions";

const root = resolve(import.meta.dir, "..");
const output = resolve(root, "src/luban/web");
const unlock = await versionBuildLock(root);
let staging: string | undefined;
try {
const plan = await prepareVersions(root);
staging = await mkdtemp(resolve(tmpdir(), "luban-frontend-"));
const result = await Bun.build({
  entrypoints: [resolve(import.meta.dir, "app.ts")],
  outdir: staging,
  target: "browser",
  minify: true,
  naming: "app.js",
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  throw new Error("前端编译失败；组件版本未更新");
}
await Promise.all(["index.html", "style.css"].map(file => copyFile(resolve(import.meta.dir, file), resolve(staging!, file))));
await cp(resolve(import.meta.dir, "assets"), resolve(staging, "assets"), { recursive: true });
await writeFile(resolve(staging, "component-versions.json"), JSON.stringify(publicVersions(plan.registry, await frontendArtifacts(staging)), null, 2) + "\n");
await mkdir(output, { recursive: true });
// Replace assets as a set so deleted source assets cannot survive a rebuild.
await rm(resolve(output, "assets"), { recursive: true, force: true });
await cp(staging, output, { recursive: true });
await commitVersions(root, plan);
await assertComponentVersions(root);
console.log(`组件版本已核验：${plan.changed.length ? plan.changed.join("、") : "无源码变化，版本保持"}`);
console.log("Luban frontend built → src/luban/web");
} finally {
  if (staging) await rm(staging, { recursive: true, force: true });
  await unlock();
}
