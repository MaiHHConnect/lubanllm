import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { assertComponentVersions, commitVersions, componentDefinitions, frontendArtifacts, nextVersions, prepareVersions, publicVersions, versionBuildLock } from "../scripts/component-versions";

const release = { version: "0.3.12", build: 20, notes: ["现有版本"] };
const digest = (text: string) => createHash("sha256").update(text).digest("hex");
const snapshot = () => Object.fromEntries(componentDefinitions.map(c => [c.id, digest(c.id)]));
const baseline = () => nextVersions(null, snapshot(), release).registry;
async function write(root: string, path: string, value: string) {
  await mkdir(dirname(join(root, path)), { recursive: true }); await writeFile(join(root, path), value);
}
async function fixture(run: (root: string, web: string) => Promise<void>) {
  const root = await mkdtemp(join(tmpdir(), "luban-component-versions-")), web = join(root, "src/luban/web");
  try {
    for (const pattern of new Set(componentDefinitions.flatMap(c => [...c.patterns]))) await write(root, pattern.replaceAll("**", "fixture.ts").replaceAll("*", "fixture"), "source");
    await write(root, "release.json", JSON.stringify(release));
    for (const name of ["app.js", "index.html", "style.css", "assets/icon.svg"]) await write(web, name, "built");
    await run(root, web);
  } finally { await rm(root, { recursive: true, force: true }); }
}
async function build(root: string, web: string) {
  const plan = await prepareVersions(root);
  await write(web, "component-versions.json", JSON.stringify(publicVersions(plan.registry, await frontendArtifacts(web))));
  await commitVersions(root, plan); return plan;
}
describe("组件版本构建门禁", () => {
  test("首次登记基线，重复构建不升号", () => {
    expect(nextVersions(baseline(), snapshot(), release)).toEqual({ registry: baseline(), release, changed: [] });
  });
  test("只升级变更组件，并自动提升客户端版本及构建号", () => {
    const sources = { ...snapshot(), training: digest("new training") };
    const next = nextVersions(baseline(), sources, release);
    expect(next.changed).toEqual(["training"]);
    expect(next.registry.components.training.version).toBe("1.0.1");
    expect(next.registry.components.generation.version).toBe("1.0.0");
    expect(next.release).toEqual({ ...release, version: "0.3.13", build: 21 });
    expect(nextVersions(next.registry, sources, next.release).changed).toEqual([]);
  });
  test("保留已显式提升的客户端版本，说明变更也升号", () => {
    expect(nextVersions(baseline(), snapshot(), { ...release, version: "0.4.0", build: 25 }).release.build).toBe(25);
    const result = nextVersions(baseline(), snapshot(), { ...release, notes: ["新说明"] });
    expect(result.changed).toEqual([]); expect(result.release.version).toBe("0.3.13");
  });
  test("登记损坏、缺组件和客户端回退均拒绝", () => {
    const prior = baseline(); delete prior.components.training;
    expect(() => nextVersions(prior, snapshot(), release)).toThrow("组件登记");
    expect(() => nextVersions(baseline(), { ...snapshot(), operators: "bad" }, release)).toThrow("指纹");
    expect(() => nextVersions(baseline(), snapshot(), { ...release, build: 19 })).toThrow("回退");
    expect(() => nextVersions(baseline(), snapshot(), { ...release, version: "0.3.11" })).toThrow("回退");
  });
  test("原七组件仅允许补图片组件，保留旧版本并拒绝其它缺项或未知项", () => {
    const prior = baseline(); delete prior.components.vision;
    const next = nextVersions(prior, snapshot(), release);
    expect(next.changed).toEqual(["vision"]);
    expect(next.registry.components.vision.version).toBe("1.0.0");
    for (const [id, value] of Object.entries(prior.components)) expect(next.registry.components[id]).toEqual(value);
    expect(next.release).toEqual({ ...release, version: "0.3.13", build: 21 });
    delete prior.components.training;
    expect(() => nextVersions(prior, snapshot(), release)).toThrow("组件登记");
    const unknown = baseline(); unknown.components.unknown = unknown.components.speech;
    expect(() => nextVersions(unknown, snapshot(), release)).toThrow("组件登记");
  });
  test("图片组件跟踪正式实现，研究候选不会冒充产品版本", async () => fixture(async (root, web) => {
    await build(root, web);
    await write(root, ".native-migration/research/native-image-learning-v1/Learner.cpp", "tiny CPU research");
    expect((await prepareVersions(root)).changed).toEqual([]);
    await write(root, "src/luban/vision.py", "changed image implementation");
    expect((await build(root, web)).changed).toEqual(["service", "vision"]);
    const manifest = publicVersions((await prepareVersions(root)).registry, {});
    expect(manifest.components.filter(c => ["vision", "speech", "training"].includes(c.id)).map(c => c.name))
      .toEqual(["训练引擎", "图片理解", "声音识别"]);
    expect(manifest.pending_components).toEqual([{ name: "深度学习", status: "操作学习实现随训练引擎版本；真实质量、采纳与执行待验收" }]);
    await assertComponentVersions(root, web);
  }));
  test("真实文件：训练算子变更同时更新训练和算子，忽略隔离候选", async () => fixture(async (root, web) => {
    await build(root, web); await assertComponentVersions(root, web);
    await write(root, ".native-migration/native/Training.cpp", "unadopted");
    expect((await prepareVersions(root)).changed).toEqual([]);
    await write(root, "native/Training.cpp", "changed");
    await expect(assertComponentVersions(root, web)).rejects.toThrow("源码已变化");
    const plan = await build(root, web);
    expect(plan.changed).toEqual(["operators", "training"]);
    expect(plan.registry.components.speech.version).toBe("1.0.0");
    await assertComponentVersions(root, web);
  }));
  test("图音操作产品源与原生算子归入真实实现组件，不计研究候选", async () => fixture(async (root, web) => {
    await build(root, web);
    await write(root, "src/luban/image_learning.py", "image product change");
    expect((await build(root, web)).changed).toEqual(["service", "training", "vision"]);
    await write(root, "src/luban/operation_learning.py", "operation product change");
    expect((await build(root, web)).changed).toEqual(["service", "training", "vision"]);
    await write(root, "src/luban/speech_learning.py", "ASR product change");
    expect((await build(root, web)).changed).toEqual(["service", "training", "speech"]);
    await write(root, "native/asr/Graph.h", "native ASR graph change");
    expect((await build(root, web)).changed).toEqual(["operators", "training", "speech"]);
    await write(root, "native/ImageTraining.cpp", "native image and action graph change");
    expect((await build(root, web)).changed).toEqual(["operators", "training", "vision"]);
    await write(root, ".native-migration/research/native-asr/Graph.h", "isolated CPU");
    expect((await prepareVersions(root)).changed).toEqual([]);
  }));
  test("原生进化权限头与构建overlay变化必须更新训练组件，研究证据不入版本", async () => fixture(async (root, web) => {
    await build(root, web);
    await write(root, "native/AcceptedSourcePermissions.h", "changed held-FD source permission policy");
    await expect(assertComponentVersions(root, web)).rejects.toThrow("源码已变化");
    expect((await build(root, web)).changed).toEqual(["training"]);
    await write(root, "scripts/native-evolution-source-permissions.ts", "changed runtime source overlay");
    await expect(assertComponentVersions(root, web)).rejects.toThrow("源码已变化");
    expect((await build(root, web)).changed).toEqual(["training"]);
    await write(root, ".native-migration/research/source-permission/result.json", "CPU evidence");
    expect((await prepareVersions(root)).changed).toEqual([]);
  }));
  test("图片和声音二进制/收据缺一或错误配对拒绝", async () => fixture(async (root) => {
    for (const [name, suffix] of [["libluban_image_training.dylib", ".build.json"], ["luban-asr-training", ".build.json"]]) {
      const binary = "build/native/" + name;
      await write(root, binary, "binary");
      await expect(prepareVersions(root)).rejects.toThrow("同时存在");
      await write(root, binary + suffix, JSON.stringify({binary_sha256: digest("wrong")}));
      await expect(prepareVersions(root)).rejects.toThrow("SHA不匹配");
      await write(root, binary + suffix, JSON.stringify({binary_sha256: digest("binary")}));
      await prepareVersions(root);
    }
  }));
  test("增加及删除源码都计入对应指纹", async () => fixture(async (root, web) => {
    await build(root, web); await write(root, "frontend/extra.ts", "extra");
    expect((await build(root, web)).changed).toEqual(["frontend"]);
    await rm(join(root, "frontend/extra.ts"));
    expect((await build(root, web)).registry.components.frontend.version).toBe("1.0.2");
  }));
  test("产物或版本清单篡改不能通过，包内独立副本同样检查", async () => fixture(async (root, web) => {
    await build(root, web);
    await write(web, "app.js", "tampered");
    await expect(assertComponentVersions(root, web)).rejects.toThrow("产物");
    await write(web, "app.js", "built");
    await write(web, "component-versions.json", "{}");
    await expect(assertComponentVersions(root, web)).rejects.toThrow("产物");
  }));
  test("构建中源码或元数据变化拒绝提交，旧登记保留", async () => fixture(async (root, web) => {
    await build(root, web);
    const saved = await readFile(join(root, "component-versions.json"), "utf8"), plan = await prepareVersions(root);
    await write(root, "src/luban/speech.py", "changed while compiling");
    await expect(commitVersions(root, plan)).rejects.toThrow("构建中源码");
    expect(await readFile(join(root, "component-versions.json"), "utf8")).toBe(saved);
    const newPlan = await prepareVersions(root);
    await write(root, "release.json", JSON.stringify({ ...release, build: 25 }));
    await expect(commitVersions(root, newPlan)).rejects.toThrow("版本记录");
  }));
  test("二进制与回执必须成对，构建锁防止并发覆盖", async () => fixture(async (root) => {
    const unlock = await versionBuildLock(root);
    await expect(versionBuildLock(root)).rejects.toThrow("构建锁"); await unlock();
    await (await versionBuildLock(root))();
    await write(root, "build/native/luban-worker", "binary");
    await expect(prepareVersions(root)).rejects.toThrow("同时存在");
    await write(root, "build/native/luban-worker.json", JSON.stringify({ binary_sha256: digest("wrong") }));
    await expect(prepareVersions(root)).rejects.toThrow("SHA不匹配");
    await write(root, "build/native/luban-worker.json", JSON.stringify({ binary_sha256: digest("binary") }));
    expect((await prepareVersions(root)).registry.components.generation.version).toBe("1.0.0");
  }));
});
