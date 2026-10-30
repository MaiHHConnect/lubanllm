import { describe, expect, test } from "bun:test";
import { cp, mkdir, mkdtemp, rm, symlink, writeFile, open } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { assertReleasePayload, binaryFiles, licenseFiles, runtimeFiles, evolutionPrefix, evolutionCapsulePrefix, evolutionPayload } from "../scripts/release-payload";
import { macho, payloadFixture } from "./release-payload.fixture";

const project = resolve(import.meta.dir, "..");
async function temporary(run: (directory: string, app: string) => Promise<void>) {
  const directory = await mkdtemp(join(tmpdir(), "luban-payload-gate-"));
  try { await run(directory, join(directory, "Luban.app")); }
  finally { await rm(directory, { recursive: true, force: true }); }
}
async function zip(path: string, entries: [string, Buffer, number?][]) {
  const script = String.raw`import base64,json,sys,zipfile
with zipfile.ZipFile(sys.argv[1],'w',zipfile.ZIP_DEFLATED) as z:
 for name,content,mode in json.load(sys.stdin):
  info=zipfile.ZipInfo(name); info.create_system=3
  info.external_attr=(mode or 0o100644)<<16
  z.writestr(info,base64.b64decode(content))`;
  const child = Bun.spawn([join(project, ".venv/bin/python"), "-c", script, path],
    { stdin: "pipe", stdout: "ignore", stderr: "pipe" });
  child.stdin.write(JSON.stringify(entries.map(([name, bytes, mode]) => [name, bytes.toString("base64"), mode ?? null])));
  child.stdin.end();
  const error = await new Response(child.stderr).text();
  if (await child.exited !== 0) throw new Error(error);
}

describe("发行载荷最小白名单", () => {
  test("独立安装应用不能缺少内置Python运行环境", async () => {
    await temporary(async (_, app) => {
      await payloadFixture(app);
      await writeFile(join(app, "Contents/Resources/launcher.json"), JSON.stringify({ mode: "bundled" }));
      await expect(assertReleasePayload(app)).rejects.toThrow("独立应用缺少便携Python运行环境");
    });
  });
  test("MLX 原生程序必须携带匹配GPU资源，研究脚本不得入包", async () => {
    await temporary(async (_, app) => {
      const files = await payloadFixture(app);
      const metal = "Contents/Resources/backend/native/mlx.metallib";
      await writeFile(join(app, metal), Buffer.from("changed GPU library"));
      await expect(assertReleasePayload(app)).rejects.toThrow("MLX 原生程序");
      await writeFile(join(app, metal), files[metal]);
      const receiptPath = "Contents/Resources/backend/native/luban-mlx-worker.json";
      const receipt = JSON.parse(files[receiptPath].toString());
      await writeFile(join(app, receiptPath), JSON.stringify({...receipt, binary_sha256: "0".repeat(64)}));
      await expect(assertReleasePayload(app)).rejects.toThrow("MLX 原生程序");
      await writeFile(join(app, receiptPath), files[receiptPath]);
      await writeFile(join(app, "Contents/Resources/backend/native/community_mlx_study.py"), "research only");
      await expect(assertReleasePayload(app)).rejects.toThrow("白名单");
    });
  });
  test("原生进化仅允许固定37项胶囊与worker，最终绑定和观察程序不可替换", async () => {
    await temporary(async (_, app) => {
      const files = await payloadFixture(app);
      expect(evolutionPayload).toHaveLength(37);
      for (const name of [evolutionPrefix + "runtime-authority.json", evolutionPrefix + "build.json",
        evolutionCapsulePrefix + "native/EvolutionObserve.swift", evolutionCapsulePrefix + "native/bin/persistent-evolution.dSYM"]) {
        await writeFile(join(app, name), "private build artifact");
        await expect(assertReleasePayload(app)).rejects.toThrow("白名单");
        await rm(join(app, name));
      }
      const binary = evolutionCapsulePrefix + "native/bin/cached-none";
      await writeFile(join(app, binary), Buffer.concat([files[binary], Buffer.from("changed")]));
      await expect(assertReleasePayload(app)).rejects.toThrow("最终文件绑定");
      await writeFile(join(app, binary), files[binary]);
      const manifestPath = evolutionCapsulePrefix + "native/runtime-release.json";
      const manifest = JSON.parse(files[manifestPath].toString());
      manifest.programs.asrDecoder.proof.stdout = "research log";
      await writeFile(join(app, manifestPath), JSON.stringify(manifest));
      await expect(assertReleasePayload(app)).rejects.toThrow("研究日志或源码变换");
      await writeFile(join(app, manifestPath), files[manifestPath]);
      const receiptPath = "Contents/Resources/backend/native/luban-evolution-observe.build.json";
      const receipt = JSON.parse(files[receiptPath].toString());
      for (const change of [{layers: 23}, {precision: "bf16"}, {python_invocations: 1}, {binary_sha256: "0".repeat(64)}, {source_bundle_sha256: "0".repeat(64)}]) {
        await writeFile(join(app, receiptPath), JSON.stringify({...receipt, ...change}));
        await expect(assertReleasePayload(app)).rejects.toThrow("Q4观察程序");
      }
      await writeFile(join(app, receiptPath), files[receiptPath]);
      await writeFile(join(app, evolutionPrefix, "worker.js"), "export const changed = true;");
      await expect(assertReleasePayload(app)).rejects.toThrow("worker与实际构建");
    });
  });
  test("仅白名单运行文件即可导入真实服务，不借用项目源码", async () => {
    await temporary(async (directory) => {
      expect(runtimeFiles).toContain("tool_protocol.py");
      const source = join(directory, "src"), module = join(source, "luban");
      await mkdir(module, { recursive: true });
      for (const file of runtimeFiles.filter(name => !name.startsWith("web/")))
        await cp(join(project, "src/luban", file), join(module, file));
      const child = Bun.spawn([join(project, ".venv/bin/python"), "-c", "import luban.server; print(luban.server.__file__)"], {
        cwd: directory, env: { ...process.env, PYTHONPATH: source, PYTHONDONTWRITEBYTECODE: "1" },
        stdout: "pipe", stderr: "pipe",
      });
      const [output, error, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
      expect(code, error).toBe(0);
      expect(output.trim()).toBe(join(module, "server.py"));
    });
  }, 30000);
  test("图音操作运行模块与图片原生库必须入包，篡改构建回执拒绝", async () => {
    await temporary(async (_, app) => {
      const files = await payloadFixture(app);
      const videoHelper = "Contents/Resources/backend/native/luban-video-frames";
      expect(files[videoHelper]).toBeDefined();
      await rm(join(app, videoHelper));
      await expect(assertReleasePayload(app)).rejects.toThrow("缺少必要运行文件");
      await writeFile(join(app, videoHelper), files[videoHelper]);
      for (const name of ["media_learning", "image_learning", "native_image_training", "operation_learning", "native_operation_training", "native_video_training", "speech_learning", "video_learning"]) {
        const path = "Contents/Resources/backend/src/luban/" + name + ".py";
        expect(files[path]).toBeDefined();
      }
      const path = "Contents/Resources/backend/native/libluban_image_training.dylib.build.json";
      const receipt = JSON.parse(files[path].toString());
      for (const changed of [{binary_sha256: "0".repeat(64)}, {source_bundle_sha256: "0".repeat(64)},
        {sources: [".native-migration/research/Training.cpp"]}, {no_python_linkage: false}, {actual_compile: {exit_code: 1}}]) {
        await writeFile(join(app, path), JSON.stringify({...receipt, ...changed}));
        await expect(assertReleasePayload(app)).rejects.toThrow("图片训练库");
      }
      await writeFile(join(app, path), files[path]);
      await rm(join(app, "Contents/Resources/backend/native/libluban_image_training.dylib"));
      await expect(assertReleasePayload(app)).rejects.toThrow("缺少必要运行文件");
    });
  });
  test("包内组件版本和前端字节必须与实际构建一致", async () => {
    await temporary(async (_, app) => {
      const files = await payloadFixture(app);
      const metadata = "Contents/Resources/backend/src/luban/web/component-versions.json";
      const manifest = JSON.parse(files[metadata].toString()); manifest.components[0].version = "99.0.0";
      await writeFile(join(app, metadata), JSON.stringify(manifest));
      await expect(assertReleasePayload(app)).rejects.toThrow("组件版本");
      await writeFile(join(app, metadata), files[metadata]);
      await writeFile(join(app, "Contents/Resources/backend/src/luban/web/app.js"), "stale bundle");
      await expect(assertReleasePayload(app)).rejects.toThrow("产物");
    });
  });
  test("保留运行文件、真实许可证及空__init__，目录和ZIP均通过", async () => {
    await temporary(async (directory, app) => {
      const files = await payloadFixture(app);
      expect((await assertReleasePayload(app)).binaries).toBe(binaryFiles.length);
      const archive = join(directory, "release.zip");
      await zip(archive, Object.entries(files).map(([name, bytes]) => ["Luban.app/" + name, bytes]));
      expect((await assertReleasePayload(archive)).files).toBe(Object.keys(files).length);
    });
  });
  test("研究脚本、调试产物、凭据和未知源码不能混进目录", async () => {
    await temporary(async (_, app) => {
      await payloadFixture(app);
      for (const name of ["Contents/Resources/鲁班零号大模型后台管理skill/SKILL.md", "Contents/Resources/鲁班零号大模型后台管理skill/credentials.json",
        "Contents/Resources/research/result.json", "Contents/Resources/scripts/train.py",
        "Contents/Resources/weights.safetensors", "Contents/Resources/backend/src/luban/probe.py",
        "Contents/Resources/web/app.js.map", "Contents/MacOS/Luban.dSYM/Contents/Resources/DWARF/Luban",
        "Contents/Resources/.env", "Contents/Resources/credentials.json", "Contents/Resources/native/Training.cpp"]) {
        const path = join(app, name);
        const parent = path.substring(0, path.lastIndexOf("/"));
        await mkdir(parent, { recursive: true }); await writeFile(path, "private fixture");
        await expect(assertReleasePayload(app)).rejects.toThrow("白名单");
        // Remove only this fixture's unknown top-level directory/file.
        const relative = name.startsWith("Contents/Resources/backend/src/luban/") ? name :
          name.startsWith("Contents/MacOS/") ? "Contents/MacOS/Luban.dSYM" : name.split("/").slice(0, 3).join("/");
        await rm(join(app, relative), { recursive: true, force: true });
      }
    });
  });
  test("缺失、空或被替换的许可证与缺失运行文件拒绝", async () => {
    await temporary(async (_, app) => {
      const files = await payloadFixture(app);
      const license = Object.values(licenseFiles)[0];
      for (const bytes of [Buffer.alloc(0), Buffer.from("replacement")]) {
        await writeFile(join(app, license), bytes);
        await expect(assertReleasePayload(app)).rejects.toThrow("许可证");
      }
      await rm(join(app, license));
      await expect(assertReleasePayload(app)).rejects.toThrow("缺少");
      await writeFile(join(app, license), files[license]);
      const icon = "Contents/Resources/LubanIcon.icns";
      await rm(join(app, icon));
      await expect(assertReleasePayload(app)).rejects.toThrow("LubanIcon.icns");
      await writeFile(join(app, icon), files[icon]);
      await rm(join(app, "Contents/Resources/backend/src/luban/server.py"));
      await expect(assertReleasePayload(app)).rejects.toThrow("server.py");
    });
  });
  test("目录内与ZIP内符号链接都拒绝", async () => {
    await temporary(async (directory, app) => {
      const files = await payloadFixture(app), name = "Contents/Resources/backend/src/luban/server.py";
      const source = join(directory, "external.py");
      await writeFile(source, "external source"); await rm(join(app, name));
      await symlink(source, join(app, name));
      await expect(assertReleasePayload(app)).rejects.toThrow("链接");
      const archive = join(directory, "symlink.zip");
      await zip(archive, Object.entries(files).map(([path, bytes]) =>
        ["Luban.app/" + path, path === name ? Buffer.from("external.py") : bytes, path === name ? 0o120777 : 0o100644]));
      await expect(assertReleasePayload(archive)).rejects.toThrow("链接");
    });
  });
  test("ZIP研究产物、路径逃逸、重复成员和额外根目录拒绝", async () => {
    await temporary(async (directory, app) => {
      const files = await payloadFixture(app);
      const entries: [string, Buffer][] = Object.entries(files).map(([name, bytes]) => ["Luban.app/" + name, bytes]);
      const archive = join(directory, "poisoned.zip");
      for (const name of ["Luban.app/Contents/Resources/scripts/research.py", "Luban.app/Contents/Resources/../probe.py",
        "Luban.app/Contents/Resources\\probe.py", "Other.app/Contents/Info.plist", entries[0][0]]) {
        await zip(archive, [...entries, [name, Buffer.from("unexpected")]]);
        await expect(assertReleasePayload(archive)).rejects.toThrow("门禁失败");
      }
    });
  });
  test("Mach-O中的STABS与DWARF，以及多架构隐藏的调试符号拒绝", async () => {
    await temporary(async (_, app) => {
      await payloadFixture(app);
      const path = join(app, binaryFiles[0]);
      const command = Buffer.alloc(24); command.writeUInt32LE(2); command.writeUInt32LE(24, 4);
      command.writeUInt32LE(56, 8); command.writeUInt32LE(1, 12);
      command.writeUInt32LE(72, 16); command.writeUInt32LE(1, 20);
      const symbol = Buffer.alloc(16); symbol[4] = 0x64;
      const debug = Buffer.concat([macho(command, 1), symbol, Buffer.alloc(1)]);
      await writeFile(path, debug);
      await expect(assertReleasePayload(app)).rejects.toThrow("STABS");
      const fat = Buffer.alloc(48); fat.writeUInt32BE(0xcafebabe); fat.writeUInt32BE(2, 4);
      fat.writeUInt32BE(48, 16); fat.writeUInt32BE(32, 20); fat.writeUInt32BE(80, 36); fat.writeUInt32BE(debug.length, 40);
      await writeFile(path, Buffer.concat([fat, macho(), debug]));
      await expect(assertReleasePayload(app)).rejects.toThrow("STABS");
      const segment = Buffer.alloc(72); segment.writeUInt32LE(25); segment.writeUInt32LE(72, 4);
      segment.write("__DWARF", 8);
      await writeFile(path, macho(segment, 1));
      await expect(assertReleasePayload(app)).rejects.toThrow("DWARF");
      segment.fill(0, 8, 24); segment.write("__TEXT", 8);
      const section = Buffer.alloc(80); section.write("__debug_info");
      segment.writeUInt32LE(152, 4); segment.writeUInt32LE(1, 64);
      await writeFile(path, macho(Buffer.concat([segment, section]), 1));
      await expect(assertReleasePayload(app)).rejects.toThrow("调试节");
    });
  });
  test("保留普通动态符号，格式截断或非Mach-O程序失败关闭", async () => {
    await temporary(async (_, app) => {
      await payloadFixture(app);
      const path = join(app, binaryFiles[0]);
      const command = Buffer.alloc(24); command.writeUInt32LE(2); command.writeUInt32LE(24, 4);
      command.writeUInt32LE(56, 8); command.writeUInt32LE(1, 12);
      const symbol = Buffer.alloc(16); symbol[4] = 0x0f;
      await writeFile(path, Buffer.concat([macho(command, 1), symbol]));
      await assertReleasePayload(app);
      for (const bytes of [Buffer.from("not binary"), Buffer.from([0xcf, 0xfa, 0xed, 0xfe]), macho(command, 1)]) {
        await writeFile(path, bytes);
        await expect(assertReleasePayload(app)).rejects.toThrow("Mach-O");
      }
    });
  });
  test("只允许Apple strip的精确兼容占位符，不能用N_OPT隐藏调试内容", async () => {
    await temporary(async (_, app) => {
      await payloadFixture(app);
      const path = join(app, binaryFiles[0]);
      const command = Buffer.alloc(24); command.writeUInt32LE(2); command.writeUInt32LE(24, 4);
      command.writeUInt32LE(56, 8); command.writeUInt32LE(1, 12);
      command.writeUInt32LE(72, 16); command.writeUInt32LE(16, 20);
      const symbol = Buffer.alloc(16); symbol.writeUInt32LE(1); symbol[4] = 0x3c;
      symbol.writeBigUInt64LE(0x05614542n, 8);
      const strings = Buffer.from("\0radr://5614542\0");
      await writeFile(path, Buffer.concat([macho(command, 1), symbol, strings]));
      await assertReleasePayload(app);
      symbol.writeBigUInt64LE(0n, 8);
      await writeFile(path, Buffer.concat([macho(command, 1), symbol, strings]));
      await expect(assertReleasePayload(app)).rejects.toThrow("STABS");
      symbol.writeBigUInt64LE(0x05614542n, 8);
      strings[1] = 0x78;
      await writeFile(path, Buffer.concat([macho(command, 1), symbol, strings]));
      await expect(assertReleasePayload(app)).rejects.toThrow("STABS");
    });
  });
  test("生产ditto生成的实际ZIP执行相同规则", async () => {
    await temporary(async (directory, app) => {
      await payloadFixture(app);
      const archive = join(directory, "ditto.zip");
      const child = Bun.spawn(["/usr/bin/ditto", "-c", "-k", "--norsrc", "--noextattr", "--keepParent", app, archive],
        { stdout: "ignore", stderr: "pipe" });
      const error = await new Response(child.stderr).text();
      if (await child.exited !== 0) throw new Error(error);
      await assertReleasePayload(archive);
    });
  });
});


test("ordinary oversized packages stay rejected and full-name packages cannot omit models", async () => {
  await temporary(async(directory,app)=>{
    const sparse=join(directory,"unknown.zip"),handle=await open(sparse,"w");
    await handle.truncate(512*1024**2+1);await handle.close();
    await expect(assertReleasePayload(sparse)).rejects.toThrow("ZIP类型/大小无效");
    await payloadFixture(app);
    const archive=join(directory,"Luban-0.3.218-macos-arm64-full.zip");
    const p=Bun.spawn(["/usr/bin/ditto","-c","-k","--norsrc","--noextattr","--keepParent",app,archive]);
    expect(await p.exited).toBe(0);
    await expect(assertReleasePayload(archive)).rejects.toThrow("完整安装包必须包含全部模型");
  });
},30000);
