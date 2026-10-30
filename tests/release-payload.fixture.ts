import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { binaryFiles, licenseFiles, requiredPayload, evolutionPrefix, evolutionCapsulePrefix, evolutionBinaryNames, evolutionLicenseNames, evolutionRoleNames, mlxSourceFiles } from "../scripts/release-payload";

export function macho(commands = Buffer.alloc(0), count = 0) {
  const header = Buffer.alloc(32);
  header.writeUInt32LE(0xfeedfacf, 0);
  header.writeUInt32LE(0x100000c, 4);
  header.writeUInt32LE(2, 12);
  header.writeUInt32LE(count, 16);
  header.writeUInt32LE(commands.length, 20);
  return Buffer.concat([header, commands]);
}

// Synthetic Mach-O files test structural checks; they are not runnable applications.
export async function payloadFixture(app: string, project = resolve(import.meta.dir, "..")) {
  const files: Record<string, Buffer> = {};
  for (const name of requiredPayload) files[name] = Buffer.from("runtime fixture");
  files["Contents/Resources/launcher.json"] = Buffer.from(JSON.stringify({ project: "/legacy-development" }));
  files["Contents/Resources/backend/src/luban/__init__.py"] = Buffer.alloc(0);
  for (const name of binaryFiles) files[name] = macho();
  const source = Buffer.concat([Buffer.from("native/ImageTraining.cpp\0"), await readFile(join(project, "native/ImageTraining.cpp")), Buffer.from("\0")]);
  files["Contents/Resources/backend/native/libluban_image_training.dylib.build.json"] = Buffer.from(JSON.stringify({
    schema: "luban-native-image-training-build-v1", production_shape: true, no_python_linkage: true,
    sources: ["native/ImageTraining.cpp"], actual_compile: {exit_code: 0},
    source_bundle_sha256: createHash("sha256").update(source).digest("hex"),
    binary_sha256: createHash("sha256").update(macho()).digest("hex"),
  }));
  const sources = ["native/EvolutionObserve.swift", "native/Worker.swift", "native/Tokenizer.swift"], observerBytes: Buffer[] = [];
  for (const source of sources) observerBytes.push(Buffer.from(source + "\0"), await readFile(join(project, source)), Buffer.from("\0"));
  const digest = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
  const mlxSource = createHash("sha256");
  for (const file of mlxSourceFiles) { mlxSource.update(file + "\n"); mlxSource.update(await readFile(join(project, file))); }
  const mlxMetal = files["Contents/Resources/backend/native/mlx.metallib"];
  files["Contents/Resources/backend/native/luban-mlx-worker.json"] = Buffer.from(JSON.stringify({
    schema: "luban-mlx-worker-build-v1", language: "Swift", protocol: "luban-worker-jsonl-v1", cpu_smoke: true,
    mlx_swift_lm: "3.32.3", mlx_swift: "0.32.3", assets: ["luban-mlx-worker", "mlx.metallib"],
    binary_sha256: digest(macho()), source_sha256: mlxSource.digest("hex"),
    metallib_bytes: mlxMetal.length, metallib_sha256: digest(mlxMetal),
  }));
  files["Contents/Resources/backend/native/luban-evolution-observe.build.json"] = Buffer.from(JSON.stringify({
    schema: "luban-native-evolution-observer-build-v1", sources, source_bundle_sha256: digest(Buffer.concat(observerBytes)),
    binary_sha256: digest(macho()), precision: "mixed-q4-g16", layers: 24, seed: 42, temperature: 0, python_invocations: 0, compiled: true, signed: true,
  }));
  const binding = (bytes: Buffer) => ({ bytes: bytes.length, sha256: digest(bytes) });
  for (const name of evolutionLicenseNames) files[evolutionCapsulePrefix + name] = await readFile(join(project, "build/native-evolution/Resources", name));
  const payload = Object.fromEntries(evolutionBinaryNames.map(name => [name, { final: binding(files[evolutionCapsulePrefix + name]) }]));
  files[evolutionPrefix + "worker.js"] = await readFile(join(project, "build/native-evolution/worker.js"));
  files[evolutionCapsulePrefix + "native/runtime-release.json"] = Buffer.from(JSON.stringify({
    schema: "luban-native-unified-service-role-capsule-v6", payload,
    programs: Object.fromEntries(evolutionBinaryNames.filter(name => name.startsWith("native/bin/")).map((name, index) =>
      [evolutionRoleNames[index], { path: name, proof: { source_build_sha256: "a".repeat(64) } }])),
    libraries: Object.fromEntries(evolutionBinaryNames.filter(name => name.startsWith("native/lib/")).map(name => [name, payload[name].final])),
    licenses: Object.fromEntries(evolutionLicenseNames.map(name => [name, binding(files[evolutionCapsulePrefix + name])])),
    assets: { "native/bin/luban-worker.json": binding(files[evolutionCapsulePrefix + "native/bin/luban-worker.json"]) },
  }));
  files["Contents/Resources/LubanIcon.icns"] = await readFile(resolve(import.meta.dir, "../native/assets/LubanIcon.icns"));
  for (const name of requiredPayload.filter(p => p.startsWith("Contents/Resources/backend/src/luban/web/"))) files[name] = await readFile(join(project, "src/luban/web", name.split("/web/")[1]));
  for (const [source, name] of Object.entries(licenseFiles)) files[name] = await readFile(resolve(import.meta.dir, "..", source));
  for (const [name, bytes] of Object.entries(files)) {
    const destination = join(app, name);
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, bytes);
  }
  return files;
}
