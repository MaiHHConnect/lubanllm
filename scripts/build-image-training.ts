// Build only; never loads model weights or launches training.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";
const root = resolve(import.meta.dir, "..");
const torch = join(root, ".venv/lib/python3.12/site-packages/torch");
const output = join(root, "build/native/libluban_image_training.dylib");
await mkdir(join(root, "build/native"), { recursive: true });
const sources = ["native/ImageTraining.cpp"];
const sha = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const argv = ["/usr/bin/clang++", "-std=c++17", "-O1", "-g0", "-dynamiclib", "-fPIC",
  "-Wno-deprecated-declarations", "-I" + join(torch, "include"), "-I" + join(torch, "include/torch/csrc/api/include"),
  join(root, "native/ImageTraining.cpp"), "-L" + join(torch, "lib"), "-Wl,-rpath,@loader_path/../python-runtime/torch/lib",
  "-Wl,-rpath," + join(torch, "lib"), "-ltorch", "-ltorch_cpu", "-lc10", "-o", output];
const child = Bun.spawn(argv, { stdout: "pipe", stderr: "pipe" });
const [stdout, stderr, exit] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
await writeFile(output + ".build.json", JSON.stringify({ argv, stdout, stderr, exit, model_loaded: false, python_linkage: false }, null, 2));
if (exit) throw new Error(stderr);
const signing = [];
for (const args of [["--force", "--sign", "-", output], ["--verify", "--strict", output]]) {
  const command = ["/usr/bin/codesign", ...args], actual = Bun.spawn(command, {stdout: "pipe", stderr: "pipe"});
  const [stdout, stderr, exit_code] = await Promise.all([new Response(actual.stdout).text(), new Response(actual.stderr).text(), actual.exited]);
  signing.push({argv: command, stdout, stderr, exit_code});
  if (exit_code) { await writeFile(output + ".build.json", JSON.stringify({argv, stdout, stderr, exit, signing, model_loaded: false}, null, 2)); throw new Error("Image training library signing failed"); }
}
const sourceBundle = Buffer.concat(await Promise.all(sources.map(async path =>
  Buffer.concat([Buffer.from(path + "\0"), await readFile(join(root, path)), Buffer.from("\0")]))));
await writeFile(output + ".build.json", JSON.stringify({schema: "luban-native-image-training-build-v1",
  binary_sha256: sha(await readFile(output)), sources, source_bundle_sha256: sha(sourceBundle),
  production_shape: true, model_loaded: false, no_python_linkage: true, python_linkage: false,
  actual_compile: {argv, stdout, stderr, exit_code: exit}, signing,
  dependency_library_directory: join(torch, "lib")}, null, 2) + "\n");
console.log(output);
