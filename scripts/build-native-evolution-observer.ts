/** Build the ordinary-Q4 acceptance executable from the existing product tokenizer/worker. */
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { createHash } from "node:crypto";

const root = resolve(import.meta.dir, ".."), output = resolve(root, "build/native/luban-evolution-observe");
const sources = ["native/EvolutionObserve.swift", "native/Worker.swift", "native/Tokenizer.swift"];
const sha = (data: Uint8Array) => createHash("sha256").update(data).digest("hex");
await mkdir(dirname(output), { recursive: true });
const commands = [["/usr/bin/swiftc", ...sources, "-O", "-o", output],
  ["/usr/bin/strip", "-S", "-x", output], ["/usr/bin/codesign", "--force", "--sign", "-", output],
  ["/usr/bin/codesign", "--verify", "--strict", output]];
const began = performance.now();
for (const args of commands) {
  const child = Bun.spawn(args, { cwd: root, stdout: "inherit", stderr: "inherit" });
  assert.equal(await child.exited, 0, "原生Q4观察程序构建失败：" + args[0]);
}
const sourceBytes: Buffer[] = [];
for (const source of sources) sourceBytes.push(Buffer.from(source + "\0"), await readFile(resolve(root, source)), Buffer.from("\0"));
await writeFile(output + ".build.json", JSON.stringify({ schema: "luban-native-evolution-observer-build-v1",
  sources, source_bundle_sha256: sha(Buffer.concat(sourceBytes)), binary_sha256: sha(await readFile(output)),
  precision: "mixed-q4-g16", layers: 24, seed: 42, temperature: 0, python_invocations: 0,
  build_seconds: (performance.now() - began) / 1000, compiled: true, signed: true,
  actual_model_validated: false, commands }, null, 2) + "\n");
console.log(output);
