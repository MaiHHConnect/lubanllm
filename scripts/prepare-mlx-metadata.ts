/** Copy only the upstream metadata still needed by native MLX into a new directory. */
import { constants } from "node:fs";
import { copyFile, lstat, mkdir, readFile, realpath, rm, stat } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";

const required = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
  "upstream-file-manifest.json", "LICENSE"];
const optional = ["generation_config.json", "merges.txt", "vocab.json", "preprocessor_config.json",
  "processor_config.json", "video_preprocessor_config.json"];

export async function prepareMlxMetadata(source: string, destination: string) {
  if (!isAbsolute(source) || !isAbsolute(destination)) throw new Error("Absolute source and destination paths are required");
  source = await realpath(source);
  destination = resolve(destination);
  const parent = dirname(destination);
  if (!(await stat(parent)).isDirectory()) throw new Error("Destination parent must exist");
  if (destination.startsWith("/Volumes/")) {
    const volume = destination.split("/").slice(0, 3).join("/");
    if ((await stat(volume)).dev === (await stat("/Volumes")).dev) throw new Error("Destination volume is not mounted");
  }
  const files = [...required];
  for (const name of optional) {
    try { await lstat(join(source, name)); files.push(name); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  }
  // A release foundation is a small accepted core replacement, not the BF16
  // backbone. Preserve its explicit four-file payload for first-use recovery.
  let foundation: string | undefined;
  try { foundation = await readFile(join(source, "luban-foundation.json"), "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  if (foundation !== undefined) {
    const record = JSON.parse(foundation);
    if (typeof record.version !== "string" || !/^core-[a-f0-9]{12}$/.test(record.version)) throw new Error("Invalid foundation version");
    const expected = [record.version + ".json", record.version + ".safetensors", record.version + ".retention.json",
      "evolution-" + record.version.slice(5) + ".json"].sort();
    if (JSON.stringify(Object.keys(record.files ?? {}).sort()) !== JSON.stringify(expected)) throw new Error("Invalid foundation file list");
    if (!(await lstat(join(source, "luban-foundation"))).isDirectory()) throw new Error("Foundation must be a plain directory");
    files.push("luban-foundation.json", ...expected.map(name => "luban-foundation/" + name));
  }
  const records = [];
  for (const name of files) {
    const info = await lstat(join(source, name));
    if (!info.isFile() || info.size > 32 * 1024 ** 2) throw new Error("Missing, linked or oversized metadata: " + name);
    records.push({ name, bytes: info.size });
  }
  await mkdir(destination, { mode: 0o700 }); // Existing directories are never replaced.
  try {
    if (foundation !== undefined) await mkdir(join(destination, "luban-foundation"), { mode: 0o700 });
    for (const { name } of records) await copyFile(join(source, name), join(destination, name), constants.COPYFILE_EXCL);
  } catch (error) {
    await rm(destination, { recursive: true, force: true });
    throw error;
  }
  return { source, destination, files: records, bytes: records.reduce((total, row) => total + row.bytes, 0),
    original_weights_copied: false, settings_changed: false };
}

if (import.meta.main) {
  const [source, destination, ...extra] = Bun.argv.slice(2);
  if (!source || !destination || extra.length) throw new Error("Usage: bun scripts/prepare-mlx-metadata.ts /original/model /new/metadata-directory");
  console.log(JSON.stringify(await prepareMlxMetadata(source, destination), null, 2));
}
