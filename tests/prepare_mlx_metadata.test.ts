import { expect, test } from "bun:test";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { prepareMlxMetadata } from "../scripts/prepare-mlx-metadata";

test("lean metadata excludes original weights and never replaces an existing directory", async () => {
  const root = await mkdtemp(join(tmpdir(), "mlx-metadata-"));
  try {
    const source = join(root, "source"), target = join(root, "lean");
    await mkdir(source);
    const names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
      "upstream-file-manifest.json", "LICENSE"];
    for (const name of names) await writeFile(join(source, name), "fixture:" + name);
    await writeFile(join(source, "model.safetensors"), "original-do-not-copy");
    const result = await prepareMlxMetadata(source, target);
    expect(result.original_weights_copied).toBe(false);
    expect((await readdir(target)).sort()).toEqual(names.sort());
    expect(await readFile(join(target, "upstream-file-manifest.json"), "utf8")).toBe("fixture:upstream-file-manifest.json");
    await expect(prepareMlxMetadata(source, target)).rejects.toThrow();
    expect(await readFile(join(source, "model.safetensors"), "utf8")).toBe("original-do-not-copy");
  } finally { await rm(root, { recursive: true, force: true }); }
});
