import { expect, test } from "bun:test";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { assertPythonRuntime, checkPythonLinkage, safeRuntimePath } from "../scripts/package-python-runtime";
import lock from "../scripts/python-runtime-lock.json";

test("runtime paths reject escapes, internal product data, metadata and links", () => {
  for (const value of ["/etc/passwd", "../secret", "lib/../../secret", "lib//x", "lib\\secret", "internal/map.html", "lib/._module", "lib/__pycache__/x.pyc", "lib/a\0b"]) {
    expect(safeRuntimePath(value)).toBe(false);
  }
  expect(safeRuntimePath("lib/python3.12/site-packages/torch/lib/libtorch_cpu.dylib")).toBe(true);
});

test("Python library closure permits only packaged or system dependencies", () => {
  const executable = "/package/python/bin/python", library = "/package/python/lib/libpython3.12.dylib";
  const images = {
    [executable]: { dependencies: ["@rpath/libpython3.12.dylib", "/usr/lib/libSystem.B.dylib"], rpaths: ["@executable_path/../lib"], id: null },
    [library]: { dependencies: [], rpaths: [], id: "@loader_path/libpython3.12.dylib" },
  };
  expect(() => checkPythonLinkage(images, executable)).not.toThrow();
  expect(() => checkPythonLinkage({ ...images, [executable]: { ...images[executable], dependencies: ["/Users/developer/libpython3.12.dylib"] } }, executable)).toThrow("escapes/unresolved");
  expect(() => checkPythonLinkage({ ...images, [executable]: { ...images[executable], rpaths: ["@loader_path/../../../elsewhere"] } }, executable)).toThrow("search path escapes");
});

test("runtime manifest cannot omit licenses or admit unknown files", async () => {
  const work = await mkdtemp(join(tmpdir(), "luban-runtime-gate-"));
  try {
    const output = join(work, "runtime"), source = join(work, "NOTICE");
    await mkdir(join(output, "licenses"), { recursive: true });
    await writeFile(source, "OK"); await writeFile(join(output, "licenses/NOTICE"), "OK");
    const plan = { base: work, site: work, packages: lock.packages, files: [{ source, path: "licenses/NOTICE", bytes: 2 }] };
    const manifest = { schema: "luban-python-runtime-v1", python: lock.python, standalone_build: lock.standalone_build,
      packages: lock.packages, files: ["licenses/NOTICE"], licenses: ["licenses/NOTICE"], binaries: [], binary_bindings: {}, bytes: 2 };
    const write = (value: unknown) => writeFile(join(output, "runtime-manifest.json"), JSON.stringify(value));
    await write(manifest); expect((await assertPythonRuntime(output, plan)).files).toBe(2);
    await write({ ...manifest, licenses: [] });
    await expect(assertPythonRuntime(output, plan)).rejects.toThrow("license inventory differs");
    await write(manifest); await writeFile(join(output, "licenses/NOTICE"), "NO");
    await expect(assertPythonRuntime(output, plan)).rejects.toThrow("license changed");
    await writeFile(join(output, "licenses/NOTICE"), "OK"); await writeFile(join(output, "secret.json"), "{}");
    await expect(assertPythonRuntime(output, plan)).rejects.toThrow("Unapproved Python runtime file");
  } finally { await rm(work, { recursive: true, force: true }); }
});
