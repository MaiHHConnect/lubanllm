/** Existing Mach-O parser and bound dependency closure; callers supply inspection. */
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

export const NAME = /^lib[a-zA-Z0-9_+.-]+\.dylib$/;
export const PYTHON = /libpython|libtorch_python|Python\.framework/i;
export type Binding = { sha256: string; bytes: number };
export type Image = { dependencies: string[]; rpaths: string[]; id: string | null };
function need(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }

export function parseImage(linkage: string, loads: string): Image {
  const dependencies = linkage.split("\n").slice(1).flatMap(line => {
    const match = line.match(/^\s+(.+) \(compatibility version /);
    return match ? [match[1]] : [];
  });
  const rpaths = [...loads.matchAll(/cmd LC_RPATH\s+cmdsize \d+\s+path (.+) \(offset \d+\)/g)].map(match => match[1]);
  const id = loads.match(/cmd LC_ID_DYLIB\s+cmdsize \d+\s+name (.+) \(offset \d+\)/)?.[1] ?? null;
  return { dependencies: dependencies.filter(path => path !== id), rpaths, id };
}
export const system = (path: string) => (path.startsWith("/System/Library/") || path.startsWith("/usr/lib/")) && resolve(path) === path;
export function relativeName(value: string) {
  need(NAME.test(value) && value === basename(value), "Escaping/unsupported native dependency");
  need(!PYTHON.test(value), "Python native linkage forbidden");
  return value;
}

/** Only sources cryptographically bound by the build receipt can enter the closure. */
export async function dependencyClosure(binary: string, bindings: Record<string, Binding>, inspectSource: (path: string) => Promise<Image>) {
  const images: Record<string, Image> = {};
  const names = new Map<string, string>();
  for (const path of Object.keys(bindings)) {
    need(isAbsolute(path) && resolve(path) === path, "Invalid dependency binding path");
    const name = relativeName(basename(path));
    need(!names.has(name), "Duplicate native library basename");
    names.set(name, path);
  }
  const binaryImage = await inspectSource(binary);
  async function visit(path: string, image?: Image) {
    if (images[path]) return;
    image ??= await inspectSource(path);
    need(![...image.dependencies, ...image.rpaths, image.id ?? ""].some(value => PYTHON.test(value)), "Python native linkage forbidden");
    images[path] = image;
    for (const dependency of image.dependencies) {
      if (system(dependency)) continue;
      let target: string | undefined;
      if (dependency.startsWith("@rpath/")) {
        const name = relativeName(dependency.slice("@rpath/".length));
        const bound = names.get(name);
        if (bound) {
          // Resolve using each image's loader location, with executable rpaths as fallback.
          const localPaths = image.rpaths.map(value => value.replace(/^@loader_path(?=\/|$)/, dirname(path)).replace(/^@executable_path(?=\/|$)/, dirname(binary)));
          const executablePaths = binaryImage.rpaths.map(value => value.replace(/^@loader_path(?=\/|$)/, dirname(binary)).replace(/^@executable_path(?=\/|$)/, dirname(binary)));
          target = [...localPaths, ...executablePaths].some(value => !value.startsWith("@") && resolve(value, name) === bound) ? bound : undefined;
        }
      } else if (dependency.startsWith("@loader_path/")) {
        const name = relativeName(dependency.slice("@loader_path/".length));
        const proposed = join(dirname(path), name);
        if (bindings[proposed]) target = proposed;
      } else if (isAbsolute(dependency) && resolve(dependency) === dependency && bindings[dependency]) {
        relativeName(basename(dependency)); target = dependency;
      }
      need(target, `Unresolved/unbound non-system dependency: ${dependency}`);
      await visit(target);
    }
  }
  await visit(binary, binaryImage);
  return images;
}

