/** Product-only, relocatable CPython and locked wheel payload; no venv or local checkout. */
import { createReadStream, existsSync } from "node:fs";
import { chmod, copyFile, lstat, mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import lock from "./python-runtime-lock.json";
import { parseImage, system, type Image } from "../backend/native-runtime-macho";
import { checkReleaseMachO } from "./release-macho";

const root = resolve(import.meta.dir, "..");
export const pythonRuntimePrefix = "Contents/Resources/backend/python";
export const pythonRuntimeSource = join(homedir(), "Library/Application Support/Luban/runtime/python/bin/python");
const licenseRoot = join(root, "native/python/licenses");
const licenseNames = ["standalone-LICENSE", "standalone-LICENSE.bzip2.txt", "standalone-LICENSE.cpython.txt",
  "standalone-LICENSE.expat.txt", "standalone-LICENSE.libedit.txt", "standalone-LICENSE.libffi.txt",
  "standalone-LICENSE.liblzma.txt", "standalone-LICENSE.mpdecimal.txt", "standalone-LICENSE.ncurses.txt",
  "standalone-LICENSE.openssl-3.txt", "standalone-LICENSE.sqlite.txt", "standalone-LICENSE.zlib.txt", "tokenizers-LICENSE"];
type File = { source: string; path: string; bytes: number };
export type PythonRuntimePlan = { base: string; site: string; files: File[]; packages: typeof lock.packages };
function need(ok: unknown, message: string): asserts ok { if (!ok) throw Error(message); }
async function binarySHA(path: string) {
  const hash = new Bun.CryptoHasher("sha256");
  for await (const block of createReadStream(path)) hash.update(block as Buffer);
  return hash.digest("hex");
}
export function safeRuntimePath(path: string) {
  return path.length > 0 && !isAbsolute(path) && !path.includes("\\") && !path.includes("\0")
    && path.split("/").every(part => !!part && ![".", "..", "internal", "__pycache__"].includes(part) && !part.startsWith("._"));
}
async function run(argv: string[], env: Record<string, string> = { PATH: "/usr/bin:/bin" }) {
  const child = Bun.spawn(argv, { env, cwd: "/", stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  need(code === 0, `${argv[0]} failed (${code}): ${stderr.slice(-4000)}`); return { stdout, stderr };
}

/** Rebuild the allowlist from pinned installed distributions, not from an app's own manifest. */
export async function pythonRuntimePlan(python = pythonRuntimeSource): Promise<PythonRuntimePlan> {
  const result = await run([python, "-I", "-B", "-c", String.raw`
import importlib.metadata as md, json, pathlib, sys, sysconfig
lock=json.loads(sys.argv[1]); base=pathlib.Path(sys.base_prefix).resolve(); site=pathlib.Path(sysconfig.get_path('purelib')).resolve()
assert sys.version.split()[0]==lock['python'], 'Python version differs'
assert (base/'BUILD').read_text().strip()==lock['standalone_build'], 'CPython standalone build differs'
files={}
def add(source,dest,scope):
 source=source.resolve(); assert source.is_relative_to(scope) and source.is_file(), 'External runtime link'
 assert dest not in files or files[dest]['source']==str(source), 'Overlapping package files'
 files[dest]={'source':str(source),'path':dest,'bytes':source.stat().st_size}
add(base/'bin/python3.12','bin/python',base)
add(base/'lib/libpython3.12.dylib','lib/libpython3.12.dylib',base)
stdlib=base/'lib/python3.12'
for p in stdlib.rglob('*'):
 rel=p.relative_to(stdlib); parts=rel.parts
 if any(x in ('__pycache__','site-packages','ensurepip','idlelib','turtledemo','tkinter','test') or x.startswith('config-') for x in parts): continue
 if not p.is_file() or p.name.startswith(('._','_tkinter.')) or p.suffix in ('.pyc','.pyo','.a'): continue
 add(p,'lib/python3.12/'+rel.as_posix(),base)
for name,version in lock['packages'].items():
 d=md.distribution(name); assert d.version==version, name+' version differs'
 assert d.files, name+' RECORD missing'
 for entry in d.files:
  rel=pathlib.PurePosixPath(str(entry))
  if rel.is_absolute() or '..' in rel.parts: continue
  if '__pycache__' in rel.parts or any(x.startswith('._') for x in rel.parts) or rel.suffix in ('.pyc','.pyo','.pth'): continue
  if rel.name in ('direct_url.json','INSTALLER','REQUESTED','RECORD'): continue
  p=site/rel
  add(p,'lib/python3.12/site-packages/'+rel.as_posix(),site)
print(json.dumps({'base':str(base),'site':str(site),'files':list(files.values()),'packages':lock['packages']}))
`, JSON.stringify(lock)]);
  const plan: PythonRuntimePlan = JSON.parse(result.stdout);
  for (const name of licenseNames) {
    const source = join(licenseRoot, name);
    need((await lstat(source)).isFile() && !name.startsWith("."), "Invalid Python license source");
    plan.files.push({ source, path: "licenses/" + name, bytes: (await stat(source)).size });
  }
  need(plan.files.length > 1000 && new Set(plan.files.map(file => file.path)).size === plan.files.length, "Incomplete/duplicate runtime inventory");
  for (const file of plan.files) need(safeRuntimePath(file.path), "Unsafe Python runtime path: " + file.path);
  plan.files.sort((a, b) => a.path.localeCompare(b.path)); return plan;
}

export function isMachO(bytes: Uint8Array) {
  return ["cffaedfe", "feedfacf", "cefaedfe", "feedface", "cafebabe", "bebafeca", "cafebabf", "bfbafeca"].includes(Buffer.from(bytes.subarray(0, 4)).toString("hex"));
}
async function inspect(path: string) {
  const [link, loads] = await Promise.all([run(["/usr/bin/otool", "-L", path]), run(["/usr/bin/otool", "-l", path])]);
  return parseImage(link.stdout, loads.stdout);
}
function loaderRelative(binary: string, target: string) { return "@loader_path/" + relative(dirname(binary), target); }
function expandPath(path: string, binary: string, executable: string) {
  return path.replace(/^@loader_path(?=\/|$)/, dirname(binary)).replace(/^@executable_path(?=\/|$)/, dirname(executable));
}

/** Resolve every non-system dependency against the final candidate, including nested @rpaths. */
export function checkPythonLinkage(images: Record<string, Image>, executable: string) {
  const known = new Set(Object.keys(images));
  for (const [binary, image] of Object.entries(images)) {
    const rpaths = [...image.rpaths.map(path => resolve(expandPath(path, binary, executable))),
      ...(images[executable]?.rpaths ?? []).map(path => resolve(expandPath(path, executable, executable)))];
    need(rpaths.every(path => !path.startsWith("@") && (path === dirname(dirname(executable)) || path.startsWith(dirname(dirname(executable)) + "/"))), "Runtime search path escapes package");
    for (const dependency of image.dependencies) {
      if (system(dependency)) continue;
      const targets = dependency.startsWith("@rpath/") ? rpaths.map(path => resolve(path, dependency.slice(7)))
        : [resolve(expandPath(dependency, binary, executable))];
      need(targets.some(path => known.has(path)), `Python dependency escapes/unresolved: ${binary}: ${dependency}`);
    }
  }
}

export async function packagePythonRuntime(output: string, python = pythonRuntimeSource) {
  output = resolve(output);
  need(!existsSync(output), "Python runtime output must be fresh");
  const plan = await pythonRuntimePlan(python);
  await mkdir(output, { recursive: true });
  const sourceToTarget = new Map(plan.files.map(file => [file.source, join(output, file.path)]));
  const binaries: string[] = [];
  const binaryBindings: Record<string, { source: string; source_sha256: string; final_sha256: string; bytes: number }> = {};
  for (const file of plan.files) {
    const target = join(output, file.path); await mkdir(dirname(target), { recursive: true }); await copyFile(file.source, target);
    const handle = Bun.file(target), header = new Uint8Array(await handle.slice(0, 4).arrayBuffer());
    if (isMachO(header)) binaries.push(target);
    await chmod(target, file.path === "bin/python" ? 0o755 : 0o644);
  }
  for (const binary of binaries) {
    const image = await inspect(binary), args: string[] = [];
    for (const dependency of image.dependencies) {
      if (system(dependency) || dependency.startsWith("@")) continue;
      const target = sourceToTarget.get(dependency);
      need(target, "Unbound absolute Python dependency: " + dependency);
      args.push("-change", dependency, loaderRelative(binary, target));
    }
    if (image.id) args.push("-id", loaderRelative(binary, binary));
    for (const rpath of image.rpaths) {
      if (rpath.startsWith("@")) continue;
      const source = [plan.base, plan.site].find(prefix => rpath === prefix || rpath.startsWith(prefix + "/"));
      // Wheels sometimes retain unused CI build search directories. Remove them;
      // the complete post-relocation dependency walk still rejects missing libraries.
      if (!source) { args.push("-delete_rpath", rpath); continue; }
      const destination = source === plan.site ? join(output, "lib/python3.12/site-packages", relative(source, rpath)) : join(output, relative(source, rpath));
      args.push("-rpath", rpath, loaderRelative(binary, destination));
    }
    if (args.length) await run(["/usr/bin/install_name_tool", ...args, binary]);
    await run(["/usr/bin/strip", "-S", binary]);
    await run(["/usr/bin/codesign", "--force", "--sign", "-", binary]);
    await run(["/usr/bin/codesign", "--verify", "--strict", binary]);
    checkReleaseMachO(await readFile(binary), binary);
    const name = relative(output, binary), source = plan.files.find(file => file.path === name)!;
    binaryBindings[name] = { source: name, source_sha256: await binarySHA(source.source), final_sha256: await binarySHA(binary), bytes: (await stat(binary)).size };
  }
  const images: Record<string, Image> = {};
  for (const path of binaries) images[path] = await inspect(path);
  checkPythonLinkage(images, join(output, "bin/python"));
  const manifest = { schema: "luban-python-runtime-v1", python: lock.python, standalone_build: lock.standalone_build,
    platform: lock.platform, packages: lock.packages, files: plan.files.map(file => file.path),
    binaries: binaries.map(path => relative(output, path)).sort(),
    binary_bindings: binaryBindings,
    licenses: plan.files.filter(file => /(^|\/)(?:licenses?|copying|notice)(?:[^/]*)(\/|$)/i.test(file.path)).map(file => file.path),
    bytes: (await Promise.all(plan.files.map(file => stat(join(output, file.path))))).reduce((n, file) => n + file.size, 0) };
  await writeFile(join(output, "runtime-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  await assertPythonRuntime(output, plan);
  return { output, ...manifest };
}

/** Host build gate uses a trusted source plan; never trust manifest.files alone. */
export async function assertPythonRuntime(output: string, plan?: PythonRuntimePlan) {
  plan ??= await pythonRuntimePlan();
  const manifest = JSON.parse(await readFile(join(output, "runtime-manifest.json"), "utf8"));
  need(manifest.schema === "luban-python-runtime-v1" && manifest.python === lock.python && manifest.standalone_build === lock.standalone_build
    && JSON.stringify(manifest.packages) === JSON.stringify(lock.packages), "Python runtime versions differ");
  const allowed = new Set([...plan.files.map(file => file.path), "runtime-manifest.json"]), found = new Set<string>();
  const licenses = plan.files.filter(file => /(^|\/)(?:licenses?|copying|notice)(?:[^/]*)(\/|$)/i.test(file.path));
  need(JSON.stringify(manifest.licenses) === JSON.stringify(licenses.map(file => file.path)), "Python license inventory differs");
  const binaries: string[] = [];
  let bytes = 0;
  async function visit(path: string) {
    for (const name of await readdir(path)) {
      const target = join(path, name), item = await lstat(target), rel = relative(output, target);
      need(safeRuntimePath(rel) && !item.isSymbolicLink(), "Unsafe Python runtime entry: " + rel);
      if (item.isDirectory()) { await visit(target); continue; }
      need(item.isFile() && allowed.has(rel), "Unapproved Python runtime file: " + rel); found.add(rel);
      if (rel !== "runtime-manifest.json") bytes += item.size;
      if (isMachO(new Uint8Array(await Bun.file(target).slice(0, 4).arrayBuffer()))) binaries.push(rel);
    }
  }
  await visit(output);
  need(found.size === allowed.size && [...allowed].every(path => found.has(path)), "Python runtime file missing");
  need(JSON.stringify(manifest.files) === JSON.stringify(plan.files.map(file => file.path))
    && JSON.stringify([...manifest.binaries].sort()) === JSON.stringify(binaries.sort()), "Python runtime inventory differs");
  need(manifest.binary_bindings && JSON.stringify(Object.keys(manifest.binary_bindings).sort()) === JSON.stringify(binaries.sort()), "Python binary binding inventory differs");
  need(manifest.bytes === bytes, "Python runtime byte count differs");
  for (const file of licenses) {
    need((await readFile(file.source)).equals(await readFile(join(output, file.path))), "Python license changed: " + file.path);
  }
  const images: Record<string, Image> = {};
  for (const rel of binaries) {
    const path = join(output, rel); checkReleaseMachO(await readFile(path), rel);
    const binding = manifest.binary_bindings[rel];
    need(binding.source === rel && /^[a-f0-9]{64}$/.test(binding.source_sha256) && binding.bytes === (await stat(path)).size
      && binding.final_sha256 === await binarySHA(path), "Python release binary binding differs: " + rel);
    await run(["/usr/bin/codesign", "--verify", "--strict", path]); images[path] = await inspect(path);
  }
  checkPythonLinkage(images, join(output, "bin/python"));
  return { files: allowed.size, binaries, licenses: licenses.map(file => file.path), bytes };
}

/** Genuine isolated interpreter/import/CPU I/O check with no PYTHONPATH, venv or user-site. */
export async function probePythonRuntime(output: string) {
  const executable = join(resolve(output), "bin/python");
  const temporary = await mkdtemp(join(dirname(resolve(output)), "python-probe-"));
  try {
  const result = await run([executable, "-I", "-B", "-c", String.raw`
import io,json,pathlib,sys,ssl,tempfile
root=pathlib.Path(sys.executable).resolve().parent.parent
assert pathlib.Path(sys.prefix).resolve()==root and pathlib.Path(sys.base_prefix).resolve()==root
assert all(pathlib.Path(p).resolve().is_relative_to(root) for p in sys.path)
import torch,numpy as np,PIL.Image,soundfile,safetensors.torch,tokenizers,transformers,fastapi,uvicorn,httpx,psutil,certifi
tensor=torch.tensor([1.,2.]); assert torch.equal(tensor+tensor,torch.tensor([2.,4.]))
assert torch.equal(safetensors.torch.load(safetensors.torch.save({'x':tensor}))['x'],tensor)
audio=io.BytesIO(); soundfile.write(audio,np.zeros(160,dtype=np.float32),16000,format='WAV'); audio.seek(0)
assert soundfile.read(audio)[1]==16000
assert PIL.Image.new('RGB',(2,2)).size==(2,2)
assert transformers.AutoConfig.for_model('qwen3_5').model_type=='qwen3_5'
assert len(ssl.create_default_context(cafile=certifi.where()).get_ca_certs())>0
outside=[]
for name,module in list(sys.modules.items()):
 path=getattr(module,'__file__',None)
 if name in ('torch.ops','torch.classes'): continue # PyTorch exposes synthetic module filenames.
 if name=='_remote_module_non_scriptable' and path and pathlib.Path(path).resolve().is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve()): continue
 if path and not path.startswith('<') and not pathlib.Path(path).resolve().is_relative_to(root): outside.append([name,path])
assert not outside, outside
print(json.dumps({'python':sys.version.split()[0],'prefix':str(root),'isolated':sys.flags.isolated,'user_site_disabled':bool(sys.flags.no_user_site),'modules_checked':len(sys.modules),'torch':torch.__version__,'transformers':transformers.__version__,'safetensors_roundtrip':True,'audio_roundtrip':True,'ssl_certificates':True,'outside_imports':outside}))
`], { PATH: "/usr/bin:/bin", TMPDIR: temporary, DYLD_PRINT_LIBRARIES: "1", TOKENIZERS_PARALLELISM: "false" });
  const plan = await pythonRuntimePlan();
  need(!result.stderr.includes(plan.base) && !result.stderr.includes(plan.site), "Runtime dynamically loaded original installation");
  return { ...JSON.parse(result.stdout), dynamic_loader_external_source: false };
  } finally { await rm(temporary, { recursive: true, force: true }); }
}

if (import.meta.main) {
  const [output, python] = Bun.argv.slice(2); need(output, "Usage: bun scripts/package-python-runtime.ts /new/runtime [source-python]");
  const result = await packagePythonRuntime(output, python); console.log(JSON.stringify({ output, files: result.files.length, binaries: result.binaries.length, bytes: result.bytes }));
}
