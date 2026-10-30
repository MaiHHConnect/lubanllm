import { randomBytes } from "node:crypto";
import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { chmod, mkdir, mkdtemp, realpath } from "node:fs/promises";
import { createServer } from "node:net";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";
import { Settings, privateJSON, requireStorage } from "./settings.ts";
import { createBackend } from "./server.ts";
import { NativeEvolutionService } from "./native-evolution-service.ts";

const { values } = parseArgs({ args: Bun.argv.slice(2), options: {
  port: { type: "string", default: "8765" }, home: { type: "string" },
  project: { type: "string", default: process.cwd() }, backend: { type: "string" },
  autoload: { type: "boolean", default: false },
  "no-model": { type: "boolean", default: false },
}, strict: true });
if (!/^\d+$/.test(values.port!) || Number(values.port) > 65535) throw new Error("端口需要0–65535的整数");
const project = resolve(values.project!), backendRoot = resolve(values.backend ?? join(project, "src"));
const settings = new Settings(values.home);
const cachedPython = join(settings.home, "runtime/python/bin/python");
const bundledPython = join(dirname(backendRoot), "python/bin/python");
const development = backendRoot === join(project, "src");
// Installed apps must never depend on the developer checkout or a user's old venv.
const python = !development ? bundledPython : existsSync(cachedPython) ? cachedPython : join(project, ".venv/bin/python");
if (!existsSync(python) || !existsSync(join(backendRoot, "luban/server.py"))) throw new Error("应用运行文件缺失，请重新下载安装应用。");
const rpcToken = randomBytes(32).toString("base64url"), computeToken = randomBytes(32).toString("base64url");
const computePort = await new Promise<number>((done, fail) => {
  const reservation = createServer(); reservation.once("error", fail);
  reservation.listen(0, "127.0.0.1", () => {
    const address = reservation.address();
    if (!address || typeof address === "string") { reservation.close(); fail(new Error("无法分配私有计算端口")); return; }
    reservation.close(error => error ? fail(error) : done(address.port));
  });
});
let release = { version: "0.3.10", build: 18 };
const metadata = join(dirname(backendRoot), "release.json");
if (existsSync(metadata)) release = JSON.parse(readFileSync(metadata, "utf8"));
const options = { settings, port: Number(values.port), computeURL: `http://127.0.0.1:${computePort}`, computeToken,
  rpcToken, webRoot: join(backendRoot, "luban/web"), version: process.env.LUBAN_DESKTOP_VERSION ?? release.version,
  build: Number(process.env.LUBAN_DESKTOP_BUILD ?? release.build) };
if (typeof options.version !== "string" || !options.version || !Number.isSafeInteger(options.build) || options.build < 1) throw new Error("应用版本资料无效");
async function nativeWorker() {
  const packaged = backendRoot !== join(project, "src");
  const workerPath = packaged ? join(dirname(dirname(backendRoot)), "native-evolution/worker.js")
    : join(project, "build/native-evolution/worker.js");
  if (!existsSync(workerPath)) throw new Error("原生进化程序缺失，请重新构建应用");
  return import(pathToFileURL(workerPath).href);
}
const nativeEvolution = new NativeEvolutionService({
  compute: (path, body) => fetch(options.computeURL + path, { method: "POST", headers: {
    authorization: "Bearer " + computeToken, "content-type": "application/json" },
    body: JSON.stringify(body), redirect: "manual", proxy: null }),
  async open(lease, signal, material, mode) {
    signal.throwIfAborted();
    const worker = await nativeWorker();
    if (typeof worker.openNativeExpertRuntime !== "function") throw new Error("原生专家程序未提供完整运行入口");
    requireStorage(lease.root);
    const parent = join(settings.home, "native-evolution-work");
    await mkdir(parent, { recursive: true, mode: 0o700 });
    await chmod(parent, 0o700);
    const work = await mkdtemp(join(await realpath(parent), "round-"));
    signal.throwIfAborted();
    if(settings.snapshot().native_inference_precision==='mlx-q4')return worker.openMLXLearningRuntime({home:settings.home,model:lease.model,store:lease.store,
      root:lease.root,identity:lease.identity,work,material,mode,memory:application.memory(lease.root),signal,binary:backendRoot===join(project,'src')?join(project,'build/native/luban-mlx-worker'):join(dirname(backendRoot),'native/luban-mlx-worker'),
      routes:async(records:unknown[])=>{signal.throwIfAborted();const response=await fetch(options.computeURL+'/internal/native-evolution/mlx-routes',{method:'POST',headers:{authorization:'Bearer '+computeToken,'content-type':'application/json'},body:JSON.stringify({lease_id:lease.lease_id,records}),redirect:'manual',proxy:null});const value:any=await response.json();if(!response.ok)throw Error(value.detail??'MLX原路由验证失败');return value;}});
    return worker.openNativeExpertRuntime({ home: settings.home, model: lease.model, store: lease.store,
      root: lease.root, identity: lease.identity, work, material, mode });
  },
});
const application = createBackend(Object.assign(options, { nativeEvolution,
  async inspectNativeEvolution(scope: {root:string;store:string;identity:string}, memory: import("./memory").Memory) {
    const worker = await nativeWorker();
    if (typeof worker.inspectNativeExpertRuntime !== "function") throw new Error("原生专家材料与调度入口尚未完成接线");
    if(settings.snapshot().native_inference_precision==='mlx-q4')return worker.inspectMLXLearningRuntime({...scope,memory,home:settings.home,model:String(settings.snapshot().model_path),binary:backendRoot===join(project,'src')?join(project,'build/native/luban-mlx-worker'):join(dirname(backendRoot),'native/luban-mlx-worker')});
    return worker.inspectNativeExpertRuntime({ ...scope, memory });
  },
}));
const server = Bun.serve({ hostname: "127.0.0.1", port: options.port, idleTimeout: 255,
  maxRequestBodySize: 16 * 1024 ** 3 + 32 * 1024 ** 2, fetch: application.fetch });
options.port = server.port;
const environment: Record<string, string> = {};
for (const [key, value] of Object.entries(process.env)) {
  if (value !== undefined && !/^(PYTHON|DYLD_)/.test(key) && !["VIRTUAL_ENV", "CONDA_PREFIX"].includes(key)) environment[key] = value;
}
environment.PYTHONPATH = backendRoot;
environment.PYTHONDONTWRITEBYTECODE = "1";
environment.PYTHONNOUSERSITE = "1";
if (!development) environment.SSL_CERT_FILE = join(dirname(dirname(bundledPython)), "lib/python3.12/site-packages/certifi/cacert.pem");
if (python === cachedPython) environment.DYLD_LIBRARY_PATH = join(settings.home, "runtime/python/lib/python3.12/site-packages/torch/lib");
environment.TOKENIZERS_PARALLELISM = "false";
environment.LUBAN_BUN_RPC_URL = `http://127.0.0.1:${server.port}/internal/rpc`;
environment.LUBAN_BUN_RPC_TOKEN = rpcToken;
environment.LUBAN_COMPUTE_TOKEN = computeToken;
environment.LUBAN_PUBLIC_PORT = String(server.port);
environment.LUBAN_COMPUTE_AUTOLOAD = values.autoload && !values["no-model"] ? "1" : "0";
environment.LUBAN_NATIVE_EVOLUTION = "1";
let compute: ReturnType<typeof Bun.spawn>;
try {
  compute = Bun.spawn([python, "-P", "-m", "luban.server", "--port", String(computePort), "--home", settings.home],
    { cwd: settings.home, env: environment, detached: true, stdout: "inherit", stderr: "inherit" });
} catch (error) { application.close(); server.stop(true); throw error; }
const launch = join(settings.home, "launch.json");
let stopping = false;
function terminateGroup(signal: NodeJS.Signals) {
  try { process.kill(-compute.pid, signal); } catch (error: any) { if (error.code !== "ESRCH") throw error; }
}
async function stop(exitCode: number) {
  if (stopping) return;
  stopping = true;
  try { await nativeEvolution.close(); }
  catch (error) {
    console.error("原生进化清理尚未完成，保留计算进程与任务所有者：" + settings.mask(String(error)));
    stopping = false;
    return;
  }
  application.beginClose();
  // Keep data RPC serving until computation has saved/rolled back its last transaction.
  if (compute.exitCode === null) {
    compute.kill("SIGTERM");
    let timer: ReturnType<typeof setTimeout> | undefined;
    const completed = await Promise.race([compute.exited.then(() => true), new Promise<false>(done => { timer = setTimeout(() => done(false), 60_000); })]);
    if (timer) clearTimeout(timer);
    if (!completed) { terminateGroup("SIGKILL"); await compute.exited; }
  }
  terminateGroup("SIGTERM");
  application.close(); server.stop(true);
  try {
    const record = JSON.parse(readFileSync(launch, "utf8"));
    if (record.pid === process.pid) unlinkSync(launch);
  } catch (error: any) { if (error.code !== "ENOENT") console.error("无法清理启动回执"); }
  process.exit(exitCode);
}
process.on("SIGTERM", () => { void stop(0); });
process.on("SIGINT", () => { void stop(0); });
void compute.exited.then(code => { if (!stopping) { console.error("模型计算进程已退出"); void stop(code || 1); } });
try {
  privateJSON(launch, { port: server.port, pid: process.pid, token: settings.secret("ui_token"), backend: "bun", compute_pid: compute.pid });
} catch {
  console.error("无法写入启动回执");
  await stop(1);
}
console.log(`Bun本地服务已启动：127.0.0.1:${server.port}`);
