/** Local SDK smoke: temporary Bun candidate over the unchanged installed model API. */
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createBackend } from "../backend/server.ts";
import { Settings } from "../backend/settings.ts";

const [sdkDirectory, rawParent] = process.argv.slice(2);
if (!sdkDirectory || !rawParent) throw new Error("Usage: bun scripts/verify-claude-api.ts /path/to/@anthropic-ai/sdk /mounted/experiment/parent");
const volume = "/Volumes/移动数据库";
if (!resolve(rawParent).startsWith(volume + "/") || statSync(volume).dev === statSync("/Volumes").dev) throw new Error("Expected mounted research volume");
const installedHome = join(homedir(), "Library/Application Support/Luban");
const installed = JSON.parse(readFileSync(join(installedHome, "settings.json"), "utf8"));
const credentials = JSON.parse(readFileSync(join(installedHome, "credentials.json"), "utf8"));
const launch = JSON.parse(readFileSync(join(installedHome, "launch.json"), "utf8"));
if (installed.teacher?.answer_mode !== "local_first") throw new Error("Smoke requires the existing local_first setting; it does not change settings");
const directory = mkdtempSync(join(tmpdir(), "luban-claude-smoke-"));
const settings = new Settings(join(directory, "home"));
settings.update({ storage_root: join(directory, "data") });
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response("starting", { status: 503 }) });
const backend = createBackend({ settings, port: server.port!, computeURL: `http://127.0.0.1:${launch.port}`,
  computeToken: credentials.access_key, rpcToken: "unused-isolated-smoke-rpc", webRoot: resolve("src/luban/web"), version: "candidate", build: 0 });
server.reload({ fetch: backend.fetch });
const redact = (text: string) => [credentials.access_key, credentials.teacher_key, credentials.ui_token, launch.token, settings.secret("access_key")]
  .filter(Boolean).reduce((value, key) => value.replaceAll(key, "[redacted]"), text);
try {
  const health = await (await fetch(`http://127.0.0.1:${launch.port}/api/health`)).json() as any;
  if (health.state !== "ready") throw new Error("Installed model must already be ready");
  const { default: Anthropic } = await import(pathToFileURL(join(resolve(sdkDirectory), "index.mjs")).href);
  const sdkVersion = JSON.parse(readFileSync(join(sdkDirectory, "package.json"), "utf8")).version;
  const client = new Anthropic({ baseURL: `http://127.0.0.1:${server.port}`, apiKey: settings.secret("access_key"), maxRetries: 0, timeout: 30000 });
  const params = { model: "luban0", max_tokens: 32, temperature: 0, system: "请简短回答。",
    messages: [{ role: "user", content: "只输出数字：2+3等于多少？" }] };
  const started = performance.now();
  const plain = await client.messages.create(params);
  const stream = client.messages.stream(params), deltas: string[] = [];
  stream.on("text", (text: string) => deltas.push(text));
  const streamed = await stream.finalMessage();
  const openaiResponse = await fetch(`http://127.0.0.1:${server.port}/v1/chat/completions`, {
    method: "POST", headers: { authorization: "Bearer " + settings.secret("access_key"), "content-type": "application/json" },
    body: JSON.stringify({ model: "luban0", max_tokens: 32, temperature: 0, thinking: "off",
      messages: [{ role: "system", content: params.system }, ...params.messages] }),
  });
  if (!openaiResponse.ok) throw new Error("OpenAI smoke failed: " + openaiResponse.status);
  const openai = await openaiResponse.json() as any;
  const content = (value: any) => value.content.map((block: any) => block.text).join("");
  if (!content(plain).trim() || content(plain) !== content(streamed) || content(streamed) !== deltas.join("")
      || content(plain) !== openai.choices[0].message.content) throw new Error("Three protocol outputs differ or are empty");
  if (streamed.usage.input_tokens !== plain.usage.input_tokens || streamed.usage.output_tokens !== plain.usage.output_tokens) throw new Error("SDK final usage differs");
  let toolRejected = false;
  try { await client.messages.create({ ...params, tools: [{ name: "shell", input_schema: { type: "array" } }] }); }
  catch (error: any) { if (error.status !== 400) throw error; toolRejected = true; }
  if (!toolRejected) throw new Error("Invalid tool schemas must be explicitly rejected");
  const result = { schema: "luban-claude-smoke-v1", local_date: "2026-10-26", source: "temporary candidate Bun adapter over unchanged installed public API",
    installed: health, sdk_version: sdkVersion, precision: installed.native_inference_precision,
    seed: "API does not expose a seed; temperature=0", split: "synthetic protocol smoke, no TRAIN/VALID/TEST experiment",
    memory_measurement: "not measured; no resource-performance claim", elapsed_seconds: (performance.now() - started) / 1000,
    requests: 3, teacher_requests: 0, adam_updates: 0, installed_files_modified: false,
    json: plain, sdk_stream_final: streamed, stream_text_events: deltas.length,
    openai: { message: openai.choices[0].message, usage: openai.usage }, invalid_tool_schema_rejected: toolRejected,
    all_three_outputs_equal: true, scope: "one short text prompt; not Claude model execution, tools, long-context or agent qualification" };
  const rawDirectory = mkdtempSync(join(resolve(rawParent), "claude-protocol-20261026-"));
  writeFileSync(join(rawDirectory, "result.json"), redact(JSON.stringify(result, null, 2)) + "\n");
  console.log(JSON.stringify({ raw_directory: rawDirectory, ...result }, null, 2));
} catch (error) { throw new Error(redact(String(error))); }
finally { backend.close(); server.stop(true); rmSync(directory, { recursive: true, force: true }); }
