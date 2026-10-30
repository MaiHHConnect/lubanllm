import { resolve } from "node:path";
const binary = process.argv[2] ?? resolve(import.meta.dir, "../../build/native/luban-mlx-worker");
const valid = { id: "cpu", tokens: [1, 2], max_tokens: 3, eos_ids: [248044], temperature: 0.6, seed: 42, cache_reuse: true };
let checked = 0;
async function probe(value: unknown, ok: boolean) {
  const child = Bun.spawn([binary, "--validate-request"], { stdin: "pipe", stdout: "pipe", stderr: "pipe" });
  child.stdin.write(JSON.stringify(value) + "\n"); child.stdin.end();
  const output = await new Response(child.stdout).text();
  await new Response(child.stderr).text();
  const status = await child.exited;
  if ((status === 0) !== ok || (ok && !output.includes('"event":"validated"'))) throw new Error(`Protocol CPU check failed: ${status} ${output}`);
  checked++;
}
await probe(valid, true);
for (const prefill_step of [32, 64, 128, 256, 512]) await probe({ ...valid, prefill_step }, true);
for (const prefill_step of [0, 16, 48, 1024, true, "128", 128.5]) await probe({ ...valid, prefill_step }, false);
await probe({ ...valid, tokens: [] }, false);
await probe({ ...valid, tokens: [248320] }, false);
await probe({ ...valid, eos_ids: [] }, false);
await probe({ ...valid, max_tokens: 2049 }, false);
await probe({ ...valid, tokens: Array(32768).fill(1) }, false);
await probe({ ...valid, context_limit: 65536, tokens: Array(32768).fill(1) }, true);
for (const limit of [65536, 131072, 262144]) {
  await probe({ ...valid, context_limit: limit, tokens: Array(limit - valid.max_tokens).fill(1) }, true);
  await probe({ ...valid, context_limit: limit, tokens: Array(limit - valid.max_tokens + 1).fill(1) }, false);
}
for (const context_limit of [0, -1, 262145, true, 65536.5, "65536"]) {
  await probe({ ...valid, context_limit }, false);
}
await probe({ ...valid, temperature: -1 }, false);
await probe({ ...valid, temperature: 3 }, false);
await probe({ ...valid, seed: -1 }, false);
await probe({ ...valid, cache_reuse: "true" }, false);
console.log(JSON.stringify({ schema: "native-mlx-worker-cpu-v1", checks: checked, gpu_used: false, passed: true }));
