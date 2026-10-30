import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, mkdir, readFile, writeFile, symlink, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { NativeQ4Observer, assertNativeQ4Observation } from "../backend/native-evolution-q4.ts";

// Real Swift tokenizer/observer plus a protocol fixture, never a model quality result.
const modelSource = "/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1";
let work: string, binary: string, executable: string, manifestSHA: string;
const rows = [{ id: "q4-protocol", answer: "K", prompt: "Reply K", provenance: { generation_check: { type: "exact", expected: "K", max_new_tokens: 8 } } }];
const sha = (value: Uint8Array) => createHash("sha256").update(value).digest("hex");
beforeAll(async () => {
  work = await mkdtemp(join(tmpdir(), "luban-q4-observation-test-")); binary = join(work, "fixture-worker");
  const source = (await readFile(join(import.meta.dir, "native_worker_fixture.swift"), "utf8"))
    .replace("for token in [42, 43]", "for token in (mode == \"no-eos\" ? [42, 43] : [42, 248044])")
    .replace("mode == \"wrong-final\" ? [42, 44] : [42, 43]", "mode == \"no-eos\" ? [42, 43] : [42, 248044]")
    .replace("var cancelled = false", "FileManager.default.createFile(atPath: args[1] + \"/request-started\", contents: Data()); var cancelled = false");
  await writeFile(join(work, "Fixture.swift"), source);
  const build = Bun.spawn(["/usr/bin/swiftc", "-O", "-parse-as-library", join(work, "Fixture.swift"), "-o", binary], { stdout: "pipe", stderr: "pipe" });
  expect(await build.exited, await new Response(build.stderr).text()).toBe(0);
  executable=join(work,"observer");
  const observerBuild=Bun.spawn(["/usr/bin/swiftc","native/EvolutionObserve.swift","native/Worker.swift","native/Tokenizer.swift","-o",executable],{stdout:"pipe",stderr:"pipe"});
  expect(await observerBuild.exited,await new Response(observerBuild.stderr).text()).toBe(0);
  await writeFile(binary + ".json", JSON.stringify({ protocol: "luban-worker-jsonl-v1", binary_sha256: sha(await readFile(binary)) }));
  manifestSHA = sha(await readFile(join(modelSource, "upstream-file-manifest.json")));
  for (const mode of ["normal", "no-eos", "cancel", "expert-normal", "expert-wrong-width"]) {
    const model = join(work, mode); await mkdir(model);
    for (const file of ["upstream-file-manifest.json", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
      await symlink(join(modelSource, file), join(model, file));
  }
}, 30_000);
afterAll(async () => { if (work) await rm(work, { recursive: true, force: true }); });
const input = (mode = "normal") => ({ executable,
  binary, model: join(work, mode), manifestSHA, identity: "b".repeat(64), records: rows, work });

test("real native tokenizer and protocol produce sealed complete Q4 observations, not caller passed flags", async () => {
  const observer = new NativeQ4Observer();
  const [result, second] = await observer.observePair(input());
  expect(result.precision).toBe("mixed-q4-g16"); expect(result.rows[0]!.tokens).toEqual([42, 248044]);
  expect(result.rows[0]!.answer).toBe("K"); expect(result.rows[0]!.passed).toBe(true); expect(result.rows[0]!.complete).toBe(true);
  assertNativeQ4Observation(result, rows);
  assertNativeQ4Observation(second, rows); expect(second).not.toBe(result); expect(second.rows).toEqual(result.rows);
  expect(() => assertNativeQ4Observation({ ...result }, rows)).toThrow("本进程");
  expect(() => assertNativeQ4Observation(result, [{ ...rows[0], prompt: "changed" }])).toThrow("材料一致");
  expect(Object.isFrozen(result.rows[0]!.tokens)).toBe(true);
  await observer.close();
}, 15_000);

test("a stop event without real EOS cannot certify an answer; observer closes before reuse", async () => {
  const observer = new NativeQ4Observer();
  await expect(observer.observe(input("no-eos"))).rejects.toThrow("缺少EOS");
  await observer.close(); expect((await observer.observe(input())).rows[0]!.passed).toBe(true);
}, 15_000);

test("cancellation during startup is retained until the native handler is ready", async () => {
  const observer = new NativeQ4Observer(), controller = new AbortController();
  const task = observer.observe({ ...input("cancel"), signal: controller.signal });
  const timer = setTimeout(() => controller.abort(new Error("fixture cancellation")), 600);
  try { await expect(task).rejects.toThrow("fixture cancellation"); } finally { clearTimeout(timer); await observer.close(); }
  expect((await observer.observe(input())).rows[0]!.complete).toBe(true);
}, 15_000);

test("cancellation during native generation closes and reaps before the model can be reused", async () => {
  await rm(join(work, "cancel/request-started"), { force: true });
  const observer = new NativeQ4Observer(), controller = new AbortController();
  const task = observer.observe({ ...input("cancel"), signal: controller.signal });
  const timer = setInterval(() => { if (existsSync(join(work, "cancel/request-started"))) controller.abort(new Error("active cancellation")); }, 10);
  try { await expect(task).rejects.toThrow("active cancellation"); } finally { clearInterval(timer); await observer.close(); }
  expect((await observer.observe(input())).rows[0]!.complete).toBe(true);
}, 15_000);


test("actual observer binds expert SHA/core/binary and keeps expert evidence out of core-only guards", async()=>{
  const coreFile=join(work,"core.safetensors"),expertFile=join(work,"expert.safetensors");
  await writeFile(coreFile,"fixture core");await writeFile(expertFile,"fixture expert");
  const core={file:coreFile,tensor_key:"down.weight",sha256:sha(await readFile(coreFile))};
  const expert={file:expertFile,sha256:sha(await readFile(expertFile)),width:128,layer:23 as const,core_version:"accepted-fixture-core"};
  const owner=new NativeQ4Observer();
  const [first,second]=await owner.observePair({...input("expert-normal"),core,expert});
  expect(first.rows[0]!.branch_enabled).toBe(true);expect(first.expert).toEqual({...expert,core_file_sha256:core.sha256});
  expect(first.worker_binary_sha256).toBe(sha(await readFile(binary)));
  assertNativeQ4Observation(first,rows,expert);assertNativeQ4Observation(second,rows,expert);
  expect(()=>assertNativeQ4Observation(first,rows)).toThrow("专家证据范围");
  expect(()=>assertNativeQ4Observation(first,rows,{...expert,core_version:"other"})).toThrow("专家证据范围");
  await expect(owner.captureTraining({...input(),core,expert})).rejects.toThrow("cannot include an expert");
  await expect(owner.observe({...input("expert-wrong-width"),core,expert})).rejects.toThrow();await owner.close();
  expect((await owner.observe(input())).rows[0]!.branch_enabled).toBe(false);
},15000);
