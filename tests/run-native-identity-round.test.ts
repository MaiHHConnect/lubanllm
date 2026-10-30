import { expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { readIdentityMaterial, parseComputeConnection, runIdentityRound } from "../scripts/run-native-identity-round.ts";
import {expandIdentityMaterial} from '../scripts/expand-native-identity-material.ts';

const authorized = "/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/authorized-identity-material.json";
test('authored expansion adds only authorized model facts and preserves all original rows and splits',async()=>{
 const original=await readIdentityMaterial(authorized),expanded=expandIdentityMaterial(original),dir=await mkdtemp(join(tmpdir(),'identity-expanded-'));
 try{
  expect(expanded.samples.slice(0,50)).toEqual(original.samples);
  expect(expanded.retentionRegressions).toEqual(original.retentionRegressions);
  for(const[fact,value]of Object.entries(original.directedTeaching.facts)as [string,any][]){
   expect(expanded.directedTeaching.facts[fact].validation_ids).toEqual(value.validation_ids);
   expect(expanded.directedTeaching.facts[fact].training_ids.length).toBe(value.training_ids.length+4);
  }
  expect(expanded.samples.slice(50).every(r=>r.purpose==='training'&&r.training_allowed&&r.provenance.evolution_role==='new'&&r.answer===expanded.directedTeaching.facts[r.provenance.teaching_fact].value)).toBe(true);
  const path=join(dir,'material.json');await writeFile(path,JSON.stringify(expanded));expect((await readIdentityMaterial(path)).samples.length).toBe(62);
  expect(original.samples.length).toBe(50);
 }finally{await rm(dir,{recursive:true,force:true});}
});
test("the explicit authorized material preserves all three read-only user regressions", async () => {
  const material = await readIdentityMaterial(authorized), temp = await mkdtemp(join(tmpdir(), "identity-material-"));
  try {
    expect(material.samples).toHaveLength(50); expect(material.retentionRegressions).toHaveLength(3);
    expect(material.retentionRegressions!.every(row => row.training_allowed === false && row.purpose === "validation")).toBe(true);
    const path = join(temp, "bad.json"); material.directedTeaching.facts.model_name.value = "林浩";
    await writeFile(path, JSON.stringify(material)); await expect(readIdentityMaterial(path)).rejects.toThrow();
  } finally { await rm(temp, { recursive: true }); }
});

test("private connection must belong to the actual same-user application child", () => {
  const launch = { pid: 456, compute_pid: 789 }, token = "x".repeat(48), uid = process.getuid!();
  const line = `456 ${uid} /private/runtime/python -m luban.server --port 54321 --home /tmp/home`;
  const environment = `ENV=test LUBAN_COMPUTE_TOKEN=${token} OTHER=not-copied`;
  expect(parseComputeConnection(launch, line, environment)).toEqual({ url: "http://127.0.0.1:54321", token });
  expect(() => parseComputeConnection(launch, line.replace(/^456/, "999"), environment)).toThrow();
  expect(() => parseComputeConnection(launch, line.replace(` ${uid} `, ` ${uid + 1} `), environment)).toThrow();
  expect(() => parseComputeConnection(launch, line.replace("luban.server", "unrelated"), environment)).toThrow();
  expect(() => parseComputeConnection(launch, line, "LUBAN_COMPUTE_TOKEN=")).toThrow();
});

test("research invocation uses the product lease/session, logs real progress, preserves pending and resumes without a new budget", async () => {
  const root = await mkdtemp(join(tmpdir(), "identity-round-")), material = await readIdentityMaterial(authorized);
  try {
    for (const resumed of [false, true]) {
      const calls: string[] = []; let leaseID: unknown, opens = 0, closes = 0;
      const output = join(root, resumed ? "resume" : "fresh");
      const result = await runIdentityRound({ home: root, output, material, ...(resumed ? { resume: "abcdef123456" } : {}),
        async compute(path, body) {
          calls.push(path);
          if (path.endsWith("/acquire")) { leaseID = body.lease_id; return Response.json({ lease_id: leaseID, was_ready: true,
            identity: "a".repeat(64), root, store: root, model: root, activeCore: null }); }
          if (path.endsWith("/release")) { expect(body.lease_id).toBe(leaseID); expect(closes).toBe(1); return Response.json({ released: true, was_ready: true }); }
          return Response.json({ state: "loading" });
        }, worker: { async openNativeEvolutionRuntime(input) {
          opens++; expect(input.material).toEqual(material); expect(calls).toEqual(["/internal/native-evolution/acquire"]);
          return { async runRound(options) {
            expect(options.resume).toBe(resumed ? "abcdef123456" : undefined); expect(options.round).toBe(1);
            options.onProgress({ step: resumed ? 225 : 0, phase: "training" }); options.onProgress({ step: 450, phase: "awaiting_quality" });
            return { step: 450, phase: "committed", quality_checked: true, adopted: true, pending: true };
          }, async close() { closes++; calls.push("native-close"); } };
        } } });
      expect(opens).toBe(1); expect(result.status.state).toBe("complete"); expect(result.status.last_result.pending).toBe(true);
      expect(result.fixed_steps).toBe(450); expect(result.resume).toBe(resumed ? "abcdef123456" : null);
      expect(result.status.lease_held).toBe(false); expect(result.total_seconds).toBeGreaterThan(0);
      expect(calls).toEqual(["/internal/native-evolution/acquire", "native-close", "/internal/native-evolution/release", "/api/model/reload"]);
      const log = (await readFile(join(output, "progress.jsonl"), "utf8")).trim().split("\n").map(line => JSON.parse(line));
      expect(log.filter(row => row.phase === "awaiting_quality").map(row => row.step)).toEqual([450]);
      expect(log.at(-1).phase).toBe("finished");
      expect(JSON.parse(await readFile(join(output, "result.json"), "utf8")).material_counts.readonly_user_regressions).toBe(3);
    }
  } finally { await rm(root, { recursive: true }); }
});

test("a requested durable pause aborts before the next segment and closes owners before releasing the lease", async () => {
  const root = await mkdtemp(join(tmpdir(), "identity-pause-")), material = await readIdentityMaterial(authorized);
  const calls: string[] = []; let updates = 0;
  try {
    const result = await runIdentityRound({ home: root, output: join(root, "pause"), material, pauseAtStep: 16,
      async compute(path, body) {
        calls.push(path);
        return Response.json(path.endsWith("/acquire") ? { lease_id: body.lease_id, was_ready: false,
          identity: "a".repeat(64), root, store: root, model: root, activeCore: null } : { released: true, was_ready: false });
      }, worker: { async openNativeEvolutionRuntime() {
        return { async runRound(options) {
          updates = 16; options.onProgress({ step: updates, phase: "training" });
          expect(options.signal.aborted).toBe(true); options.signal.throwIfAborted();
          updates += 16; throw new Error("The second segment must not start");
        }, async close() { calls.push("native-close"); } };
      } } });
    expect(updates).toBe(16); expect(result.status.step).toBe(16); expect(result.status.state).toBe("paused");
    expect(result.fixed_steps).toBe(450); expect(result.requested_pause_at_step).toBe(16);
    expect(result.status.lease_held).toBe(false);
    expect(calls).toEqual(["/internal/native-evolution/acquire", "native-close", "/internal/native-evolution/release"]);
  } finally { await rm(root, { recursive: true }); }
});
