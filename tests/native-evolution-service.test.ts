import type { NativeEvolutionMaterial } from "../backend/native-evolution-cohort.ts";
import { expect, test } from "bun:test";
import { NativeEvolutionService, type NativeEvolutionSession, type NativeEvolutionRoundResult } from "../backend/native-evolution-service.ts";

const completed = (pending = false): NativeEvolutionRoundResult => ({ step: 450, phase: "committed", quality_checked: true, adopted: true, pending });
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done; }); return { promise, resolve }; }
function setup(open: (log: string[], material: NativeEvolutionMaterial | null) => Promise<NativeEvolutionSession>, acquireLost = false) {
  const log: string[] = [], leases: string[] = [];
  const service = new NativeEvolutionService({
    async compute(path, body) {
      log.push(path);
      if (path.endsWith("/acquire")) {
        leases.push(body.lease_id as string);
        if (acquireLost) throw Error("响应丢失");
        return Response.json({ lease_id: body.lease_id, was_ready: true, identity: "a".repeat(64), root: "/data", store: "/data/modules/luban0-v1", model: "/models/luban0", activeCore: null });
      }
      if (path.endsWith("/release")) { expect(body.lease_id).toBe(leases.at(-1)); return Response.json({ released: true, was_ready: true }); }
      return Response.json({ state: "loading" });
    },
    async open(_lease, _signal, material) { log.push("open"); return open(log, material); },
  });
  return { service, log, leases };
}

test("concurrent starts reserve one hand-off and restore inference only after native close", async () => {
  const { service, log } = setup(async log => ({ async runRound() { log.push("round"); return completed(); }, async close() { log.push("close"); } }));
  service.start(); expect(() => service.start()).toThrow("正在运行");
  const result = await service.wait();
  expect(log).toEqual(["/internal/native-evolution/acquire", "open", "round", "close", "/internal/native-evolution/release", "/api/model/reload"]);
  expect(result.state).toBe("complete"); expect(result.lease_held).toBe(false);
});

test("continuous evolution opens a new bound session only after the preceding round closes", async () => {
  let rounds = 0;
  const { service, log } = setup(async log => ({ async runRound(input) { rounds++; expect(input.round).toBe(rounds); log.push("round" + rounds); return completed(rounds === 1); }, async close() { log.push("close" + rounds); } }));
  service.start(); expect((await service.wait()).round).toBe(2);
  expect(log.slice(1, 7)).toEqual(["open", "round1", "close1", "open", "round2", "close2"]);
});

test("awaiting quality is not accepted and cannot silently start a successor", async () => {
  const { service, log } = setup(async () => ({ async runRound() { return { ...completed(true), phase: "awaiting_quality", quality_checked: false, adopted: false }; }, async close() {} }));
  service.start(); const status = await service.wait();
  expect(status.state).toBe("unaccepted"); expect(status.last_result.adopted).toBe(false); expect(log.filter(x => x === "open")).toHaveLength(1);
});

test("quality rejection preserves pending material without an immediate failed-round loop", async () => {
  const { service, log } = setup(async () => ({ async runRound() { return { ...completed(true), phase: "rejected", adopted: false }; }, async close() {} }));
  service.start(); const status = await service.wait();
  expect(status.state).toBe("unaccepted"); expect(status.last_result.pending).toBe(true); expect(status.last_result.adopted).toBe(false);
  expect(log.filter(x => x === "open")).toHaveLength(1);
});

test("accepted progress with validation still pending waits for trainable material without clearing pending", async () => {
  const continuation_reason = "训练材料已掌握，等待新增或改进训练材料；验证题不参与训练。";
  const { service, log } = setup(async () => ({ async runRound() {
    return { ...completed(true), phase: "committed_waiting_material", continuation_reason };
  }, async close() { log.push("closed"); } }));
  service.start(); const status = await service.wait();
  expect(status.state).toBe("waiting_material"); expect(status.round).toBe(1);
  expect(status.last_result).toEqual({ ...completed(true), phase: "committed_waiting_material", continuation_reason });
  expect(status.lease_held).toBe(false); expect(log.filter(x => x === "open")).toHaveLength(1);
  expect(log.slice(-3)).toEqual(["closed", "/internal/native-evolution/release", "/api/model/reload"]);
});

test("waiting for material cannot bypass adoption and pending-result checks", async () => {
  for (const invalid of [{ adopted: false }, { pending: false }, { continuation_reason: "" }, { step: 449 }]) {
    const { service } = setup(async () => ({ async runRound() {
      return { ...completed(true), phase: "committed_waiting_material", continuation_reason: "等待新材料", ...invalid };
    }, async close() {} }));
    service.start(); expect((await service.wait()).state).toBe("failed");
  }
});

test("cancellation retains compute exclusion through checkpoint and native teardown", async () => {
  const entered = deferred<void>(), released = deferred<NativeEvolutionRoundResult>(), closed = deferred<void>();
  let signal!: AbortSignal;
  const { service, log } = setup(async () => ({ async runRound(input) { signal = input.signal; entered.resolve(); return released.promise; }, async close() { await closed.promise; } }));
  service.start(); await entered.promise; service.cancel();
  expect(signal.aborted).toBe(true); expect(service.status().lease_held).toBe(true);
  released.resolve({ step: 16, phase: "paused_after_cancel", quality_checked: false, adopted: false, pending: true });
  await Promise.resolve(); expect(log).not.toContain("/internal/native-evolution/release");
  closed.resolve(); expect((await service.wait()).state).toBe("paused");
});

test("failed cleanup keeps the same lease and session until explicit close succeeds", async () => {
  let fail = true, closes = 0;
  const { service, log } = setup(async () => ({ async runRound() { return completed(); }, async close() { closes++; if (fail) throw Error("native owner alive"); } }));
  service.start(); expect((await service.wait()).state).toBe("quarantined");
  expect(service.status().lease_held).toBe(true); expect(log).not.toContain("/internal/native-evolution/release");
  expect(() => service.start()).toThrow("清理");
  fail = false; await service.close(); expect(closes).toBe(3);
  expect(service.status().lease_held).toBe(false); expect(log).not.toContain("/api/model/reload");
});

test("lost acquire response releases the client-owned id and restores the previously loaded model", async () => {
  const { service, log, leases } = setup(async () => { throw Error("must not open"); }, true);
  service.start(); expect((await service.wait()).state).toBe("failed");
  expect(leases[0]).toMatch(/^[a-f0-9-]{36}$/);
  expect(log).toEqual(["/internal/native-evolution/acquire", "/internal/native-evolution/release", "/api/model/reload"]);
});

test("resume id reaches only the first round and fixed progress cannot claim premature adoption", async () => {
  const resume = "abcdef123456", calls: (string | undefined)[] = [];
  const { service } = setup(async () => ({ async runRound(input) { calls.push(input.resume); input.onProgress({ step: 16, phase: "training" }); return calls.length === 1 ? completed(true) : { ...completed(), step: 32 }; }, async close() {} }));
  service.start({ resume }); const result = await service.wait();
  expect(calls).toEqual([resume, undefined]); expect(result.state).toBe("failed"); expect(result.error).toContain("质量验收");
});


test("only committed pending rounds pass a private cloned authorization to the next session, including resume", async () => {
  for (const resumed of [false, true]) {
    let rounds = 0, reads = 0;
    const authorized: NativeEvolutionMaterial = { samples: [{ id: "authorized-train", purpose: "training", answer: "luban0" }, { id: "independent-valid", purpose: "validation", answer: "麦好火" }], directedTeaching: { authorization: "original-human" }, retentionRegressions: [{ id: "unknown-user" }] };
    const { service } = setup(async (_log, material) => {
      rounds++;
      if (rounds === 1) expect(material).toBe(null);
      else { expect(material).toEqual({ ...authorized, samples: [{ ...authorized.samples[0], answer: "luban0" }, authorized.samples[1]] }); expect(material).not.toBe(authorized); }
      return { async runRound(input) { expect(input.resume).toBe(resumed && rounds === 1 ? "abcdef123456" : undefined); return completed(rounds === 1); },
        continuationMaterial() { reads++; return authorized; }, async close() { if (rounds === 1) authorized.samples[0]!.answer = "mutated-after-close"; } };
    });
    service.start(resumed ? { resume: "abcdef123456" } : {}); const status = await service.wait();
    expect(status.state).toBe("complete"); expect(rounds).toBe(2); expect(reads).toBe(1);
    expect(JSON.stringify(status)).not.toContain("authorized-train");
  }
});

test("unaccepted and waiting-material rounds never read or pass continuation authorization", async () => {
  for (const result of [{ ...completed(true), adopted: false, phase: "rejected_q4" }, { ...completed(true), phase: "committed_waiting_material", continuation_reason: "等待" }]) {
    let reads = 0;
    const { service } = setup(async () => ({ async runRound() { return result; }, continuationMaterial() { reads++; throw Error("must not read"); }, async close() {} }));
    service.start(); await service.wait(); expect(reads).toBe(0);
  }
});

test("expert sessions report cumulative budget and stop after verified early acceptance", async () => {
  let rounds = 0;
  const { service, log } = setup(async log => ({ stepBudget: 192, kind: "learning", earlyStop: true,
    async runRound({ onProgress }) { rounds++; onProgress({ step: 8, phase: "validating" });
      return { ...completed(rounds === 1), step: 8 }; }, async close() { log.push("expert-close"); } }));
  expect(service.start({ mode: "learning" })).toMatchObject({ kind: "learning", steps: 0, phase: "acquiring_model" }); const state = await service.wait();
  expect(state).toMatchObject({ state: "complete", kind: "learning", steps: 192, step: 8, round: 2, lease_held: false });
  expect(log.filter(x => x === "expert-close")).toHaveLength(2);
});

test("expert early stop never bypasses quality or accepts reset/over-budget updates", async () => {
  for (const scenario of ["quality", "reset", "over-budget"] as const) {
    const { service } = setup(async () => ({ stepBudget: 192, kind: "learning", earlyStop: true,
      async runRound({ onProgress }) { onProgress({ step: 72, phase: "candidate_2" });
        return { ...completed(), step: scenario === "reset" ? 8 : scenario === "over-budget" ? 193 : 72,
          quality_checked: scenario !== "quality" }; }, async close() {} }));
    service.start(); expect((await service.wait()).state).toBe("failed");
    expect(service.status().lease_held).toBe(false);
  }
});
