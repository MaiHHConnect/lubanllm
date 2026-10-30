/** One application-owned round: BF16 training/staging, serial Q4 validation, then real publication. */
import assert from "node:assert/strict";
import { mkdir, realpath } from "node:fs/promises";
import { join } from "node:path";
import * as runtime from "./native-evolution-worker.ts";
import type { NativeEvolutionRuntimeInput } from "./native-evolution-worker.ts";
import type { NativeEvolutionSession } from "./native-evolution-service.ts";
import { NativeCore450Owner } from "../.native-migration/research/native-core450-owner-controller/owner.ts";
import { stageNativeEvolutionFinal, admitNativeEvolutionFinal, type NativeEvolutionFinalRequest } from "./native-evolution-coordinator.ts";
import { NativeQ4Observer } from "./native-evolution-q4.ts";
import { readNativeEvolutionGraphQualification } from "./native-evolution-graph-qualification.ts";
import { retainNativeEvolutionScratch, removeNativeEvolutionScratch } from "./native-evolution-segment-files.ts";
import { nativeEvolutionResumeMaterial } from "./native-evolution-resume.ts";
import { withNativeEvolutionMaterial, selectIndependentNativeEvolutionCohort } from "./native-evolution-cohort.ts";
import { withNativeEvolutionCalibration } from "./native-evolution-calibration.ts";
import { assertNativeEvolutionPending } from "./native-evolution-journal.ts";
import { NativeEvolutionQualityRejection } from "./native-evolution-q4-quality.ts";

export async function openNativeEvolutionSession(input: NativeEvolutionRuntimeInput): Promise<NativeEvolutionSession> {
  const resources = runtime.nativeEvolutionRuntimeResources();
  const graph = readNativeEvolutionGraphQualification({ identity: input.identity,
    graph_binary_sha256: resources.role("persistentExpert").sha256, training_binary_sha256: resources.role("trainer").sha256 });
  const q4 = await runtime.nativeEvolutionQ4Resources();
  let opened: Awaited<ReturnType<typeof runtime.openNativeEvolutionEngine>> | undefined;
  let preview: InstanceType<typeof runtime.Memory> | undefined = new runtime.Memory(input.root); preview.model_identity = input.identity;
  let observer: NativeQ4Observer | undefined, phaseOwner: NativeCore450Owner | undefined, phaseWork: Promise<unknown> | undefined;
  let running = false, used = false, closed = false, adopted = false;
  let scratch: string | undefined;
  let continuationMaterial = structuredClone(input.material ?? null);

  const closeEngine = async () => { if (opened) { await opened.close(); opened = undefined; } };
  const closeObserver = async () => { if (observer) { await observer.close(); observer = undefined; } };
  const closePhase = async () => {
    if (!phaseOwner) return;
    await phaseOwner.close(); await phaseWork?.then(() => {}, () => {});
    phaseOwner = undefined; phaseWork = undefined;
  };
  async function owned<T>(signal: AbortSignal, action: (owner: NativeCore450Owner) => Promise<T>): Promise<T> {
    assert(!phaseOwner); const owner = phaseOwner = new NativeCore450Owner();
    const cancel = () => owner.cancel(); signal.addEventListener("abort", cancel, { once: true });
    let done = false;
    const monitor = (async () => { while (!done) { if (owner.status().blocked) await owner.close(); await Bun.sleep(100); } })();
    void monitor.catch(() => {});
    const work = phaseWork = owner.run(action);
    let failure: unknown;
    try { return await Promise.race([work, monitor.then((): never => { throw new Error("采纳恢复监护提前结束"); })]); }
    catch (error) { failure = error; throw error; }
    finally {
      done = true; signal.removeEventListener("abort", cancel);
      // A failed owner close keeps the exact owner and work promise available to session.close().
      try { await closePhase(); await monitor; }
      catch (cleanup) { if (failure && cleanup !== failure) throw new AggregateError([failure, cleanup], "原生阶段及其清理均失败，保留原所有者"); throw cleanup; }
    }
  }
  return {
    continuationMaterial() { assert(adopted, "仅已采纳轮次可传递后续授权材料"); return continuationMaterial; },
    async close() {
      if (closed) return;
      await closePhase(); await closeObserver(); await closeEngine();
      if(preview){preview.close();preview=undefined;}
      if (adopted && scratch) { await removeNativeEvolutionScratch(scratch); scratch = undefined; }
      closed = true;
    },
    async runRound(options) {
      assert(!closed && !running && !used && preview, "每个原生session只能执行自己的一轮");
      options.signal.throwIfAborted(); running = true; used = true;
      let work = join(input.work, "round-" + options.round);
      let material = continuationMaterial;
      const { role, release } = resources, trainer = role("trainer"), rng = role("rng"), basis = role("replayBasis"), reader = role("reader");
      try {
        if (options.resume && material === null) {
          const row = preview!.connect(db => db.query("SELECT id,identity,kind,state,body FROM training_runs WHERE id=? AND identity=?").get(options.resume!, input.identity));
          material = nativeEvolutionResumeMaterial(row, input.identity);
        }
        continuationMaterial = material;
        await mkdir(work, { mode: 0o700 });
        work = await realpath(work); await retainNativeEvolutionScratch(work); scratch = work;
        options.onProgress({ step: 0, phase: "preparing" });
        assertNativeEvolutionPending(preview!,input.identity,options.resume);
        let calibration: Awaited<ReturnType<NativeQ4Observer['captureTraining']>> | undefined, calibrationRecords:unknown;
        if(!options.resume){
          const selected=await withNativeEvolutionMaterial(preview!,material,async()=>{
            const cohort=selectIndependentNativeEvolutionCohort(preview!,input.store,resources.base);
            return {records:cohort.records,core:cohort.core};
          });
          calibrationRecords=selected.records;preview!.close();preview=undefined;
          options.onProgress({step:0,phase:'q4_calibration'});observer=new NativeQ4Observer();
          calibration=await observer.captureTraining({...q4,model:input.model,manifestSHA:resources.base.model_manifest_sha256,identity:input.identity,records:selected.records.groups.replay_training,work,
            ...(selected.core?{core:{file:join(input.store,selected.core.version+'.safetensors'),tensor_key:resources.base.core_parameter,sha256:selected.core.sha256}}:{}),signal:options.signal});
          await closeObserver();options.signal.throwIfAborted();
        }
        if(preview){preview.close();preview=undefined;}
        const engineWork=join(work,'training-source');await mkdir(engineWork,{mode:0o700});
        opened=await runtime.openNativeEvolutionEngine({...input,work:engineWork});
        const preparationWork=join(work,'preparation');await mkdir(preparationWork,{mode:0o700});
        const request = { memory: opened.memory, work:preparationWork, prefix: { receipt: release.path, sha256: release.sha256 }, rng,
          basis: { binary: basis.path, sha256: basis.sha256 }, execution: { package_receipt: release.path,
            package_receipt_sha256: release.sha256, binary: trainer.path, binary_sha256: trainer.sha256, ...resources.training },
          outputParent: input.work, signal: options.signal, automatic: options.automatic, onProgress: options.onProgress };
        // The resume entry must restore the existing journal; never silently create a new run.
        const resume = Reflect.get(runtime, "runNativeEvolutionResume");
        if (options.resume) assert(typeof resume === "function", "原生完整checkpoint续跑入口尚未接通；原checkpoint保留");
        const result: Awaited<ReturnType<typeof runtime.runNativeEvolutionRound>> = options.resume
          ? await resume(opened.engine, request, options.resume, material)
          : await withNativeEvolutionCalibration(calibration!,calibrationRecords,()=>runtime.runNativeEvolutionRound(opened!.engine, request, material));
        assert(typeof result.step === "number" && Number.isSafeInteger(result.step) && result.step >= 0);
        options.onProgress({ step: result.step, phase: result.phase });
        const final = result.final_evidence;
        if (result.step !== 450 || !final || final.report.arithmetic_gates_passed !== true)
          return { ...result, step: result.step, pending: true, quality_checked: false, adopted: false };
        assert(result.previous_model_restored === true && result.rollback_scope_completed === true);
        const common = { store: input.store, base: resources.base, binary: reader.path, binary_sha256: reader.sha256,
          outputParent: input.work, expected: result.expected, final_checkpoint: result.checkpoint,
          material, signal: options.signal,
          producerResult: { ...result, previous_model_restored: true as const, rollback_scope_completed: true as const } };
        const stageDirectory = join(work, "publication-stage"); await mkdir(stageDirectory, { mode: 0o700 });
        options.onProgress({ step: 450, phase: "staging_candidate" });
        const stageRequest: NativeEvolutionFinalRequest = { ...common, memory: opened.memory, directory: stageDirectory };
        const staged = await owned(options.signal, owner => stageNativeEvolutionFinal(opened!.engine, stageRequest, owner));
        await closeEngine(); options.signal.throwIfAborted();
        options.onProgress({ step: 450, phase: "q4_baseline" });
        const observe = { ...q4, model: input.model, manifestSHA: staged.manifestSHA, identity: staged.identity,
          records: staged.records, work: input.work, signal: options.signal };
        observer = new NativeQ4Observer();
        const baseline = await observer.observePair({ ...observe, ...(staged.baselineCore ? { core: staged.baselineCore } : {}) });
        await closeObserver(); options.signal.throwIfAborted();
        options.onProgress({ step: 450, phase: "q4_candidate" }); observer = new NativeQ4Observer();
        const candidate = await observer.observePair({ ...observe, core: staged.core });
        await closeObserver(); options.signal.throwIfAborted();
        const adoptionDirectory = join(work, "publication-admit"); await mkdir(adoptionDirectory, { mode: 0o700 });
        opened = await runtime.openNativeEvolutionEngine({ ...input, work: adoptionDirectory });
        options.onProgress({ step: 450, phase: "adopting" });
        let committed: Awaited<ReturnType<typeof admitNativeEvolutionFinal>>;
        try {
          committed = await owned(options.signal, owner => admitNativeEvolutionFinal(opened!.engine, {
            ...common, memory: opened!.memory, directory: adoptionDirectory, staged, graph, q4: { baseline, candidate },
            messages: [{ role: "user", content: "你好" }] }, owner));
        } catch (error) {
          if (!(error instanceof NativeEvolutionQualityRejection)) throw error;
          return { ...result, step: 450, phase: "rejected_q4", quality_checked: true, adopted: false, pending: true,
            report: error.report, rejection_reason: error.message };
        }
        assert(committed.state === "committed" && committed.actual_reload_confirmed === true);
        adopted = true;
        const pending = committed.report.learning_progress.pending_ids.length > 0;
        const pendingIDs = new Set(committed.report.learning_progress.pending_ids);
        const waitingMaterial = pending && !committed.report.split_ids.new_training.some((id: string) => pendingIDs.has(id));
        return { ...result, ...committed, step: 450, phase: waitingMaterial ? "committed_waiting_material" : "committed",
          quality_checked: true, adopted: true, pending,
          ...(waitingMaterial ? { continuation_reason: "训练材料已掌握，独立验证仍有待学项；等待新增或改进训练材料，不把验证题改作训练。" } : {}) };
      } finally { running = false; }
    },
  };
}
