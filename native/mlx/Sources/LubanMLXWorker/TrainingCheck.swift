import Foundation
import MLX
import MLXNN
import MLXLMCommon

private final class FixtureLayer: Module {
    @ModuleInfo var mlp: Module
    init(_ base: Module) { _mlp.wrappedValue = base }
}
private final class FixtureInner: Module {
    @ModuleInfo var layers: [FixtureLayer]
    init(_ layers: [FixtureLayer]) { _layers.wrappedValue = layers }
}
private final class FixtureLanguage: Module {
    @ModuleInfo var model: FixtureInner
    init(_ inner: FixtureInner) { _model.wrappedValue = inner }
}
private final class TrainingFixture: Module, LanguageModel {
    @ModuleInfo(key:"language_model") var language: FixtureLanguage
    let embedding: MLXArray
    let head: MLXArray
    var forwardCalls = 0
    override init() {
        let matrix = MLXArray((0..<64).map { Float($0%11-5)/32 },[8,8]).asType(.bfloat16)
        let layers = (0..<24).map { _ in FixtureLayer(Linear(weight:matrix)) }
        _language.wrappedValue = FixtureLanguage(FixtureInner(layers))
        embedding = MLXArray((0..<128).map { Float($0%17-8)/16 },[16,8]).asType(.bfloat16)
        head = MLXArray((0..<128).map { Float($0%23-11)/16 },[16,8]).asType(.bfloat16)
    }
    var decodeIsCompiled: Bool { compiled.isCompiled }
    private let compiled = CompiledTrace<TrainingFixture> { model,args in [model.forward(args[0])] }
    func callAsFunction(_ inputs: MLXArray,cache:[KVCache]?) -> MLXArray {
        inputs.dim(1) == 1 ? compiled(self,inputs) : forward(inputs)
    }
    private func forward(_ inputs: MLXArray) -> MLXArray {
        forwardCalls += 1
        let x = take(embedding,inputs,axis:0)
        let h = ((language.model.layers[23].mlp as! any UnaryLayer)(x) + x).asType(.bfloat16)
        return matmul(h,head.T).asType(.bfloat16)
    }
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }
    func prepare(_ input: LMInput,cache:[KVCache],state:LMOutput.State?,prefill:PrefillParameters) throws -> PrepareResult { .tokens(input.text) }
}

func checkTraining() throws {
    try Device.withDefaultDevice(.cpu) {
        let options = try Options(["worker","/unused",String(repeating:"a",count:64),"0","--mlx-q4","--mlx-model=/unused","--mlx-model-sha256="+String(repeating:"b",count:64)])
        let model = TrainingFixture()
        let trainer = try ExpertTraining(model:model,options:options,hidden:8,vocabulary:16)
        func command(_ fields: [String:Any]) throws -> TrainingCommand {
            var value=fields; value["id"]="cpu-check"
            return try JSONDecoder().decode(TrainingCommand.self,from:JSONSerialization.data(withJSONObject:value))
        }
        let sample: [String:Any] = ["tokens":[1,2,3,4,5],"labels":[-100,-100,3,4,5]]
        let protect: [String:Any] = ["tokens":[6,7,8,9],"labels":[-100,-100,8,9]]
        _ = try trainer.execute(command(["op":"train_observe","sample":sample,"source":"core"]))
        // Compile with the old tree first, then replace its MLP as Qwen decode does.
        let one = MLXArray([1],[1,1])
        let compiledCore = model(one,cache:nil); eval(compiledCore)
        try require(model.decodeIsCompiled, "Core trace was not compiled")
        for _ in 0..<3 { try trainer.select("core") }
        try require(model.decodeIsCompiled, "Repeated same-source selection discarded core trace")
        _ = try trainer.execute(command(["op":"train_begin","width":32,"seed":42,"initialize":"fresh","binding":"fixture-v1"]))
        let compiledInitial = model(one,cache:nil); eval(compiledInitial)
        let residentCandidate = trainer.adapter
        for _ in 0..<3 { try trainer.select("candidate") }
        try require(trainer.adapter === residentCandidate && model.decodeIsCompiled, "Repeated candidate selection replaced its module/trace")
        try require(all(compiledCore .== compiledInitial).item(Bool.self), "Zero-down candidate changed compiled core output")
        let initial = trainer.weights
        let firstResult = try trainer.execute(command(["op":"train_step","sample":sample,"preservation":protect,"teacher":"core"]))
        let compiledUpdated = model(one,cache:nil); eval(compiledUpdated)
        let uncompiledUpdated = model(MLXArray([1,1],[1,2]),cache:nil)[0,0]
        try require(all(compiledUpdated[0,0] .== uncompiledUpdated).item(Bool.self), "Compiled and current parameter forward differ")
        try require(trainer.step == 1 && (firstResult["gradient_norm"] as? Float ?? 0) > 0, "CPU gradient/update missing")
        try require(any(trainer.weights[2] .!= initial[2]).item(Bool.self), "Adam did not update expert down projection")
        // Independent scalar Adam reference from realized first-step moments; no MLX
        // update expression is reused. BF16 rounding is checked against actual export.
        var maximumAdamError: Float = 0
        for i in 0..<3 {
            let before=initial[i].asType(.float32).asArray(Float.self)
            let m=trainer.first[i].asArray(Float.self), v=trainer.second[i].asArray(Float.self)
            let expected=before.indices.map { j in Float(Double(before[j]) - 0.0002 * (Double(m[j])/0.1) / (sqrt(Double(v[j])/0.001)+0.000001)) }
            let rounded=MLXArray(expected,initial[i].shape).asType(.bfloat16).asType(.float32)
            maximumAdamError=Swift.max(maximumAdamError,max(abs(rounded-trainer.weights[i].asType(.float32))).item(Float.self))
        }
        try require(maximumAdamError <= 0.000001, "Independent scalar Adam reference differs")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("luban-mlx-training-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:root) }
        let path=root.appendingPathComponent("expert.safetensors").path, checkpoint=root.appendingPathComponent("checkpoint.safetensors").path
        _ = try trainer.execute(command(["op":"train_save","path":path,"checkpoint_path":checkpoint]))
        let expectedKeys=Set(expertKeys)
        try require(Set(try MLX.loadArrays(url:URL(fileURLWithPath:path)).keys) == expectedKeys, "Export is not standard expert safetensors")
        _ = try trainer.execute(command(["op":"train_step","sample":sample,"preservation":protect,"teacher":"core"]))
        let afterTwo=trainer.weights+trainer.first+trainer.second+[trainer.rng]
        eval(afterTwo)
        _ = try trainer.execute(command(["op":"train_restore","checkpoint_path":checkpoint,"binding":"fixture-v1"]))
        try require(trainer.step == 1, "Checkpoint ordinal not restored")
        try require(all(model(one,cache:nil) .== compiledUpdated).item(Bool.self), "Restored candidate compiled output differs")
        _ = try trainer.execute(command(["op":"train_step","sample":sample,"preservation":protect,"teacher":"core"]))
        let repeated=trainer.weights+trainer.first+trainer.second+[trainer.rng]
        try require(zip(afterTwo,repeated).allSatisfy { all($0 .== $1).item(Bool.self) }, "Restored Adam/RNG repeat differed")
        var rejected=0
        do { _ = try trainer.execute(command(["op":"train_restore","checkpoint_path":checkpoint,"binding":"wrong"])) } catch { rejected += 1 }
        do { _ = try trainer.execute(command(["op":"train_step","sample":["tokens":[1,2],"labels":[-100,-100]],"preservation":protect])) } catch { rejected += 1 }
        do { _ = try trainer.execute(command(["op":"train_step","sample":sample,"preservation":protect,"teacher":"parent"])) } catch { rejected += 1 }
        do { _ = try trainer.execute(command(["op":"train_save","path":path,"checkpoint_path":checkpoint])) } catch { rejected += 1 }
        let (arrays,metadata) = try MLX.loadArraysAndMetadata(url:URL(fileURLWithPath:checkpoint))
        var corrupted=arrays; corrupted["adam.m.0"]=MLXArray.full(trainer.first[0].shape,values:MLXArray(Float.nan))
        let bad=root.appendingPathComponent("bad.safetensors").path
        try MLX.save(arrays:corrupted,metadata:metadata,url:URL(fileURLWithPath:bad))
        do { _ = try trainer.execute(command(["op":"train_restore","checkpoint_path":bad,"binding":"fixture-v1"])) } catch { rejected += 1 }
        try require(rejected == 5 && trainer.step == 2 && zip(afterTwo,trainer.weights+trainer.first+trainer.second+[trainer.rng]).allSatisfy { all($0 .== $1).item(Bool.self) }, "Failed operation changed state")
        _ = try trainer.execute(command(["op":"train_end","restore_parent":true]))
        try require(!trainer.begun && trainer.weights.isEmpty && trainer.adapter == nil, "Training end failed to restore core")
        // A first learning round has no accepted old material. Exercise actual pure
        // CE differentiation/Adam without inventing a replay or teacher forward.
        _ = try trainer.execute(command(["op":"train_begin","width":32,"seed":42,"initialize":"fresh","binding":"first-learning-fixture"]))
        let beforeCE = trainer.weights
        let callsBeforeCE = model.forwardCalls
        let ceResult = try trainer.execute(command(["op":"train_step","sample":sample]))
        try require(model.forwardCalls == callsBeforeCE + 1, "CE-only update evaluated a teacher or replay")
        try require(ceResult["teacher"] as? String == "none" && ceResult["preservation_kl"] as? Float == 0 && ceResult["loss"] as? Float == ceResult["ce"] as? Float, "CE-only loss or teacher is incorrect")
        try require(trainer.step == 1 && (ceResult["gradient_norm"] as? Float ?? 0) > 0 && any(trainer.weights[2] .!= beforeCE[2]).item(Bool.self), "CE-only Adam did not update the expert")
        let cePath=root.appendingPathComponent("ce-expert.safetensors").path, ceCheckpoint=root.appendingPathComponent("ce-checkpoint.safetensors").path
        _ = try trainer.execute(command(["op":"train_save","path":cePath,"checkpoint_path":ceCheckpoint]))
        _ = try trainer.execute(command(["op":"train_step","sample":sample]))
        let ceAfterTwo=trainer.weights+trainer.first+trainer.second+[trainer.rng]
        eval(ceAfterTwo)
        _ = try trainer.execute(command(["op":"train_restore","checkpoint_path":ceCheckpoint,"binding":"first-learning-fixture"]))
        try require(trainer.step == 1, "CE-only checkpoint ordinal not restored")
        _ = try trainer.execute(command(["op":"train_step","sample":sample]))
        try require(zip(ceAfterTwo,trainer.weights+trainer.first+trainer.second+[trainer.rng]).allSatisfy { all($0 .== $1).item(Bool.self) }, "CE-only restored Adam/RNG repeat differed")
        var ceRejected = false
        do { _ = try trainer.execute(command(["op":"train_step","sample":sample,"teacher":"core"])) } catch { ceRejected = true }
        try require(ceRejected && trainer.step == 2 && zip(ceAfterTwo,trainer.weights+trainer.first+trainer.second+[trainer.rng]).allSatisfy { all($0 .== $1).item(Bool.self) }, "Teacher without replay was accepted or changed state")
        _ = try trainer.execute(command(["op":"train_end","restore_parent":true]))
        try require(all(model(one,cache:nil) .== compiledCore).item(Bool.self), "CE-only end did not restore core")
        // A second resident session owns a real immutable parent and can shrink/grow.
        try installAdapterModules(model,core:nil,expert:Dictionary(uniqueKeysWithValues:zip(expertKeys,afterTwo.prefix(3))))
        let withParent=try ExpertTraining(model:model,options:options,hidden:8,vocabulary:16)
        let baseline=try withParent.execute(command(["op":"train_observe","sample":sample,"source":"parent"]))["loss"] as! Float
        _ = try withParent.execute(command(["op":"train_begin","width":64,"seed":7,"initialize":"parent","binding":"parent-fixture"]))
        try require(all(withParent.weights[0][0..<32] .== afterTwo[0]).item(Bool.self) && all(withParent.weights[2][0...,0..<32] .== afterTwo[2]).item(Bool.self), "Parent expansion did not preserve old columns")
        _ = try withParent.execute(command(["op":"train_step","sample":sample,"preservation":protect,"teacher":"parent"]))
        _ = try withParent.execute(command(["op":"train_end","restore_parent":true]))
        let restored=try withParent.execute(command(["op":"train_observe","sample":sample,"source":"parent"]))["loss"] as! Float
        try require(baseline == restored, "Candidate mutated immutable parent")
        let selected = try withParent.execute(command(["op":"train_select","source":"parent"]))
        try require(selected["compilation_enabled"] as? Bool == false, "Learning observation compilation mode missing")
        // Trace lifecycle remains independently covered, even though learning uses eager.
        MLX.compile(enable: true)
        let beforeReplacement=model(one,cache:nil); eval(beforeReplacement)
        let large=[MLXArray.full([32,8],values:MLXArray(Float(0.5))).asType(.bfloat16), MLXArray.full([32,8],values:MLXArray(Float(0.2))).asType(.bfloat16), MLXArray.full([8,32],values:MLXArray(Float(0.4))).asType(.bfloat16)]
        try withParent.install(large)
        let replaced=model(one,cache:nil), direct=model(MLXArray([1,1],[1,2]),cache:nil)[0,0]
        try require(any(replaced .!= beforeReplacement).item(Bool.self) && all(replaced[0,0] .== direct).item(Bool.self), "Compiled trace failed to rebind changed module weights")
        try withParent.select("parent")
        try require(all(model(one,cache:nil) .== beforeReplacement).item(Bool.self), "Compiled trace failed to restore parent")
        // Stress actual destroy/retrace and source changes numerically, not merely
        // the isCompiled flag. This CPU fixture cannot qualify GPU GDN kernels.
        for _ in 0..<24 {
            try withParent.install(nil)
            try require(all(model(one,cache:nil) .== compiledCore).item(Bool.self), "Repeated core rebuild changed values")
            try withParent.install(large)
            try require(all(model(one,cache:nil) .== replaced).item(Bool.self), "Repeated changed-expert rebuild changed values")
            try withParent.select("parent")
            try require(all(model(one,cache:nil) .== beforeReplacement).item(Bool.self), "Repeated parent rebuild changed values")
        }
        try emit(["event":"training_cpu","passed":true,"device":"cpu","actual_adam_updates":7,"ce_only_updates":3,"ce_only_checkpoint_repeat_exact":true,"ce_only_single_forward":true,"ce_only_teacher_rejected":ceRejected,"independent_scalar_adam_max_error":maximumAdamError,"parent_expansion_restored":true,"compiled_trace_rebound":true,"same_source_trace_preserved":true,"numeric_rebuild_cycles":24,"checkpoint_repeat_exact":true,"invalid_operations_rejected":rejected+1,"standard_bf16_export":true,"parent_restored":true,"model_scope":"tiny synthetic tail; not Qwen quality or GPU qualification"])
    }
}
