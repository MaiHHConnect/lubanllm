import Foundation
import MLX
import MLXNN
import MLXLMCommon

struct TrainingSample: Decodable, Sendable {
    let tokens: [Int]
    let labels: [Int]
    func validate(vocabulary: Int) throws {
        try require(tokens.count >= 2 && tokens.count <= 2048 && labels.count == tokens.count, "Invalid training sequence length")
        try require(tokens.allSatisfy { $0 >= 0 && $0 < vocabulary } && labels.allSatisfy { $0 == -100 || ($0 >= 0 && $0 < vocabulary) }, "Invalid training token/label")
        try require(labels.dropFirst().contains { $0 != -100 }, "Training sample has no answer/EOS target")
    }
    var positions: [Int] { labels.indices.dropFirst().filter { labels[$0] != -100 }.map { $0 - 1 } }
    var targets: [Int] { labels.dropFirst().filter { $0 != -100 } }
}

struct TrainingCommand: Decodable, Sendable {
    let id: String
    let op: String
    let seed: UInt64?
    let width: Int?
    let learning_rate: Float?
    let preservation_weight: Float?
    let initialize: String?
    let binding: String?
    let sample: TrainingSample?
    let preservation: TrainingSample?
    let teacher: String?
    let source: String?
    let path: String?
    let checkpoint_path: String?
    let restore_parent: Bool?
}

let expertKeys = ["gate.weight", "up.weight", "down.weight"]

// One resident inference graph owns the training state. Only these three expert arrays are
// differentiated; the quantized core and its BF16 identity patch never enter Adam.
final class ExpertTraining {
    let model: any LanguageModel
    let layer: Module
    let base: Module
    let original: ExpertMLP?
    let parent: [MLXArray]?
    let sourceBinding: String
    let hidden: Int
    let vocabulary: Int
    var adapter: ExpertMLP?
    var weights = [MLXArray](), first = [MLXArray](), second = [MLXArray]()
    var rng = MLXRandom.key(0)
    var step = 0
    var rate: Float = 0.0002
    var preservationWeight: Float = 5
    var binding = ""
    var width = 0
    var begun = false
    private var selectedSource: String?

    init(model: any LanguageModel, options: Options, hidden: Int = 2048, vocabulary: Int = 248320) throws {
        self.hidden = hidden; self.vocabulary = vocabulary
        self.model = model
        var foundLayer: Module?, foundMLP: Module?
        model.visit { key, module in
            if key == "language_model.model.layers.23" { foundLayer = module }
            if key == "language_model.model.layers.23.mlp" { foundMLP = module }
        }
        guard let foundLayer, let foundMLP else { throw WorkerFailure("Missing training layer") }
        layer = foundLayer
        original = foundMLP as? ExpertMLP
        base = original?.base ?? foundMLP
        parent = original.map { [$0.gate.weight, $0.up.weight, $0.down.weight] }
        selectedSource = original == nil ? "core" : "parent"
        sourceBinding = [options.manifest, options.modelSHA, options.coreSHA ?? "none", options.expertSHA ?? "none"].joined(separator: ":")
    }

    func install(_ arrays: [MLXArray]?) throws {
        selectedSource = nil
        if let arrays {
            let next = ExpertMLP(base: base, weights: Dictionary(uniqueKeysWithValues: zip(expertKeys, arrays)))
            next.detachFrozenInput = true
            next.train(false)
            try layer.update(modules: ModuleChildren.unflattened(["mlp":next]), verify:.noUnusedKeys)
            adapter = next
        } else {
            try layer.update(modules: ModuleChildren.unflattened(["mlp":base]), verify:.noUnusedKeys)
            adapter = nil
        }
        // Upstream traces snapshot module-array objects. Replacing an adapter must
        // invalidate that snapshot; the next decode recompiles with current state.
        model.invalidateCompiledTraces()
    }

    func select(_ source: String) throws {
        try require(["candidate","parent","core"].contains(source), "Invalid observation source")
        if source == "candidate" { try require(begun, "Training not begun") }
        // Observing the same resident source does not replace modules or discard its
        // compiled decode traces. Candidate commits explicitly clear this selection.
        if selectedSource == source { return }
        switch source {
        case "candidate": try require(begun, "Training not begun"); try install(weights)
        case "parent": try install(parent)
        case "core": try install(nil)
        default: throw WorkerFailure("Invalid observation source")
        }
        selectedSource = source
    }

    func selectedLogits(_ sample: TrainingSample) -> MLXArray {
        let output = model(.init(tokens: MLXArray(sample.tokens, [1, sample.tokens.count])), cache:nil, state:nil).logits
        return take(output[0], MLXArray(sample.positions), axis:0).asType(.float32)
    }

    func begin(_ command: TrainingCommand) throws -> [String:Any] {
        try require(!begun, "Training already begun; end current candidate first")
        guard let w = command.width, let seed = command.seed, let bind = command.binding else { throw WorkerFailure("Training width, seed and binding required") }
        try require([32,64,128,256,512].contains(w) && !bind.isEmpty && bind.utf8.count <= 4096, "Invalid training width/binding")
        let lr = command.learning_rate ?? 0.0002, kw = command.preservation_weight ?? 5
        try require(lr == 0.0002 && kw == 5, "Protected training requires learning rate .0002 and KL weight 5")
        try require(command.initialize == "parent" || command.initialize == "fresh", "Invalid candidate initialization")
        try require(command.initialize != "parent" || parent != nil, "No parent expert")
        var key = MLXRandom.key(seed)
        func random(_ shape: [Int]) -> MLXArray {
            let (a,b) = MLXRandom.split(key:key); key = a
            return MLXRandom.uniform(-Float(1 / sqrt(Double(hidden))) ..< Float(1 / sqrt(Double(hidden))), shape, key:b).asType(.bfloat16)
        }
        let next = [random([w,hidden]), random([w,hidden]), MLXArray.zeros([hidden,w],dtype:.bfloat16)]
        if command.initialize == "parent", let parent {
            let count = min(w,parent[0].dim(0))
            next[0][0..<count] = parent[0][0..<count]
            next[1][0..<count] = parent[1][0..<count]
            next[2][0...,0..<count] = parent[2][0...,0..<count]
        }
        eval(next)
        try Task.checkCancellation()
        weights = next; first = next.map { MLXArray.zeros($0.shape,dtype:.float32) }; second = first.map { MLXArray.zeros($0.shape,dtype:.float32) }
        rng = key; step = 0; rate = lr; preservationWeight = kw; binding = bind; width = w
        begun = true; selectedSource = nil
        try select("candidate")
        return ["event":"train_started","step":step,"width":width,"optimizer":"Adam-FP32-beta0.9-0.999-eps1e-6","trainable_parameters":3*w*hidden,"base_frozen":true]
    }

    func update(_ command: TrainingCommand) throws -> [String:Any] {
        try require(begun && step < 64, "Candidate update budget exhausted or training not begun")
        guard let sample = command.sample else { throw WorkerFailure("Train sample required") }
        try sample.validate(vocabulary:vocabulary)
        let protection = command.preservation
        try protection?.validate(vocabulary:vocabulary)
        let teacher = protection == nil ? "none" : command.teacher ?? "core"
        if protection != nil {
            try require(teacher == "core" || teacher == "parent", "Invalid frozen teacher")
            try require(teacher != "parent" || parent != nil, "No frozen parent teacher")
        } else {
            try require(command.teacher == nil, "Frozen teacher requires a preservation sample")
        }
        defer { try? select("candidate"); Stream.defaultStream.synchronize() }
        try Task.checkCancellation()
        var reference: MLXArray?
        if let protection {
            try select(teacher)
            let logits = stopGradient(logSoftmax(selectedLogits(protection), axis:-1))
            eval(logits)
            try require(all(isFinite(logits)).item(Bool.self), "Nonfinite fixed teacher")
            reference = logits
        }
        try Task.checkCancellation()
        try select("candidate")
        guard let adapter else { throw WorkerFailure("Candidate missing") }
        let targets = MLXArray(sample.targets)
        // The graph before the final expert has no trainable inputs. Explicit stopGradient
        // at its input/base also prevents upstream attention/GDN derivatives and retention.
        selectedSource = nil
        let vg = valueAndGrad({ (values: [MLXArray]) -> [MLXArray] in
            adapter.update(parameters:ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues:zip(expertKeys,values))))
            let ce = crossEntropy(logits:self.selectedLogits(sample), targets:targets, reduction:.mean)
            if let protection, let reference {
                let student = logSoftmax(self.selectedLogits(protection), axis:-1)
                let kl = sum(exp(reference) * (reference - student)) / Float(protection.positions.count)
                return [ce + self.preservationWeight * kl, ce, kl]
            }
            return [ce, ce, MLXArray(Float(0))]
        }, argumentNumbers:[0,1,2])
        let (losses, gradients) = vg(weights)
        eval(losses, gradients)
        try require(losses.allSatisfy { all(isFinite($0)).item(Bool.self) } && gradients.allSatisfy { all(isFinite($0)).item(Bool.self) }, "Nonfinite loss/gradient")
        try Task.checkCancellation()
        let grad32 = gradients.map { $0.asType(.float32) }
        let norm = sqrt(grad32.map { sum(square($0)) }.reduce(MLXArray(Float(0)),+))
        let scale = minimum(MLXArray(Float(1)), MLXArray(Float(1)) / (norm + Float(1e-6)))
        let clipped = grad32.map { $0 * scale }
        let nextStep = step + 1
        let nextM = zip(first,clipped).map { $0 * Float(0.9) + $1 * Float(0.1) }
        let nextV = zip(second,clipped).map { $0 * Float(0.999) + square($1) * Float(0.001) }
        let correction1 = Float(1 - pow(0.9,Double(nextStep)))
        let correction2 = Float(sqrt(1 - pow(0.999,Double(nextStep))))
        let nextWeights = weights.indices.map { index in
            (weights[index].asType(.float32) - (rate/correction1) * nextM[index] / (sqrt(nextV[index])/correction2 + Float(1e-6))).asType(.bfloat16)
        }
        eval(nextWeights,nextM,nextV,norm)
        try require((nextWeights+nextM+nextV).allSatisfy { all(isFinite($0)).item(Bool.self) }, "Nonfinite updated Adam state")
        try Task.checkCancellation()
        // Commit all three parameters, both moments and ordinal together only after checks.
        weights = nextWeights; first = nextM; second = nextV; step = nextStep
        selectedSource = nil
        return ["event":"train_step","step":step,"loss":losses[0].item(Float.self),"ce":losses[1].item(Float.self),"preservation_kl":losses[2].item(Float.self),"gradient_norm":norm.item(Float.self),"teacher":teacher]
    }

    func observe(_ command: TrainingCommand) throws -> [String:Any] {
        guard let sample = command.sample else { throw WorkerFailure("Observation sample required") }
        try sample.validate(vocabulary:vocabulary)
        try select(command.source ?? "candidate")
        let loss = crossEntropy(logits:selectedLogits(sample),targets:MLXArray(sample.targets),reduction:.mean)
        eval(loss)
        try require(all(isFinite(loss)).item(Bool.self), "Nonfinite observation loss")
        try Task.checkCancellation()
        return ["event":"train_observed","step":step,"loss":loss.item(Float.self),"source":command.source ?? "candidate"]
    }

    func save(_ command: TrainingCommand) throws -> [String:Any] {
        try require(begun, "Training not begun")
        guard let path = command.path, let checkpoint = command.checkpoint_path else { throw WorkerFailure("Both save paths required") }
        try require(path != checkpoint, "Checkpoint and expert paths must differ")
        let metadata = ["schema":"luban-mlx-expert-training-v1","source":sourceBinding,"binding":binding,"width":String(width),"step":String(step),"learning_rate":String(rate),"preservation_weight":String(preservationWeight),"optimizer":"fp32-adam-v1"]
        var arrays = Dictionary(uniqueKeysWithValues:zip(expertKeys,weights))
        for i in weights.indices { arrays["adam.m.\(i)"] = first[i]; arrays["adam.v.\(i)"] = second[i] }
        arrays["rng"] = rng
        try Task.checkCancellation()
        try saveNewArrays(arrays,metadata:metadata,path:checkpoint)
        do { try saveNewArrays(Dictionary(uniqueKeysWithValues:zip(expertKeys,weights)),metadata:[:],path:path) }
        catch { try? FileManager.default.removeItem(atPath:checkpoint); throw error }
        return ["event":"train_saved","step":step,"width":width,"path":path,"checkpoint_path":checkpoint]
    }

    func restore(_ command: TrainingCommand) throws -> [String:Any] {
        try require(begun && command.binding == binding, "Checkpoint binding differs")
        guard let path = command.checkpoint_path, path.hasPrefix("/") else { throw WorkerFailure("Checkpoint path required") }
        let url = URL(fileURLWithPath:path), before = try FileBinding(url)
        try require(before.bytes <= 64*1024*1024, "Checkpoint too large")
        let (arrays,meta) = try MLX.loadArraysAndMetadata(url:url)
        try require(meta["schema"] == "luban-mlx-expert-training-v1" && meta["source"] == sourceBinding && meta["binding"] == binding && meta["width"] == String(width) && meta["optimizer"] == "fp32-adam-v1" && Float(meta["learning_rate"] ?? "") == rate && Float(meta["preservation_weight"] ?? "") == preservationWeight, "Checkpoint provenance differs")
        guard let savedStep = Int(meta["step"] ?? ""), (0...64).contains(savedStep) else { throw WorkerFailure("Invalid checkpoint step") }
        let expected = Set(expertKeys + (0..<3).flatMap { ["adam.m.\($0)","adam.v.\($0)"] } + ["rng"])
        try require(Set(arrays.keys) == expected, "Checkpoint tensors differ")
        var next = [MLXArray](), m = [MLXArray](), v = [MLXArray]()
        for i in 0..<3 {
            let w = arrays[expertKeys[i]]!, a = arrays["adam.m.\(i)"]!, b = arrays["adam.v.\(i)"]!
            try require(w.shape == weights[i].shape && w.dtype == .bfloat16 && a.shape == w.shape && b.shape == w.shape && a.dtype == .float32 && b.dtype == .float32, "Checkpoint shape/dtype differs")
            try require([w,a,b].allSatisfy { all(isFinite($0)).item(Bool.self) } && all(b .>= MLXArray(Float(0))).item(Bool.self), "Invalid checkpoint value")
            next.append(w); m.append(a); v.append(b)
        }
        let key = arrays["rng"]!
        try require(key.shape == [2] && key.dtype == .uint32 && (try FileBinding(url)) == before, "Checkpoint RNG/file changed")
        try Task.checkCancellation()
        weights = next; first = m; second = v; rng = key; step = savedStep
        selectedSource = nil
        try select("candidate")
        return ["event":"train_restored","step":step,"width":width]
    }

    func execute(_ command: TrainingCommand) throws -> [String:Any] {
        try require(!command.id.isEmpty && command.id.utf8.count <= 256, "Invalid training request id")
        // Source replacement retraces decode and changed greedy outputs in real-model
        // ABBA checks. Learning owns this process; keep its observations on the eager path.
        MLX.compile(enable: false)
        var result: [String:Any]
        switch command.op {
        case "train_begin": result = try begin(command)
        case "train_step": result = try update(command)
        case "train_observe": result = try observe(command)
        case "train_select": try select(command.source ?? "candidate"); result = ["event":"train_selected","source":command.source ?? "candidate","step":step]
        case "train_save": result = try save(command)
        case "train_restore": result = try restore(command)
        case "train_end":
            try require(command.restore_parent == true, "End requires explicit parent restoration")
            try select("parent"); weights=[]; first=[]; second=[]; begun=false; step=0
            result=["event":"train_ended","parent_restored":true]
        default: throw WorkerFailure("Unknown training operation")
        }
        result["id"] = command.id
        result["compilation_enabled"] = false
        return result
    }
}

func saveNewArrays(_ arrays: [String:MLXArray], metadata: [String:String], path: String) throws {
    try require(path.hasPrefix("/") && path.hasSuffix(".safetensors") && !FileManager.default.fileExists(atPath:path), "Save requires new absolute safetensors path")
    let destination = URL(fileURLWithPath:path)
    let temporary = destination.deletingLastPathComponent().appendingPathComponent(".mlx-\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at:temporary) }
    try MLX.save(arrays:arrays,metadata:metadata,url:temporary)
    let handle = try FileHandle(forWritingTo:temporary)
    try handle.synchronize(); try handle.close()
    // Same-volume move refuses an existing destination and preserves old checkpoints.
    try FileManager.default.moveItem(at:temporary,to:destination)
}

func performTraining(_ command: TrainingCommand, session: ModelSession, cancelFD: Int32) async throws {
    let task = Task.detached { try autoreleasepool { try JSONSerialization.data(withJSONObject:session.training.execute(command)) } }
    let source = DispatchSource.makeReadSource(fileDescriptor:cancelFD,queue:.global(qos:.userInitiated))
    source.setEventHandler {
        var bytes = [UInt8](repeating:0,count:256)
        let n = Darwin.read(cancelFD,&bytes,bytes.count)
        if n >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { task.cancel() }
    }
    source.resume()
    do {
        let result = try await task.value
        await stopCancellationSource(source)
        Stream.defaultStream.synchronize()
        try FileHandle.standardOutput.write(contentsOf:result + Data([10]))
    } catch {
        await stopCancellationSource(source)
        Stream.defaultStream.synchronize()
        if error is CancellationError {
            try emit(["id":command.id,"event":"train_cancelled","step":session.training.step,"cancelled":true])
        } else {
            try emit(["id":command.id,"event":"train_error","step":session.training.step,"error":String(describing:error)])
        }
    }
}

func tokenizeCommand(_ message: [String:Any], tokenizer: QwenTokenizer) throws {
    guard let id = message["id"] as? String, !id.isEmpty else { throw WorkerFailure("Tokenizer request id required") }
    if message["op"] as? String == "decode" {
        guard let tokens = message["tokens"] as? [Int], tokens.count <= 32768 else { throw WorkerFailure("Invalid decode tokens") }
        try emit(["id":id,"event":"decoded","text":try tokenizer.decode(tokens)])
    } else {
        guard let messages = message["messages"] as? [[String:Any]], messages.count <= 256 else { throw WorkerFailure("Invalid preparation messages") }
        let prompt = try tokenizer.encode(tokenizer.chatTemplate(messages,enableThinking:false))
        var result: [String:Any] = ["id":id,"event":"prepared","tokens":prompt,"prompt_tokens":prompt.count,"eos":248046]
        if let answer = message["answer"] as? String {
            let target = try tokenizer.encode(answer) + [248046]
            let tokens = prompt + target
            try require(tokens.count <= 2048 && !target.isEmpty, "Training preparation exceeds sequence budget")
            result["tokens"] = tokens; result["labels"] = Array(repeating:-100,count:prompt.count) + target
        }
        try emit(result)
    }
}
