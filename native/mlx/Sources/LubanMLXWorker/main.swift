import Foundation
import Darwin
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

struct Options {
    var original: String
    var manifest: String
    var cancelFD: Int32
    var modelPath: String
    var modelSHA: String
    var coreFile: String?
    var coreKey: String?
    var coreSHA: String?
    var expertFile: String?
    var expertSHA: String?
    var vision: Bool
    init(_ args: [String]) throws {
        try require(args.count >= 4, "Usage: luban-mlx-worker original-model manifest-sha cancel-fd [core key sha] --mlx-q4 --mlx-model=directory --mlx-model-sha256=sha [--expert-file=file --expert-sha=sha]")
        original = args[1]; manifest = args[2]
        guard let fd = Int32(args[3]), fd >= 0 else { throw WorkerFailure("Invalid cancellation descriptor") }
        cancelFD = fd
        var rest = Array(args.dropFirst(4))
        if rest.first?.hasPrefix("--") == false {
            try require(rest.count >= 3, "Incomplete core arguments")
            coreFile = rest.removeFirst(); coreKey = rest.removeFirst(); coreSHA = rest.removeFirst()
        }
        var flags = [String: String]()
        for flag in rest {
            let pair = flag.split(separator: "=", maxSplits: 1).map(String.init)
            try require(flags[pair[0]] == nil, "Duplicate worker option")
            flags[pair[0]] = pair.count == 2 ? pair[1] : ""
        }
        try require(Set(flags.keys).isSubset(of: ["--mlx-q4","--mlx-model","--mlx-model-sha256","--expert-file","--expert-sha","--vision"]), "Unsupported worker option")
        guard flags["--mlx-q4"] != nil, let path = flags["--mlx-model"], let sha = flags["--mlx-model-sha256"] else { throw WorkerFailure("MLX model binding required") }
        modelPath = path; modelSHA = sha
        vision = flags["--vision"] != nil
        expertFile = flags["--expert-file"]; expertSHA = flags["--expert-sha"]
        try require((expertFile == nil) == (expertSHA == nil) && (expertFile == nil || coreFile != nil), "Expert requires core and SHA binding")
        for digest in [manifest, modelSHA, coreSHA, expertSHA].compactMap({ $0 }) {
            try require(digest.count == 64 && digest.allSatisfy { "0123456789abcdef".contains($0) }, "Invalid SHA256 binding")
        }
        for path in [original,modelPath,coreFile,expertFile].compactMap({ $0 }) { try require(path.hasPrefix("/"), "Paths must be absolute") }
        if coreFile != nil { try require(coreKey == "model.language_model.layers.23.mlp.base.down_proj.weight", "Unsupported core tensor") }
    }
}

struct Request: Decodable, Sendable {
    let id: String
    let tokens: [Int]
    let max_tokens: Int
    let eos_ids: [Int]
    let temperature: Float
    let seed: UInt64
    let cache_reuse: Bool?
    let context_limit: Int?
    let prefill_step: Int?
    func validate(maximumContext: Int = 262144) throws {
        try require(!id.isEmpty && id.utf8.count <= 256, "Invalid request id")
        let limit = context_limit ?? 32768
        try require(limit > 0 && limit <= maximumContext && limit <= 262144
                    && !tokens.isEmpty && tokens.count <= limit && max_tokens > 0 && max_tokens <= 2048
                    && max_tokens <= limit - tokens.count, "Invalid context/output length")
        try require(!eos_ids.isEmpty && eos_ids.count <= 64 && (tokens + eos_ids).allSatisfy { $0 >= 0 && $0 < 248320 }, "Invalid token ID")
        try require(temperature.isFinite && temperature >= 0 && temperature <= 2, "Invalid temperature")
        try require([32, 64, 128, 256, 512].contains(prefill_step ?? 512), "Invalid prefill step")
    }
}

func emit(_ object: [String: Any]) throws {
    let bytes = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    try FileHandle.standardOutput.write(contentsOf: bytes + Data([10]))
}

// The session is accessed by one awaited request task at a time; no model work overlaps.
final class ModelSession: @unchecked Sendable {
    let model: any LanguageModel
    let training: ExpertTraining
    let contextLimit: Int
    let prefixCache = PrefixCache()
    let diskCache: DiskPrefixCache?
    init(_ model: any LanguageModel, options: Options, contextLimit: Int) throws {
        self.model = model; self.contextLimit = contextLimit
        diskCache = DiskPrefixCache.make(parts: [options.original, options.manifest, options.modelSHA, options.coreSHA ?? "-", options.expertSHA ?? "-", "mlx-swift-0.32.3", "mlx-swift-lm-3.32.3"])
        training = try ExpertTraining(model:model,options:options)
    }
}

func perform(_ request: Request, session: ModelSession, cancelFD: Int32) async throws {
    let task = Task.detached { try autoreleasepool {
        try generate(request, model: session.model, maximumContext: session.contextLimit, prefixCache: session.prefixCache, diskCache: session.diskCache)
    } }
    let source = DispatchSource.makeReadSource(fileDescriptor: cancelFD, queue: .global(qos: .userInitiated))
    source.setEventHandler {
        var bytes = [UInt8](repeating: 0, count: 256)
        let n = Darwin.read(cancelFD, &bytes, bytes.count)
        if n >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { task.cancel() }
    }
    source.resume()
    let result: Data
    do { result = try await task.value }
    catch { await stopCancellationSource(source); throw error }
    await stopCancellationSource(source)
    // No old cancellation handler may consume the next request's cancellation byte.
    try FileHandle.standardOutput.write(contentsOf: result + Data([10]))
}

func stopCancellationSource(_ source: any DispatchSourceRead) async {
    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        source.setCancelHandler { continuation.resume() }
        source.cancel()
    }
}

func clockNS() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

func generate(_ request: Request, model: any LanguageModel, input: LMInput? = nil,
              maximumContext: Int = 32768, prefixCache: PrefixCache? = nil, diskCache: DiskPrefixCache? = nil) throws -> Data {
    var succeeded = false
    defer { if !succeeded { prefixCache?.clear() } }
    try request.validate(maximumContext: maximumContext)
    let previousCacheLimit = Memory.cacheLimit
    if input == nil {
        Memory.cacheLimit = 256 * 1024 * 1024
        Stream.defaultStream.synchronize()
        Memory.clearCache()
        Memory.peakMemory = 0
    }
    defer {
        Stream.defaultStream.synchronize()
        if input == nil {
            Memory.clearCache()
            Memory.cacheLimit = previousCacheLimit
        }
    }
    let start = clockNS()
    var first: UInt64?, last: UInt64?
    var tokens = [Int](), reason = "length"
    let prefillStep = request.prefill_step ?? 512
    var parameters = GenerateParameters(maxTokens: request.max_tokens, temperature: request.temperature,
                                        topP: 0.95, prefill: .init(stepSize: prefillStep), seed: request.seed)
    let reuse = input == nil && request.cache_reuse == true && prefixCache != nil
    var loaded = prefixCache?.workingCopy(for: request.tokens, enabled: reuse, prefillStep: prefillStep) ?? (nil, 0)
    if loaded.0 == nil, reuse, let disk = diskCache,
       let hit = disk.find(request.tokens, prefillStep: prefillStep, make: { try model.newCache(parameters: parameters) }) {
        loaded = hit
    }
    let (saved, cached) = loaded
    let working = try saved ?? PrefixCache.Working(model.newCache(parameters: parameters))
    if input == nil {
        PrefixCache.reserve(working.layers, tokens: request.tokens.count + request.max_tokens)
        prefixCache?.refreshBytes()
    }
    let suffix = Array(request.tokens.dropFirst(cached))
    let checkpoint = PrefixCache.checkpoint(total: suffix.count, prefill: parameters.prefill)
    if reuse && checkpoint > 0 {
        parameters.prefill.progress = { processed, _ in
            if processed == checkpoint {
                prefixCache?.capture(request.tokens, count: cached + processed, cache: working.layers, prefillStep: prefillStep)
                diskCache?.save(request.tokens, count: cached + processed, layers: working.layers, prefillStep: prefillStep)
            }
        }
    }
    var iterator: TokenIterator?
    do {
        try Task.checkCancellation()
        iterator = try TokenIterator(input: input ?? LMInput(text: .init(tokens: MLXArray(suffix))), model: model, cache: working.layers, parameters: parameters)
    } catch is CancellationError {
        Stream.defaultStream.synchronize()
        return try JSONSerialization.data(withJSONObject: ["id":request.id,"event":"result","tokens":[Int](),"prompt_tokens":request.tokens.count,"generated_tokens":0,
                  "finish_reason":"cancelled","complete":false,"prefill_ns":clockNS() - start,
                  "ttft_ns":NSNull(),"decode_ns":NSNull(),"cached_prompt_tokens":0,"cache_capacity":0])
    }
    let prefill = UInt64(max(0, iterator!.promptPrefillTime) * 1_000_000_000)
    let prefillMemory = Memory.snapshot()
    while tokens.count < request.max_tokens {
        if Task.isCancelled { reason = "cancelled"; break }
        guard let token = iterator!.next() else { break }
        if Task.isCancelled { reason = "cancelled"; break }
        let now = clockNS(); first = first ?? now; last = now
        tokens.append(token)
        try emit(["id":request.id, "event":"token", "token":token])
        if request.eos_ids.contains(token) { reason = "stop"; break }
    }
    iterator = nil
    parameters.prefill.progress = nil
    working.layers = []
    Stream.defaultStream.synchronize()
    if reason == "cancelled" { prefixCache?.clear() }
    succeeded = reason != "cancelled"
    var result: [String: Any] = ["id":request.id, "event":"result", "tokens":tokens, "prompt_tokens":request.tokens.count,
              "generated_tokens":tokens.count, "finish_reason":reason, "complete":reason == "stop",
              "prefill_ns":prefill, "ttft_ns":first.map { $0 - start } as Any? ?? NSNull(),
              "decode_ns":tokens.count > 1 ? (last! - first!) as Any : NSNull(),
              "cached_prompt_tokens":cached, "cache_capacity":prefixCache?.tokens.count ?? 0]
    if input == nil {
        result["native_memory"] = ["prefill_step":request.prefill_step ?? 512,
                  "cache_limit_bytes":Memory.cacheLimit,
                  "prefill_active_bytes":prefillMemory.activeMemory,
                  "prefill_cache_bytes":prefillMemory.cacheMemory,
                  "request_peak_active_bytes":Memory.peakMemory]
        result["prefix_cache_bytes"] = prefixCache?.bytes ?? 0
    }
    return try JSONSerialization.data(withJSONObject: result)
}

func run() async throws {
    if CommandLine.arguments == [CommandLine.arguments[0], "--help"] {
        print("luban-worker-jsonl-v1 native Swift MLX Qwen3.5 Q4 worker; --validate-request validates JSONL on CPU without a model")
        return
    }
    if CommandLine.arguments == [CommandLine.arguments[0], "--validate-request"] {
        while let line = readLine() {
            let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8)); try request.validate()
            try emit(["id":request.id,"event":"validated"])
        }
        return
    }
    if CommandLine.arguments.dropFirst().first == "--check-training" {
        try checkTraining(); return
    }
    if CommandLine.arguments.dropFirst().first == "--check-expert" {
        try checkExpert(CommandLine.arguments); return
    }
    if CommandLine.arguments.dropFirst().first == "--check-adapter-structure" {
        try checkAdapterStructure(); return
    }
    let options = try Options(CommandLine.arguments)
    let fdFlags = fcntl(options.cancelFD, F_GETFL)
    try require(fdFlags >= 0 && fcntl(options.cancelFD, F_SETFL, fdFlags | O_NONBLOCK) == 0, "Cannot configure cancellation pipe")
    let directory = URL(fileURLWithPath: options.modelPath, isDirectory: true)
    let weightURL = directory.appendingPathComponent("model.safetensors")
    let before = try FileBinding(weightURL)
    let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("download-receipt.json"))) as? [String:Any]
    let files = receipt?["files"] as? [[String:Any]]
    let weight = files?.first { $0["name"] as? String == "model.safetensors" }
    try require(receipt?["complete"] as? Bool == true && weight?["sha256"] as? String == options.modelSHA && (weight?["bytes"] as? NSNumber)?.uint64Value == before.bytes, "MLX download receipt binding mismatch")
    guard let finished = receipt?["finished"] as? Double else { throw WorkerFailure("MLX receipt timestamp missing") }
    try require(before.modified.timeIntervalSince1970 <= finished + 1, "MLX weight file changed after verification")
    let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
    guard let json = try JSONSerialization.jsonObject(with: configData) as? [String:Any],
          let text = json["text_config"] as? [String:Any], let quant = json["quantization"] as? [String:Any] else { throw WorkerFailure("Invalid Qwen configuration") }
    try require(json["model_type"] as? String == "qwen3_5" && text["hidden_size"] as? Int == 2048 && text["intermediate_size"] as? Int == 6144 && text["num_hidden_layers"] as? Int == 24 && quant["bits"] as? Int == 4 && quant["group_size"] as? Int == 64 && quant["mode"] as? String == "affine", "Unsupported model architecture or quantization")
    if options.vision {
        try await runVision(options: options, configData: configData, before: before)
        return
    }
    let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
    let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
    let model = Qwen35Model(config)
    try require(model.vocabularySize == 248320 && model.kvHeads.count == 24, "Only Qwen3.5-2B is supported")
    try await loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
    let languageBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    let width = try attachAdapters(model, options: options)
    model.train(false)
    try require(model.modules().allSatisfy { !$0.training }, "Inference module remained in training mode")
    eval(model.parameters())
    Stream.defaultStream.synchronize()
    try require(try FileBinding(weightURL) == before, "MLX model changed during load")
    var ack: [String:Any] = ["event":"loaded", "protocol":"luban-worker-jsonl-v1", "manifest_sha256":options.manifest,
        "core_loaded":options.coreFile != nil, "core_file_sha256":options.coreSHA as Any? ?? NSNull(),
        "expert_loaded":options.expertFile != nil,"expert_file_sha256":options.expertSHA as Any? ?? NSNull(),"expert_width":width,"expert_layer":23,
        "inference_precision":"mlx-q4","direct_q4_loaded":false,"mlx_model_path":options.modelPath,
        "mlx_model_sha256":options.modelSHA,"mlx_model_weight_bytes":before.bytes,"mlx_language_weight_bytes":languageBytes,
        "mlx_source_binding":"previous_download_sha256_and_frozen_file_stat","mlx_cache_reuse":true,"mlx_kv_reservation":true,"mlx_training":model.training,
        "model_type":"qwen3_5","layers":24,"vocab_size":model.vocabularySize,"quantization_bits":4,"quantization_group_size":64,
        "execution_skip_mask":0,"execution_exit_after":24,"layer_paging_enabled":false]
    ack["mlx_model_revision"] = receipt?["revision"]
    try emit(ack)
    guard let contextLimit = text["max_position_embeddings"] as? Int, (1...262144).contains(contextLimit) else {
        throw WorkerFailure("Invalid model context limit")
    }
    let session = try ModelSession(model,options:options,contextLimit:contextLimit)
    var tokenizer: QwenTokenizer?
    while let line = readLine() {
        try require(line.utf8.count <= 4 * 1024 * 1024, "Request line too large")
        let data = Data(line.utf8)
        let message = try JSONSerialization.jsonObject(with:data) as? [String:Any]
        if let operation = message?["op"] as? String, operation != "generate" {
            if operation == "clear_cache" {
                guard let id = message?["id"] as? String, !id.isEmpty, id.utf8.count <= 256 else { throw WorkerFailure("Invalid request id") }
                session.prefixCache.clear()
                Stream.defaultStream.synchronize()
                Memory.clearCache()
                try emit(["id":id,"event":"cache_cleared"])
            } else if operation == "route_features" {
                try routeFeatures(message!, model: model)
            } else if operation == "prepare" || operation == "decode" {
                if tokenizer == nil { tokenizer = try QwenTokenizer(modelDirectory:URL(fileURLWithPath:options.original)) }
                try tokenizeCommand(message!,tokenizer:tokenizer!)
            } else {
                session.prefixCache.clear()
                let command = try JSONDecoder().decode(TrainingCommand.self,from:data)
                try await performTraining(command,session:session,cancelFD:options.cancelFD)
            }
        } else {
            let request = try JSONDecoder().decode(Request.self, from:data)
            try await perform(request, session: session, cancelFD: options.cancelFD)
        }
    }
    Stream.defaultStream.synchronize()
}

do { try await run() }
catch {
    try? emit(["event":"error", "error":String(describing:error)])
    FileHandle.standardError.write(Data(("MLX worker: \(error)\n").utf8))
    exit(1)
}
