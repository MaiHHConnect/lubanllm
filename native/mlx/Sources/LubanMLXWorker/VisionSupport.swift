// SPDX-License-Identifier: Apache-2.0
import Foundation
import Darwin
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

/// Python uses the existing validated PIL Qwen processor and chat template.
/// Only bounded, preprocessed pixels cross this private JSONL worker boundary.
struct VisionRequest: Decodable, Sendable {
    let op: String
    let pixel_file: String
    let image_grid_thw: [[Int]]

    func input(_ request: Request) throws -> LMInput {
        try request.validate()
        try require(op == "image_generate" && pixel_file.hasPrefix("/"), "Invalid vision request")
        try require((1...4).contains(image_grid_thw.count), "Expected one to four images")
        var patches = 0
        for grid in image_grid_thw {
            try require(grid.count == 3 && grid[0] == 1 && grid[1] > 0 && grid[2] > 0 && grid[1] <= 4096 && grid[2] <= 4096,
                        "Invalid image grid")
            try require(grid[1] % 2 == 0 && grid[2] % 2 == 0 && grid[1] * grid[2] <= 4096, "Image exceeds pixel budget")
            patches += grid[1] * grid[2]
        }
        try require(request.tokens.filter { $0 == 248056 }.count == patches / 4, "Image token/grid mismatch")
        let url = URL(fileURLWithPath: pixel_file)
        let before = try FileBinding(url)
        try require(before.bytes > 0 && before.bytes <= 101 * 1024 * 1024, "Image tensor file exceeds budget")
        let arrays = try MLX.loadArrays(url: url)
        guard arrays.count == 1, let pixels = arrays["pixel_values"] else { throw WorkerFailure("Invalid image tensors") }
        try require(pixels.shape == [patches, 1536] && pixels.dtype == .float32, "Invalid pixel tensor shape or precision")
        try require(all(isFinite(pixels)).item(Bool.self), "Nonfinite image pixels")
        try require(try FileBinding(url) == before, "Image pixels changed during preparation")
        return LMInput(text: .init(tokens: MLXArray(request.tokens, [1, request.tokens.count])),
                       image: .init(pixels: pixels, frames: image_grid_thw.map { THW($0[0], $0[1], $0[2]) }))
    }
}

// Same cancellation pipe and token/result events as text generation. A vision
// worker never owns a second text model or trains a text-qualified expert.
func performVision(_ request: Request, pixels: VisionRequest, session: VisionSession, cancelFD: Int32) async throws {
    let task = Task.detached { try autoreleasepool {
        try Task.checkCancellation()
        return try generate(request, model: session.model, input: pixels.input(request))
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
    try FileHandle.standardOutput.write(contentsOf: result + Data([10]))
}

final class VisionSession: @unchecked Sendable {
    let model: MLXVLM.Qwen35
    init(_ model: MLXVLM.Qwen35) { self.model = model }
}

func runVision(options: Options, configData: Data, before: FileBinding) async throws {
    try require(options.expertFile == nil, "Text experts are not qualified for image inference")
    let directory = URL(fileURLWithPath: options.modelPath, isDirectory: true)
    let config = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: configData)
    let model = MLXVLM.Qwen35(config)
    let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
    try await loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
    _ = try attachAdapters(model, options: options) // Preserve the same active core; no text expert.
    model.train(false)
    eval(model.parameters())
    Stream.defaultStream.synchronize()
    try require(try FileBinding(directory.appendingPathComponent("model.safetensors")) == before, "MLX model changed during load")
    try emit(["event":"loaded", "protocol":"luban-worker-jsonl-v1", "manifest_sha256":options.manifest,
        "core_loaded":options.coreFile != nil, "core_file_sha256":options.coreSHA as Any? ?? NSNull(),
        "expert_loaded":false, "vision_loaded":true, "inference_precision":"mlx-q4", "direct_q4_loaded":false,
        "mlx_model_path":options.modelPath, "mlx_model_sha256":options.modelSHA, "mlx_model_weight_bytes":before.bytes,
        "mlx_training":false, "layers":24, "vocab_size":model.vocabularySize,
        "execution_skip_mask":0, "execution_exit_after":24, "layer_paging_enabled":false])
    let session = VisionSession(model)
    while let line = readLine() {
        try require(line.utf8.count <= 4 * 1024 * 1024, "Request line too large")
        let data = Data(line.utf8)
        let request = try JSONDecoder().decode(Request.self, from: data)
        let pixels = try JSONDecoder().decode(VisionRequest.self, from: data)
        try await performVision(request, pixels: pixels, session: session, cancelFD: options.cancelFD)
    }
    Stream.defaultStream.synchronize()
}
