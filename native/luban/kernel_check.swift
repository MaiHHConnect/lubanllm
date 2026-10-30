// SPDX-License-Identifier: Apache-2.0
// Paired shader checks; no model loading, weight conversion, or tolerance changes.
import Foundation
import Metal

enum CheckError: Error { case failed(String) }
func require(_ ok: Bool, _ message: String) throws {
    if !ok { throw CheckError.failed(message) }
}
let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    fatalError("usage: kernel-check REFERENCE_DIR CANDIDATE_DIR [--bench] [--compile-only]")
}
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.languageVersion = .version4_0
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
func pipeline(_ directory: String, _ file: String, _ name: String) throws -> MTLComputePipelineState {
    let source = try String(contentsOfFile: directory + "/" + file, encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: options)
    return try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
let baselineProjection = try pipeline(arguments[1], "bf16.metal", "qwen35_bf16_projection")
let candidateProjection = try pipeline(arguments[2], "bf16.metal", "qwen35_bf16_projection")
let baselineDelta = try pipeline(arguments[1], "bf16_delta.metal", "qwen35_bf16_gdn_chain")
let candidateDelta = try pipeline(arguments[2], "bf16_delta.metal", "qwen35_bf16_gdn_chain")
if arguments.contains("--compile-only") {
    print("Metal 4.0 safe/fast: both projection and GDN reference/candidate compiled")
    exit(0)
}
func buffer(_ bytes: Int) -> MTLBuffer {
    device.makeBuffer(length: bytes, options: .storageModeShared)!
}
func bf16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    return UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
}
var seed: UInt32 = 42
func sample(_ scale: Float) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    return (Float(Int32(truncatingIfNeeded: seed)) / 2147483648) * scale
}
func fillBF16(_ target: MTLBuffer, _ count: Int, _ scale: Float) {
    let values = target.contents().bindMemory(to: UInt16.self, capacity: count)
    for i in 0..<count { values[i] = bf16(sample(scale)) }
}
func equal(_ a: MTLBuffer, _ b: MTLBuffer, _ bytes: Int, _ label: String) throws {
    try require(memcmp(a.contents(), b.contents(), bytes) == 0, "bit mismatch: " + label)
}
func execute(_ repeats: Int = 1, _ encode: (MTLComputeCommandEncoder) -> Void) throws -> Double {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    for _ in 0..<repeats { encode(encoder) }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try require(command.status == .completed, String(describing: command.error))
    return (command.gpuEndTime - command.gpuStartTime) / Double(repeats)
}
func projection(_ pipeline: MTLComputePipelineState, _ x: MTLBuffer, _ w: MTLBuffer,
                _ out: MTLBuffer, _ k: Int, _ n: Int, _ repeats: Int = 1) throws -> Double {
    try execute(repeats) { encoder in
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x, offset: 0, index: 0)
        var dims = [UInt32(1), UInt32(k), UInt32(n)]
        encoder.setBytes(&dims, length: 12, index: 1)
        encoder.setBuffer(w, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        encoder.dispatchThreads(MTLSize(width: 256 * ((n + 7) / 8), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
func delta(_ pipeline: MTLComputePipelineState, _ inputs: [MTLBuffer], _ state: MTLBuffer,
           _ y: MTLBuffer, _ out: MTLBuffer, _ rows: Int, _ record: Bool, _ repeats: Int = 1) throws -> Double {
    try execute(repeats) { encoder in
        encoder.setComputePipelineState(pipeline)
        for index in 0..<5 { encoder.setBuffer(inputs[index], offset: 0, index: index) }
        encoder.setBuffer(state, offset: 0, index: 5)
        var dims = [Int32(rows), record ? Int32(1) : Int32(0)]
        encoder.setBytes(&dims, length: 8, index: 6)
        encoder.setBuffer(y, offset: 0, index: 7)
        encoder.setBuffer(out, offset: 0, index: 8)
        encoder.dispatchThreads(MTLSize(width: 32, height: 128, depth: 16),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }
}
func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
let bench = arguments.contains("--bench")
var projectionCases: [[String: Any]] = []
// Real model dimensions plus non-aligned tails exercise the cleanup loop.
for (k, n) in [(127, 17), (2047, 17), (6143, 17), (2048, 16), (2048, 512),
               (2048, 2048), (2048, 6144), (6144, 2048)] + (bench ? [(2048, 248320)] : []) {
    let x = buffer(k * 2), w = buffer(n * k * 2), a = buffer(n * 2), b = buffer(n * 2)
    fillBF16(x, k, 1)
    fillBF16(w, n * k, 0.1)
    _ = try projection(baselineProjection, x, w, a, k, n)
    _ = try projection(candidateProjection, x, w, b, k, n)
    try equal(a, b, n * 2, "projection K\(k) N\(n)")
    var row: [String: Any] = ["k": k, "n": n, "bf16_equal": true]
    if bench && k % 128 == 0 {
        var old: [Double] = [], new: [Double] = []
        for round in 0..<7 {
            if round % 2 == 0 {
                old.append(try projection(baselineProjection, x, w, a, k, n, 20))
                new.append(try projection(candidateProjection, x, w, b, k, n, 20))
            } else {
                new.append(try projection(candidateProjection, x, w, b, k, n, 20))
                old.append(try projection(baselineProjection, x, w, a, k, n, 20))
            }
        }
        row["reference_seconds"] = old
        row["candidate_seconds"] = new
        row["median_speedup"] = median(old) / median(new)
    }
    projectionCases.append(row)
}
var deltaCases: [[String: Any]] = []
let stateBytes = 16 * 128 * 128 * 4
for rows in [1, 2, 17, 32] {
    let inputs = [buffer(rows * 2048 * 2), buffer(rows * 2048 * 2), buffer(rows * 2048 * 2),
                  buffer(rows * 16 * 4), buffer(rows * 16 * 2)]
    for input in inputs[0..<3] { fillBF16(input, rows * 2048, 0.08) }
    let g = inputs[3].contents().bindMemory(to: Float.self, capacity: rows * 16)
    let beta = inputs[4].contents().bindMemory(to: UInt16.self, capacity: rows * 16)
    for i in 0..<(rows * 16) { g[i] = 0.9 + sample(0.09); beta[i] = bf16(0.5 + sample(0.4)) }
    let initial = buffer(stateBytes)
    let fp32 = initial.contents().bindMemory(to: Float.self, capacity: stateBytes / 4)
    for i in 0..<(stateBytes / 4) { fp32[i] = sample(0.1) }
    for record in [false, true] {
        let states = [buffer(stateBytes), buffer(stateBytes)]
        let outputs = [buffer(rows * 2048 * 2), buffer(rows * 2048 * 2)]
        let saved = [buffer(stateBytes * (record ? rows : 1)), buffer(stateBytes * (record ? rows : 1))]
        for i in 0..<2 {
            memcpy(states[i].contents(), initial.contents(), stateBytes)
            _ = try delta(i == 0 ? baselineDelta : candidateDelta, inputs, states[i], outputs[i],
                          record ? saved[i] : states[i], rows, record)
        }
        try equal(outputs[0], outputs[1], rows * 2048 * 2, "GDN Y W\(rows) record\(record)")
        try equal(record ? saved[0] : states[0], record ? saved[1] : states[1],
                  record ? rows * stateBytes : stateBytes, "GDN FP32 state W\(rows) record\(record)")
        var row: [String: Any] = ["rows": rows, "record": record, "bf16_equal": true, "fp32_state_equal": true]
        if bench {
            var old: [Double] = [], new: [Double] = []
            // Separate destinations keep every repeated invocation on identical initial state.
            let destination = buffer(stateBytes * (record ? rows : 1))
            for round in 0..<7 {
                if round % 2 == 0 {
                    old.append(try delta(baselineDelta, inputs, initial, outputs[0], destination, rows, record, 100))
                    new.append(try delta(candidateDelta, inputs, initial, outputs[1], destination, rows, record, 100))
                } else {
                    new.append(try delta(candidateDelta, inputs, initial, outputs[1], destination, rows, record, 100))
                    old.append(try delta(baselineDelta, inputs, initial, outputs[0], destination, rows, record, 100))
                }
            }
            row["reference_seconds"] = old
            row["candidate_seconds"] = new
            row["median_speedup"] = median(old) / median(new)
        }
        deltaCases.append(row)
    }
}
let result: [String: Any] = ["device": device.name, "seed": 42, "language": "Metal 4.0", "math": "safe/fast",
                              "metric": "GPU command time / repeated dispatch count; warm buffers; not full-model speed",
                              "projection": projectionCases, "delta": deltaCases]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
