// SPDX-License-Identifier: Apache-2.0
// Independent bounded Metal qualification. No base model, tokenizer or product files.
import Foundation
import Metal

enum ExpertCheckError: Error { case failed(String) }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw ExpertCheckError.failed(message) }
}
@main struct ExpertCheck {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 3 else { throw ExpertCheckError.failed("usage: expert-check KERNEL_DIRECTORY FIXTURE_DIRECTORY") }
        let kernelDirectory = URL(fileURLWithPath: args[1]), root = URL(fileURLWithPath: args[2])
        let plan = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("cases.json"))) as! [[String: Int]]
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw ExpertCheckError.failed("Metal unavailable")
        }
        let options = MTLCompileOptions(); options.languageVersion = .version4_0
        options.mathMode = .safe; options.mathFloatingPointFunctions = .fast
        func library(_ name: String) throws -> MTLLibrary {
            try device.makeLibrary(source: String(contentsOf: kernelDirectory.appendingPathComponent(name), encoding: .utf8), options: options)
        }
        let dense = try library("bf16.metal"), branch = try library("expert.metal")
        func pipeline(_ lib: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else { throw ExpertCheckError.failed(name) }
            return try device.makeComputePipelineState(function: fn)
        }
        let vector = try pipeline(dense, "qwen35_bf16_projection")
        let matrix = try pipeline(dense, "qwen35_bf16_prefill_matrix")
        let activation = try pipeline(branch, "luban_expert_act"), addition = try pipeline(branch, "luban_expert_add")
        var checks: [[String: Any]] = []
        for (index, shape) in plan.enumerated() {
            let rows = shape["rows"]!, width = shape["width"]!, hidden = shape["hidden"]!
            let directory = root.appendingPathComponent(String(index))
            func read(_ key: String, _ count: Int) throws -> MTLBuffer {
                let data = try Data(contentsOf: directory.appendingPathComponent(key + ".raw"))
                try require(data.count == count * 2, "Invalid fixture size: " + key)
                return data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! }
            }
            func buffer(_ count: Int) -> MTLBuffer { device.makeBuffer(length: count * 2, options: .storageModeShared)! }
            let x = try read("x", rows * hidden), gateWeight = try read("gate.weight", width * hidden)
            let upWeight = try read("up.weight", width * hidden), downWeight = try read("down.weight", hidden * width)
            let base = try read("base", rows * hidden)
            let gate = buffer(rows * width), up = buffer(rows * width), act = buffer(rows * width), down = buffer(rows * hidden)
            func run(_ encode: (MTLComputeCommandEncoder) -> Void) throws {
                let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                encode(encoder); encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                try require(command.status == .completed, String(describing: command.error))
            }
            func project(_ input: MTLBuffer, _ weight: MTLBuffer, _ output: MTLBuffer, _ k: Int, _ n: Int) throws {
                try run { e in
                    let tiled = rows >= 8 && n >= 512
                    e.setComputePipelineState(tiled ? matrix : vector)
                    e.setBuffer(input, offset: 0, index: 0)
                    var dims = [UInt32(rows), UInt32(k), UInt32(n)]
                    e.setBytes(&dims, length: 12, index: 1)
                    e.setBuffer(weight, offset: 0, index: 2); e.setBuffer(output, offset: 0, index: 3)
                    let columns = tiled ? 64 : 8, tileRows = tiled ? 16 : 8
                    e.dispatchThreads(MTLSize(width: 256 * ((n + columns - 1) / columns), height: (rows + tileRows - 1) / tileRows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
            }
            func equal(_ key: String, _ buffer: MTLBuffer, _ count: Int) throws {
                let expected = try Data(contentsOf: directory.appendingPathComponent(key + ".raw"))
                try require(expected.count == count * 2, "Invalid expected fixture")
                let actual = Data(bytes: buffer.contents(), count: count * 2)
                let mismatches = zip(actual, expected).filter { $0 != $1 }.count
                checks.append(["case": index, "stage": key, "rows": rows, "width": width, "byte_mismatches": mismatches])
                try require(mismatches == 0, "Expert stage differs: case \(index) \(key), \(mismatches) bytes")
            }
            try project(x, gateWeight, gate, hidden, width); try equal("gate", gate, rows * width)
            try project(x, upWeight, up, hidden, width); try equal("up", up, rows * width)
            try run { e in
                e.setComputePipelineState(activation)
                for (i, b) in [gate, up, act].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                var count = UInt32(width); e.setBytes(&count, length: 4, index: 3)
                e.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            try equal("act", act, rows * width)
            try project(act, downWeight, down, width, hidden); try equal("expert", down, rows * hidden)
            try run { e in
                e.setComputePipelineState(addition); e.setBuffer(down, offset: 0, index: 0); e.setBuffer(base, offset: 0, index: 1)
                var count = UInt32(hidden); e.setBytes(&count, length: 4, index: 2)
                e.dispatchThreads(MTLSize(width: hidden, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            try equal("combined", base, rows * hidden)
        }
        let report: [String: Any] = ["schema": "native-q4-expert-small-graph-v1", "passed": true,
            "scope": "Synthetic sparse dyadic matrices, real BF16 boundaries and actual Metal projection/activation/add kernels; no full-model equivalence claim", "device": device.name, "cases": checks,
            "model_loaded": false, "teacher_calls": 0]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("metal-result.json"), options: .withoutOverwriting)
        FileHandle.standardOutput.write(data + Data([10]))
    }
}
