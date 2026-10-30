// SPDX-License-Identifier: Apache-2.0
// Exercise the exact GPU reduction on synthetic BF16 logits, including bad values.
import Foundation
import Metal
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.languageVersion = .version4_0
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
let source = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let library = try device.makeLibrary(source: source, options: options)
let first = try device.makeComputePipelineState(function: library.makeFunction(name: "qwen35_greedy_partial")!)
let last = try device.makeComputePipelineState(function: library.makeFunction(name: "qwen35_greedy_finish")!)
let count = 248320
let groups = (count + 255) / 256
let logits = device.makeBuffer(length: count * 2, options: .storageModeShared)!
let partial = device.makeBuffer(length: groups * 8, options: .storageModeShared)!
let result = device.makeBuffer(length: 4, options: .storageModeShared)!
let words = logits.contents().bindMemory(to: UInt16.self, capacity: count)
func gpu(_ length: Int) throws -> UInt32 {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    var n = UInt32(length), g = UInt32((length + 255) / 256)
    encoder.setComputePipelineState(first)
    encoder.setBuffer(logits, offset: 0, index: 0)
    encoder.setBytes(&n, length: 4, index: 1)
    encoder.setBuffer(partial, offset: 0, index: 2)
    encoder.dispatchThreads(MTLSize(width: Int(g) * 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.setComputePipelineState(last)
    encoder.setBuffer(partial, offset: 0, index: 0)
    encoder.setBytes(&g, length: 4, index: 1)
    encoder.setBuffer(result, offset: 0, index: 2)
    encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else { throw command.error! }
    return result.contents().load(as: UInt32.self)
}
func reference(_ length: Int) -> UInt32 {
    var best = 0
    for i in 0..<length {
        let value = Float(bitPattern: UInt32(words[i]) << 16)
        if !value.isFinite { return UInt32.max }
        if value > Float(bitPattern: UInt32(words[best]) << 16) { best = i }
    }
    return UInt32(best)
}
var checked = 0
for n in [1, 17, 255, 256, 257, 1023, count] {
    for round in 0..<8 {
        for i in 0..<n { words[i] = UInt16(truncatingIfNeeded: (Float((i * 37 + round * 71) % 1001 - 500) / 8).bitPattern >> 16) }
        if round == 0 { for i in 0..<n { words[i] = i % 2 == 0 ? 0x8000 : 0 } }
        if round == 1 { words[n-1] = 0x7fc1 }
        if round == 2 { words[n/2] = 0xff80 }
        if round == 3 { words[0] = 0x7f80 }
        let actual = try gpu(n)
        precondition(actual == reference(n), "GPU greedy result mismatch n=\(n) round=\(round)")
        checked += 1
    }
}
print("{\"passed\":true,\"cases\":\(checked),\"vocabulary\":\(count),\"weights_loaded\":false}")
