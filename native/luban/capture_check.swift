// SPDX-License-Identifier: Apache-2.0
// Research copy layout fixture only. Never loads model weights.
import Foundation
import Metal

enum CaptureCheckError: Error { case failed(String) }
func require(_ value: Bool, _ message: String) throws {
    if !value { throw CaptureCheckError.failed(message) }
}
guard CommandLine.arguments.count == 2 else {
    fatalError("usage: capture-check /absolute/path/to/trace.metal")
}
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.languageVersion = .version4_0
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
let shader = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let library = try device.makeLibrary(source: shader, options: options)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "qwen35_research_trace_copy")!)
let stride = 32, rowValues = 190464, guardValues = 256
let count = stride * rowValues
let source = device.makeBuffer(length: (count + guardValues) * 2, options: .storageModeShared)!
let destination = device.makeBuffer(length: (count + guardValues) * 2, options: .storageModeShared)!
let input = source.contents().bindMemory(to: UInt16.self, capacity: count + guardValues)
let output = destination.contents().bindMemory(to: UInt16.self, capacity: count + guardValues)
for i in 0..<(count + guardValues) { input[i] = UInt16(i % 0x7f00) }
var cases: [[String: Any]] = []
for rows in [1, 9, 32] {
    for i in 0..<(count + guardValues) { output[i] = 0xffff }
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    for layer in 0..<24 {
        for down in [false, true] {
            if layer == 23 && down { continue }
            let width = down ? 6144 : 2048
            let offset = (layer * 8192 + (down ? 2048 : 0)) * stride
            var values = UInt32(rows * width)
            encoder.setBuffer(source, offset: offset * 2, index: 0)
            encoder.setBuffer(destination, offset: offset * 2, index: 1)
            encoder.setBytes(&values, length: 4, index: 2)
            encoder.dispatchThreads(MTLSize(width: Int(values), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try require(command.status == .completed, String(describing: command.error))
    for layer in 0..<24 {
        for down in [false, true] {
            if layer == 23 && down { continue }
            let width = down ? 6144 : 2048
            let offset = (layer * 8192 + (down ? 2048 : 0)) * stride
            for index in 0..<(stride * width) {
                let expected: UInt16 = index < rows * width ? input[offset + index] : 0xffff
                try require(output[offset + index] == expected, "copy/padding mismatch")
            }
        }
    }
    for i in 0..<(count + guardValues) {
        try require(input[i] == UInt16(i % 0x7f00), "input changed")
        if i >= count { try require(output[i] == 0xffff, "end guard changed") }
    }
    cases.append(["rows": rows, "streams": 47, "exact_values": rows * rowValues,
                  "unused_rows_unchanged": true, "source_unchanged": true, "guard_unchanged": true])
}
let result: [String: Any] = ["passed": true, "cases": cases, "model_loaded": false]
print(String(data: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), encoding: .utf8)!)
