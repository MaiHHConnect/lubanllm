// SPDX-License-Identifier: Apache-2.0
// Independent CPU/GPU check for the exact research residual shader.
import Foundation
import Metal

enum CheckError: Error { case failed(String) }
func require(_ ok: Bool, _ message: String) throws { if !ok { throw CheckError.failed(message) } }
func bf16(_ x: Float) -> UInt16 {
    let u = x.bitPattern
    return UInt16(truncatingIfNeeded: (u &+ 0x7fff &+ ((u >> 16) & 1)) >> 16)
}
func fp32(_ x: UInt16) -> Float { Float(bitPattern: UInt32(x) << 16) }
guard [2, 4].contains(CommandLine.arguments.count), let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else { fatalError("usage: residual-check residual.metal [FACTOR_DIRECTORY REAL_X_BF16]") }
let options = MTLCompileOptions()
options.languageVersion = .version4_0
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
let source = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let library = try device.makeLibrary(source: source, options: options)
let project = try device.makeComputePipelineState(function: library.makeFunction(name: "qwen35_residual_project")!)
let add = try device.makeComputePipelineState(function: library.makeFunction(name: "qwen35_residual_add")!)
try require(project.threadExecutionWidth == 32, "SIMD width")
let padding = 32
func buffer(_ bytes: Int) -> MTLBuffer {
    let b = device.makeBuffer(length: bytes + padding * 2, options: .storageModeShared)!
    memset(b.contents(), 0xa5, b.length)
    return b
}
func data<T>(_ b: MTLBuffer, _ type: T.Type) -> UnsafeMutablePointer<T> {
    b.contents().advanced(by: padding).assumingMemoryBound(to: type)
}
func guards(_ b: MTLBuffer, _ bytes: Int) throws {
    let p = b.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0..<padding { try require(p[i] == 0xa5 && p[padding + bytes + i] == 0xa5, "buffer guard changed") }
}
var results = [[String: Any]]()
var cases = [(Int, Int, Int, Int, Bool, Bool, Bool)]()
for (k, n) in [(2048, 6144), (6144, 2048)] {
    for rows in [1, 9] { for rank in [8, 32] { for zero in [false, true] {
        cases.append((k, n, rows, rank, zero, false, false))
    } } }
    // Current packed-v3 column correction: real gate/up and down dimensions,
    // nonzero buffer offsets, partial rank, and prefill/decode row boundaries.
    for rows in [1, 2, 8, 9] { for zero in [false, true] {
        cases.append((k, n, rows, 8, zero, false, true))
    } }
}
if CommandLine.arguments.count == 4 {
    for rank in [8, 32] { cases.append((2048, 6144, 9, rank, false, true, false)) }
}
for (k, n, rows, rank, zero, real, onehot) in cases {
        let x = buffer(rows * k * 2), b = buffer(rank * k * 2), u = buffer(n * rank * 2)
        let z = buffer(rows * rank * 4), y = buffer(rows * n * 2)
        let xp = data(x, UInt16.self), bp = data(b, UInt16.self), up = data(u, UInt16.self)
        let zp = data(z, Float.self), yp = data(y, UInt16.self)
        // Dyadic fixtures exercise signs, cancellation, rank layouts and row offsets;
        // products remain exactly representable for an independent summation order.
        for i in 0..<(rows * k) { xp[i] = bf16(Float((i * 17 + i / k * 7) % 31 - 15) / 64) }
        for i in 0..<(rank * k) { bp[i] = bf16(Float((i * 23 + i / k * 11) % 29 - 14) / 128) }
        for i in 0..<(n * rank) { up[i] = bf16(zero ? 0 : Float((i * 7 + i / rank * 3) % 19 - 9) / 256) }
        let columns = [0, k - 1, 31, 32]
        if onehot {
            memset(bp, 0, rank * k * 2)
            for (r, column) in columns.enumerated() { bp[r * k + column] = bf16(1) }
            for i in 0..<(n * rank) where i % rank >= columns.count { up[i] = 0 }
        }
        if real {
            let factors = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2] + "/rank\(rank).bin"))
            let inputs = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3]))
            try require(inputs.count == rows * k * 2, "real input length")
            try require(factors.count >= 140 + (rank * k + n * rank) * 2, "real factor length")
            _ = inputs.withUnsafeBytes { memcpy(xp, $0.baseAddress!, inputs.count) }
            _ = factors.withUnsafeBytes { memcpy(bp, $0.baseAddress!.advanced(by: 140), rank * k * 2) }
            _ = factors.withUnsafeBytes { memcpy(up, $0.baseAddress!.advanced(by: 140 + rank * k * 2), n * rank * 2) }
        }
        let baseline: [UInt16] = (0..<(rows * n)).map { bf16(Float(($0 * 13) % 101 - 50) / 32) }
        for i in baseline.indices { yp[i] = baseline[i] }
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        var dims = [UInt32(rows), UInt32(k), UInt32(n), UInt32(rank)]
        enc.setComputePipelineState(project)
        enc.setBuffer(x, offset: padding, index: 0); enc.setBuffer(b, offset: padding, index: 1)
        enc.setBytes(&dims, length: 16, index: 2); enc.setBuffer(z, offset: padding, index: 3)
        enc.dispatchThreads(MTLSize(width: rank * 32, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.setComputePipelineState(add)
        enc.setBuffer(z, offset: padding, index: 0); enc.setBuffer(u, offset: padding, index: 1)
        enc.setBytes(&dims, length: 16, index: 2); enc.setBuffer(y, offset: padding, index: 3)
        enc.dispatchThreads(MTLSize(width: n, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        try require(cmd.status == .completed, String(describing: cmd.error))
        var maxZError: Float = 0, maxYError: Float = 0
        var referenceZ = [Float](repeating: 0, count: rows * rank)
        for row in 0..<rows { for r in 0..<rank {
            var sum: Float = 0, absoluteSum: Float = 0
            for j in 0..<k {
                let product = fp32(xp[row * k + j]) * fp32(bp[r * k + j])
                sum += product; absoluteSum += abs(product)
            }
            referenceZ[row * rank + r] = sum
            if onehot {
                let expected = r < columns.count ? fp32(xp[row * k + columns[r]]) : 0
                try require(sum == expected && zp[row * rank + r] == expected, "onehot gather mismatch")
            }
            let error = abs(sum - zp[row * rank + r]); maxZError = max(maxZError, error)
            let tolerance: Float = real ? 1e-5 * absoluteSum + 1e-7 : 0
            try require(error <= tolerance, "FP32 projection mismatch k=\(k) row=\(row) rank=\(r)")
        } }
        for row in 0..<rows { for col in 0..<n {
            var correction: Float = 0, absoluteSum: Float = 0
            for r in 0..<rank {
                let product = referenceZ[row * rank + r] * fp32(up[col * rank + r])
                correction += product; absoluteSum += abs(product)
            }
            let i = row * n + col
            let expected = bf16(fp32(baseline[i]) + correction)
            let error = abs(fp32(yp[i]) - fp32(expected)); maxYError = max(maxYError, error)
            let exponent = (UInt32(expected) >> 7) & 255
            let ulp: Float = exponent == 0 ? Float.leastNonzeroMagnitude * 65536 : pow(2, Float(Int(exponent) - 127 - 7))
            let tolerance: Float = real ? ulp + 1e-5 * absoluteSum : 0
            try require(error <= tolerance, "BF16 correction mismatch \(i) error=\(error) tolerance=\(tolerance)")
            if zero { try require(yp[i] == baseline[i], "zero residual changed baseline") }
        } }
        try guards(z, rows * rank * 4); try guards(y, rows * n * 2)
        results.append(["k": k, "n": n, "rows": rows, "rank": rank, "zero": zero,
                        "fp32_intermediates": rows * rank, "bf16_outputs": rows * n, "passed": true,
                        "real_factors_and_input": real, "onehot_column_protection": onehot,
                        "active_basis_columns": onehot ? columns : [], "max_fp32_error": maxZError, "max_bf16_error": maxYError])
}
let output: [String: Any] = ["device": device.name, "seed": "deterministic dyadic fixtures; optional real factors and captured input",
                            "real_tolerance": "Z:1e-5*sum(abs(products))+1e-7; Y:one BF16 ULP+1e-5*sum(abs(products))", "cases": results,
                            "passed": true, "scope": "residual operator only; real-model generation qualified separately"]
let json = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
print(String(decoding: json, as: UTF8.self))
