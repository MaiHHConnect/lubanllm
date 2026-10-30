// SPDX-License-Identifier: Apache-2.0
// Independent affine4 ABI check: FP32 dequantization, FP32 sums, BF16 output.
import Foundation
import Metal

enum CheckError: Error { case failed(String) }
func require(_ ok: Bool, _ message: String) throws {
    if !ok { throw CheckError.failed(message) }
}
func bf16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    return UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
}
func fp32(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
let arguments = CommandLine.arguments
let q8 = arguments.dropFirst(2).contains("--q8")
let g32 = arguments.dropFirst(2).contains("--g32")
let g16 = arguments.dropFirst(2).contains("--g16")
let groupSize = g16 ? 16 : (g32 ? 32 : 64)
let fixtureArguments = arguments.dropFirst(2).filter { $0.hasPrefix("--fixture=") }
let fixturePath = fixtureArguments.first.map { String($0.dropFirst("--fixture=".count)) }
let compileOnly = arguments.dropFirst(2).contains("--compile-only")
guard arguments.count >= 2 && arguments.dropFirst(2).allSatisfy({ ["--q8", "--g32", "--g16", "--compile-only"].contains($0) || $0.hasPrefix("--fixture=") }) else {
    fatalError("usage: mixed-check QWEN35_METAL_DIRECTORY [--q8 | --g32 | --g16] [--compile-only] [--fixture=DIR]")
}
try require([q8, g32, g16].filter { $0 }.count <= 1, "--q8, --g32 and --g16 are mutually exclusive")
try require(fixtureArguments.count <= 1 && (fixturePath == nil || ((g32 || g16) && !compileOnly)), "fixture requires --g32 or --g16 and cannot use --compile-only")
var fixtureShape: (m: Int, n: Int, k: Int)?
if let path = fixturePath {
    let bytes = try Data(contentsOf: URL(fileURLWithPath: path + "/shape.json"))
    guard let shape = try JSONSerialization.jsonObject(with: bytes) as? [String: Int],
          let m = shape["m"], let n = shape["n"], let k = shape["k"], shape["group_size"] == groupSize,
          m > 0 && m <= 4096, (n == 6144 && k == 2048) || (n == 2048 && k == 6144) else {
        throw CheckError.failed("invalid fixture shape: require selected group size and MLP dimensions, 1...4096 rows")
    }
    fixtureShape = (m, n, k)
}
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    throw CheckError.failed("Metal device unavailable")
}
let options = MTLCompileOptions()
options.languageVersion = .version4_0
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
let padding = 32
func buffer(_ bytes: Int) throws -> MTLBuffer {
    guard let value = device.makeBuffer(length: bytes + 2 * padding, options: .storageModeShared) else {
        throw CheckError.failed("allocation failed: \(bytes)")
    }
    memset(value.contents(), 0xA5, value.length)
    return value
}
func data<T>(_ buffer: MTLBuffer, _ type: T.Type) -> UnsafeMutablePointer<T> {
    buffer.contents().advanced(by: padding).assumingMemoryBound(to: T.self)
}
func guardBytes(_ buffer: MTLBuffer, _ bytes: Int) throws {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0..<padding {
        try require(p[i] == 0xA5 && p[padding + bytes + i] == 0xA5, "output guard overwritten")
    }
}
var seed: UInt32 = 42
func random() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
func sample(_ scale: Float) -> Float { Float(Int32(bitPattern: random())) / 2147483648 * scale }

func check(_ k: Int, _ n: Int, _ rows: Int, _ pipeline: MTLComputePipelineState,
           _ randomRows: Bool) throws -> [String: Any] {
    let groups = k / groupSize
    let wordsPerGroup = groupSize / 8
    let x = try buffer(rows * k * 2), w = try buffer(q8 ? n * k : n * k / 2)
    let scales = try buffer(n * groups * 2), biases = try buffer(n * groups * 2)
    let out = try buffer(rows * n * 2)
    let xp = data(x, UInt16.self), wp = data(w, UInt32.self)
    let wi = data(w, Int8.self)
    let sp = data(scales, UInt16.self), bp = data(biases, UInt16.self)
    let yp = data(out, UInt16.self)
    let selected = Array(Set([0, 1, n / 2, n - 2, n - 1] + (0..<12).map { ($0 * 383 + 7) % n })).sorted()
    if let path = fixturePath {
        for (name, target, count) in [("x.bf16", x, rows * k * 2), ("words.u32", w, n * k / 2),
                                      ("scales.bf16", scales, n * groups * 2), ("biases.bf16", biases, n * groups * 2)] {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: path + "/" + name))
            try require(bytes.count == count, "fixture \(name): expected \(count) bytes, got \(bytes.count)")
            bytes.withUnsafeBytes { raw in
                _ = memcpy(target.contents().advanced(by: padding), raw.baseAddress!, count)
            }
        }
        for i in 0..<(rows * k) { try require(fp32(xp[i]).isFinite, "nonfinite fixture input at \(i)") }
        for i in 0..<(n * groups) { try require(fp32(sp[i]).isFinite && fp32(bp[i]).isFinite, "nonfinite fixture affine parameter at \(i)") }
    } else {
        // Dyadic inputs keep the structured CPU reference exact under reassociation.
        for i in 0..<(rows * k) {
            xp[i] = bf16(randomRows ? sample(0.3) : Float((i * 13 + i / k * 5) % 17 - 8) / 16)
        }
        let selectedSet = Set(selected)
        let structuredWords: [UInt32] = (0..<128).map { index in
            var packed: UInt32 = 0
            for nibble in 0..<8 { packed |= UInt32((index / 8 + (index % 8) * 3 + nibble) % 16) << UInt32(4 * nibble) }
            return packed
        }
        let structuredQ8: [Int8] = (0..<1024).map { index in
            let row = index / 64, column = index % 64
            return Int8((row * 19 + column * 7) % 255 - 127)
        }
        for channel in 0..<n {
            let randomize = randomRows && selectedSet.contains(channel)
            for g in 0..<groups {
                let index = channel * groups + g
                sp[index] = bf16(randomize ? sample(0.08) : (g % 7 == 0 ? 0 : Float((channel % 5) - 2) / 64))
                bp[index] = bf16(randomize ? sample(0.2) : Float((g + channel) % 5 - 3) / 32)
                if q8 {
                    if g % 7 == 0 {
                        sp[index] = bf16(1)
                        memset(wi.advanced(by: index * 64), 0, 64)
                        continue
                    }
                    if randomize {
                        for j in 0..<64 { wi[index * 64 + j] = Int8(Int(random() >> 24) % 255 - 127) }
                    } else {
                        _ = structuredQ8.withUnsafeBufferPointer { memcpy(wi.advanced(by: index * 64), $0.baseAddress!.advanced(by: (channel % 16) * 64), 64) }
                    }
                    continue
                }
                for word in 0..<wordsPerGroup {
                    var packed = structuredWords[(channel % 16) * 8 + word]
                    if randomize {
                        packed = 0
                        for nibble in 0..<8 { packed |= (random() >> 28) << UInt32(4 * nibble) }
                    }
                    wp[index * wordsPerGroup + word] = packed
                }
            }
        }
    }
    let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(x, offset: padding, index: 0)
    var shape = [Int32(rows), Int32(k), Int32(n)]
    encoder.setBytes(&shape, length: q8 ? 12 : 8, index: 1)
    encoder.setBuffer(w, offset: padding, index: 2)
    encoder.setBuffer(scales, offset: padding, index: 3)
    if q8 {
        encoder.setBuffer(out, offset: padding, index: 4)
    } else {
        encoder.setBuffer(biases, offset: padding, index: 4)
        var one: Float = 1
        encoder.setBytes(&one, length: 4, index: 5)
        encoder.setBuffer(out, offset: padding, index: 6)
    }
    let matrix = q8 && rows >= 8 && n >= 512
    let columns = q8 ? (matrix ? 64 : 8) : (n == 16 ? 16 : 32)
    let tileRows = matrix ? 16 : 8
    encoder.dispatchThreads(MTLSize(width: 256 * ((n + columns - 1) / columns), height: (rows + tileRows - 1) / tileRows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try require(command.status == .completed, String(describing: command.error))
    try guardBytes(out, rows * n * 2)
    for i in 0..<(rows * n) { try require(fp32(yp[i]).isFinite, "nonfinite output at \(i)") }
    var maxError: Double = 0
    var compared = 0
    // Structured rows repeat every lcm(16, 5)=80 output channels; cache only CPU references.
    var references: [Int: (Float, Double)] = [:]
    for row in 0..<rows {
        for channel in randomRows ? selected : Array(0..<n) {
            let cacheKey = row * 80 + channel % 80
            let reference: (Float, Double)
            if !randomRows, let cached = references[cacheKey] {
                reference = cached
            } else {
                var sum: Float = 0, magnitude: Double = 0
                for j in 0..<k {
                    let index = channel * k + j
                    let code = q8 ? Float(wi[index]) : Float((wp[index / 8] >> UInt32(4 * (j % 8))) & 15)
                    // Do not cast dequantized weights to BF16: this kernel applies affine
                    // scale/bias to FP32 group dot products, rounding only the output.
                    let weight: Float = fp32(sp[index / groupSize]) * code + (q8 ? 0 : fp32(bp[index / groupSize]))
                    let product: Float = weight * fp32(xp[row * k + j])
                    sum += product
                    magnitude += abs(Double(product))
                }
                reference = (sum, magnitude)
                if !randomRows { references[cacheKey] = reference }
            }
            let actual = fp32(yp[row * n + channel]), expected = fp32(bf16(reference.0))
            let error = abs(Double(actual) - Double(expected))
            maxError = max(maxError, error)
            if randomRows {
                let exponent = floor(log2(max(abs(Double(expected)), 1e-30)))
                let tolerance = pow(2, exponent - 7) + 1e-5 * reference.1
                try require(error <= tolerance, "random M\(rows) N\(n) K\(k) [\(row),\(channel)] actual=\(actual) reference=\(expected) tolerance=\(tolerance)")
            } else {
                try require(yp[row * n + channel] == bf16(reference.0), "exact M\(rows) N\(n) K\(k) [\(row),\(channel)] actual=\(actual) reference=\(expected)")
            }
            compared += 1
        }
    }
    var embeddingValues = 0
    if q8 && rows == 1 && n == 248320 {
        let ids = try buffer(selected.count * 4), embedded = try buffer(selected.count * k * 2)
        let ip = data(ids, UInt32.self), ep = data(embedded, UInt16.self)
        for (i, id) in selected.enumerated() { ip[i] = UInt32(id) }
        let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(q8Pipelines["qwen35_q8_embed"]!)
        encoder.setBuffer(ids, offset: padding, index: 0)
        encoder.setBuffer(w, offset: padding, index: 1)
        encoder.setBuffer(scales, offset: padding, index: 2)
        encoder.setBuffer(embedded, offset: padding, index: 3)
        var dims = [UInt32(selected.count), UInt32(k)]
        encoder.setBytes(&dims, length: 8, index: 4)
        encoder.dispatchThreads(MTLSize(width: k, height: selected.count, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed, String(describing: command.error))
        try guardBytes(embedded, selected.count * k * 2)
        for (row, id) in selected.enumerated() {
            for j in 0..<k {
                let index = id * k + j
                try require(ep[row * k + j] == bf16(Float(wi[index]) * fp32(sp[index / 64])), "Q8 embedding mismatch [\(id),\(j)]")
                embeddingValues += 1
            }
        }
    }
    return ["m": rows, "n": n, "k": k, "random_selected_rows": randomRows && fixturePath == nil, "fixture": fixturePath as Any? ?? NSNull(),
            "selected_channels": randomRows ? selected : [], "group_size": groupSize,
            "threads_per_group": 256, "columns_per_group": columns, "rows_per_group": tileRows,
            "compared_outputs": compared, "all_outputs_finite": true, "output_guards_intact": true,
            "embedding_values_exact": embeddingValues,
            "exact_bf16_required": !randomRows, "max_absolute_error": maxError,
            "gpu_seconds": command.gpuEndTime - command.gpuStartTime]
}

var results: [[String: Any]] = []
var q8Pipelines: [String: MTLComputePipelineState] = [:]
if q8 {
    let source = try String(contentsOfFile: arguments[1] + "/q8.metal", encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: options)
    for name in ["qwen35_q8_projection", "qwen35_q8_prefill_matrix", "qwen35_q8_embed"] {
        guard let function = library.makeFunction(name: name) else { throw CheckError.failed("missing \(name)") }
        q8Pipelines[name] = try device.makeComputePipelineState(function: function)
    }
}
let shapes = fixtureShape.map { [($0.n, $0.k)] } ?? ((g32 || g16) ? [(6144, 2048), (2048, 6144)] :
    [(16, 2048), (512, 2048), (2048, 2048), (4096, 2048), (6144, 2048), (2048, 6144), (248320, 2048)])
for (n, k) in shapes {
    let key = "qmm_\(g16 ? "g16_" : (g32 ? "g32_" : ""))\(n)_\(k)"
    var affinePipeline: MTLComputePipelineState?
    if !q8 {
        let source = try String(contentsOfFile: arguments[1] + "/" + key + ".metal", encoding: .utf8)
        let library = try device.makeLibrary(source: source, options: options)
        let name = "custom_kernel_qwen35_\(key)_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t"
        guard let function = library.makeFunction(name: name) else { throw CheckError.failed("missing \(name)") }
        affinePipeline = try device.makeComputePipelineState(function: function)
    }
    if compileOnly { continue }
    for rows in fixtureShape.map({ [$0.m] }) ?? ((g32 || g16) ? [1, 2, 8, 9, 17, 33] : [1, 2, 8, 9]) {
        let pipeline = q8 ? q8Pipelines[rows >= 8 && n >= 512 ? "qwen35_q8_prefill_matrix" : "qwen35_q8_projection"]! : affinePipeline!
        try require(pipeline.threadExecutionWidth == 32, "SIMD width must be 32")
        for randomRows in fixturePath == nil ? [false, true] : [true] {
            results.append(try autoreleasepool { try check(k, n, rows, pipeline, randomRows) })
        }
    }
}
let report: [String: Any] = ["device": device.name, "seed": 42, "language": "Metal 4.0",
                            "bits": q8 ? 8 : 4, "group_size": groupSize, "compile_only": compileOnly,
                            "math": "safe/fast", "reference": "CPU FP32 dequantization and accumulation; BF16 final output",
                            "random_tolerance": "one BF16 ULP of reference + 1e-5 * sum(abs(products))",
                            "scope": fixturePath == nil ? "synthetic kernel ABI, not model quantization quality or end-to-end speed" : "supplied real tensors GPU vs CPU dequantized reference; not end-to-end quality", "cases": results]
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
print(String(data: json, encoding: .utf8)!)
