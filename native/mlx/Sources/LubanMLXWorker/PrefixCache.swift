import Foundation
import MLX
import MLXLMCommon

// One worker owns one fixed Qwen35 text model/source and serves requests serially.
// Its recurrent layers cannot be rewound: reuse requires an exact saved boundary.
final class PrefixCache: @unchecked Sendable {
    // MLX progress callbacks execute inline on the same serial generation task.
    final class Working: @unchecked Sendable {
        var layers: [KVCache]
        init(_ layers: [KVCache]) { self.layers = layers }
    }
    private(set) var tokens = [Int]()
    private var layers = [KVCache]()
    private var offsets = [Int]()
    // Chunk changes can alter floating-point accumulation in recurrent state.
    private var prefillStep: Int?
    private(set) var bytes = 0
    private let byteLimit = Int(min(ProcessInfo.processInfo.physicalMemory / 8, 2 * 1024 * 1024 * 1024))

    func clear() {
        tokens = []; layers = []; offsets = []; prefillStep = nil; bytes = 0
    }

    func matches(_ prompt: [Int]) -> Bool {
        !tokens.isEmpty && tokens.count < prompt.count && prompt.starts(with: tokens)
    }

    func refreshBytes() {
        bytes = layers.flatMap { $0 is KVCacheSimple ? $0.innerState() : $0.state }.reduce(0) { $0 + $1.nbytes }
        if bytes > byteLimit { clear() }
    }

    func workingCopy(for prompt: [Int], enabled: Bool, prefillStep: Int) -> (Working?, Int) {
        guard enabled && self.prefillStep == prefillStep && matches(prompt) else { clear(); return (nil, 0) }
        guard zip(layers, offsets).allSatisfy({ layer, saved in
            !(layer is KVCacheSimple) || layer.offset >= saved
        }) else { clear(); return (nil, 0) }
        let working = zip(layers, offsets).map { layer, saved -> KVCache in
            if let attention = layer as? KVCacheSimple {
                attention.trim(attention.offset - saved)
                return attention
            }
            return layer.copy()
        }
        return (Working(working), tokens.count)
    }

    // Capture once per prefill, before the mutable assistant/thinking suffix.
    // The pinned Qwen35 text path has nil LMOutput.State; all neural state lives
    // in these attention + Mamba cache objects. Vision/MTP never enter this path.
    func capture(_ prompt: [Int], count: Int, cache: [KVCache], prefillStep: Int) {
        let size = cache.flatMap { $0 is KVCacheSimple ? $0.innerState() : $0.state }.reduce(0) { $0 + $1.nbytes }
        guard count > 0 && count < prompt.count && size <= byteLimit,
              cache.allSatisfy({ !($0 is KVCacheSimple) || $0.offset == count }) else { clear(); return }
        eval(cache)
        // Full attention is append-only and trimmable. Keep its sole owner;
        // copying an array view here forces full-buffer COW on the next write.
        // Recurrent state cannot be trimmed and still needs an exact snapshot.
        let snapshot = cache.map { $0 is KVCacheSimple ? $0 : $0.copy() }
        eval(snapshot)
        Stream.defaultStream.synchronize()
        tokens = Array(prompt.prefix(count)); offsets = cache.map { $0.offset }
        layers = snapshot; self.prefillStep = prefillStep; bytes = size
    }

    // Reserve the complete request before prefill/decode. Grow retained arrays
    // one layer at a time and evaluate before releasing each old allocation.
    // This avoids keeping all layers' old and new buffers during chunk growth.
    static func reserve(_ layers: [KVCache], tokens: Int) {
        for case let layer as KVCacheSimple in layers {
            reserve(layer, tokens: tokens)
        }
    }

    private static func reserve(_ layer: KVCacheSimple, tokens: Int) {
        let existing = layer.innerState()
        guard !existing.isEmpty else { layer.step = tokens; return }
        guard existing[0].dim(2) < tokens else { return }
        let offset = layer.offset
        let grown = layer.state.map { value -> MLXArray in
            var shape = value.shape
            shape[2] = tokens - offset
            return concatenated([value, MLXArray.zeros(shape, dtype: value.dtype)], axis: 2)
        }
        eval(grown)
        Stream.defaultStream.synchronize()
        layer.state = grown
        layer.trim(tokens - offset)
        layer.step = tokens
    }

    static func checkpoint(total: Int, prefill: PrefillParameters) -> Int {
        guard total > prefill.resolvedStepSize(),
              let chunk = prefill.chunkLength(forChunking: total - 1) else { return 0 }
        return max(0, (total - 64) / chunk * chunk)
    }
}
