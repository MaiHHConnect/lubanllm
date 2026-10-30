import Foundation
import CryptoKit
import MLX
import MLXLMCommon

// Persists exact-boundary prefix snapshots so a restarted worker or an unloaded model need not
// recompute them. Entries are keyed by a hash of the token prefix and a fingerprint of the model,
// core, expert, worker binary and prefill step. Any failure is a cold prefill; unverified state is
// never used. Files hold hidden state derived from the user's conversation: private, no backup.
final class DiskPrefixCache: @unchecked Sendable {
    static let version = "luban-prefix-cache-v1"
    static let minimumTokens = 1024
    private let directory: URL
    private let fingerprint: String
    private let byteLimit: Int64 = 8 * 1024 * 1024 * 1024
    private let freeReserve: Int64 = 10 * 1024 * 1024 * 1024

    // LUBAN_PREFIX_CACHE_DIR=off disables it; a path overrides the default location.
    static func make(parts: [String]) -> DiskPrefixCache? {
        let configured = ProcessInfo.processInfo.environment["LUBAN_PREFIX_CACHE_DIR"]
        if configured == "off" { return nil }
        guard let base = configured.map({ URL(fileURLWithPath: $0, isDirectory: true) })
                ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
                    .appendingPathComponent("Luban/prefix-cache", isDirectory: true),
              let binary = Bundle.main.executableURL, let digest = fileDigest(binary) else { return nil }
        do {
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            var excluded = URLResourceValues(); excluded.isExcludedFromBackup = true
            var mutable = base; try mutable.setResourceValues(excluded)
        } catch { return nil }
        let joined = ([version, digest] + parts).joined(separator: "\n")
        return DiskPrefixCache(directory: base, fingerprint: SHA256.hash(data: Data(joined.utf8)).hex)
    }

    private init(directory: URL, fingerprint: String) { self.directory = directory; self.fingerprint = fingerprint }

    private static func fileDigest(_ url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try? handle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().hex
    }

    private func key(_ tokens: ArraySlice<Int>, step: Int) -> String {
        var hasher = SHA256()
        hasher.update(data: Data("\(fingerprint)|\(step)|\(tokens.count)".utf8))
        tokens.map { Int32(truncatingIfNeeded: $0) }.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return String(hasher.finalize().hex.prefix(32))
    }

    private func entries() -> [(url: URL, count: Int, key: String, payload: String, bytes: Int64, used: Date)] {
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys)) ?? []
        return urls.compactMap { url in
            let parts = url.deletingPathExtension().lastPathComponent.split(separator: "-").map(String.init)
            guard url.pathExtension == "safetensors", parts.count == 3, let count = Int(parts[0]),
                  let values = try? url.resourceValues(forKeys: Set(keys)) else { return nil }
            return (url, count, parts[1], parts[2], Int64(values.fileSize ?? 0), values.contentModificationDate ?? .distantPast)
        }
    }

    // Called while `layers` holds exactly `count` prompt tokens (the prefill checkpoint).
    func save(_ prompt: [Int], count: Int, layers: [KVCache], prefillStep: Int) {
        guard count >= Self.minimumTokens, count < prompt.count else { return }
        let id = key(prompt.prefix(count), step: prefillStep)
        if let existing = entries().first(where: { $0.count == count && $0.key == id }) {
            try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: existing.url.path)
            return
        }
        var arrays = [String: MLXArray](), metadata = ["version": Self.version, "fingerprint": fingerprint,
                                                      "step": "\(prefillStep)", "count": "\(count)", "layers": "\(layers.count)"]
        for (index, layer) in layers.enumerated() {
            if let attention = layer as? KVCacheSimple {
                let state = attention.state
                guard state.count == 2, attention.offset == count else { return }
                arrays["l\(index).k"] = state[0]; arrays["l\(index).v"] = state[1]; metadata["l\(index)"] = "kv"
            } else if let recurrent = layer as? ArraysCache {
                guard let first = recurrent[0], let second = recurrent[1] else { return }
                arrays["l\(index).s0"] = first; arrays["l\(index).s1"] = second
                metadata["l\(index)"] = "arrays"; metadata["l\(index).offset"] = "\(recurrent.offset)"
            } else { return }
        }
        let temporary = directory.appendingPathComponent("tmp-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: temporary) }
        do {
            let free = try directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
                .volumeAvailableCapacityForImportantUsage ?? 0
            let estimate = arrays.values.reduce(Int64(0)) { $0 + Int64($1.nbytes) }
            guard free > estimate + freeReserve else { return }
            try MLX.save(arrays: arrays, metadata: metadata, url: temporary)
            guard let payload = Self.fileDigest(temporary) else { return }
            let final = directory.appendingPathComponent("\(count)-\(id)-\(payload.prefix(32)).safetensors")
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temporary.path)
            _ = try FileManager.default.replaceItemAt(final, withItemAt: temporary)
            evict()
        } catch { return }
    }

    private func evict() {
        // Leftovers of an interrupted write are never valid entries.
        for url in (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        where url.lastPathComponent.hasPrefix("tmp-") {
            let age = -((try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)?.timeIntervalSinceNow ?? 7200)
            if age > 3600 { try? FileManager.default.removeItem(at: url) }
        }
        var all = entries().sorted { $0.used > $1.used }
        var total = all.reduce(Int64(0)) { $0 + $1.bytes }
        while total > byteLimit, let oldest = all.popLast() {
            try? FileManager.default.removeItem(at: oldest.url); total -= oldest.bytes
        }
    }

    // Longest stored boundary that is a strict prefix of `prompt`, or nil.
    func find(_ prompt: [Int], prefillStep: Int, make: () throws -> [KVCache]) -> (PrefixCache.Working, Int)? {
        let candidates = entries().filter { $0.count < prompt.count }.sorted { $0.count > $1.count }
        for entry in candidates where entry.key == key(prompt.prefix(entry.count), step: prefillStep) {
            if let layers = load(entry.url, payload: entry.payload, count: entry.count, step: prefillStep, make: make) {
                try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: entry.url.path)
                return (PrefixCache.Working(layers), entry.count)
            }
            try? FileManager.default.removeItem(at: entry.url)
        }
        return nil
    }

    private func load(_ url: URL, payload: String, count: Int, step: Int, make: () throws -> [KVCache]) -> [KVCache]? {
        guard Self.fileDigest(url).map({ $0.hasPrefix(payload) }) == true,
              let (arrays, metadata) = try? loadArraysAndMetadata(url: url),
              metadata["version"] == Self.version, metadata["fingerprint"] == fingerprint,
              metadata["step"] == "\(step)", metadata["count"] == "\(count)",
              let layers = try? make(), metadata["layers"] == "\(layers.count)" else { return nil }
        for (index, layer) in layers.enumerated() {
            if let attention = layer as? KVCacheSimple {
                guard metadata["l\(index)"] == "kv", let keys = arrays["l\(index).k"], let values = arrays["l\(index).v"],
                      keys.dim(2) == count, values.dim(2) == count else { return nil }
                attention.state = [keys, values]
            } else if let recurrent = layer as? ArraysCache {
                guard metadata["l\(index)"] == "arrays", let first = arrays["l\(index).s0"], let second = arrays["l\(index).s1"],
                      let offset = metadata["l\(index).offset"].flatMap(Int.init) else { return nil }
                recurrent[0] = first; recurrent[1] = second; recurrent.offset = offset
            } else { return nil }
        }
        return layers
    }
}

private extension Digest {
    var hex: String { map { String(format: "%02x", $0) }.joined() }
}
