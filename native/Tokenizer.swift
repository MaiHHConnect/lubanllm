// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit

enum NativeTokenizerError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String { switch self { case .invalid(let message): return message } }
}

/// Qwen3.5's pinned NFC/byte-BPE tokenizer. No model weights or interpreter required.
final class QwenTokenizer {
    enum Compatibility { case transformersQwen2, tokenizerJSON }
    private struct Merge { let rank: Int; let token: Int }
    private struct Candidate { let rank: Int; let index: Int; let left: Int; let right: Int; let token: Int }
    private var vocabulary: [String: Int] = [:]
    private var tokenBytes: [Int: [UInt8]] = [:]
    private var added: [(text: String, id: Int)] = []
    private var specialIDs: Set<Int> = []
    private var merges: [UInt64: Merge] = [:]
    private var byteIDs = [Int](repeating: -1, count: 256)
    private let regex: NSRegularExpression
    private let specialRegex: NSRegularExpression
    private let templateVerified: Bool
    static let templateSHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"

    init(modelDirectory: URL, compatibility: Compatibility = .transformersQwen2) throws {
        func read(_ name: String) throws -> [String: Any] {
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: modelDirectory.appendingPathComponent(name))) as? [String: Any] else {
                throw NativeTokenizerError.invalid("Invalid \(name)")
            }
            return object
        }
        let json = try read("tokenizer.json")
        guard let model = json["model"] as? [String: Any], model["type"] as? String == "BPE",
              let vocab = model["vocab"] as? [String: Int], let mergeEntries = model["merges"] as? [Any],
              let normalizer = json["normalizer"] as? [String: Any], normalizer["type"] as? String == "NFC",
              let pre = json["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence",
              let parts = pre["pretokenizers"] as? [[String: Any]], parts.count == 2,
              parts[0]["type"] as? String == "Split", parts[0]["behavior"] as? String == "Isolated",
              parts[0]["invert"] as? Bool == false,
              let pattern = parts[0]["pattern"] as? [String: String], let expression = pattern["Regex"],
              parts[1]["type"] as? String == "ByteLevel", parts[1]["add_prefix_space"] as? Bool == false,
              parts[1]["use_regex"] as? Bool == false,
              model["dropout"] is NSNull, model["unk_token"] is NSNull,
              model["byte_fallback"] as? Bool == false, model["ignore_merges"] as? Bool == false,
              model["continuing_subword_prefix"] as? String == "", model["end_of_word_suffix"] as? String == "" else {
            throw NativeTokenizerError.invalid("Unsupported tokenizer configuration")
        }
        // Transformers 5.13.1 Qwen2Tokenizer rebuilds this regex instead of loading JSON's \p{M} variant.
        // Preserve that real product behavior; raw-json remains faithful to tokenizer.json.
        let hfExpression = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        regex = try NSRegularExpression(pattern: compatibility == .transformersQwen2 ? hfExpression : expression)
        vocabulary = vocab
        var entries = json["added_tokens"] as? [[String: Any]] ?? []
        let config = try read("tokenizer_config.json")
        if compatibility == .transformersQwen2, let extra = config["added_tokens_decoder"] as? [String: [String: Any]] {
            for (key, value) in extra where !entries.contains(where: { ($0["id"] as? Int) == Int(key) }) {
                var entry = value; entry["id"] = Int(key); entries.append(entry)
            }
        }
        for entry in entries {
            guard let text = entry["content"] as? String, !text.isEmpty, let id = entry["id"] as? Int,
                  entry["single_word"] as? Bool == false, entry["lstrip"] as? Bool == false,
                  entry["rstrip"] as? Bool == false, entry["normalized"] as? Bool == false else {
                throw NativeTokenizerError.invalid("Unsupported added token configuration")
            }
            added.append((text, id)); vocabulary[text] = id
            if entry["special"] as? Bool == true { specialIDs.insert(id) }
        }
        let escaped = added.sorted { $0.text.utf8.count > $1.text.utf8.count }.map { NSRegularExpression.escapedPattern(for: $0.text) }
        specialRegex = try NSRegularExpression(pattern: escaped.joined(separator: "|"))
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var codepoints = bytes; var next = 256
        for byte in 0...255 where !bytes.contains(byte) { bytes.append(byte); codepoints.append(next); next += 1 }
        var inverse: [Unicode.Scalar: UInt8] = [:]
        for (byte, scalar) in zip(bytes, codepoints) {
            let character = Unicode.Scalar(scalar)!
            inverse[character] = UInt8(byte)
            guard let id = vocab[String(character)] else { throw NativeTokenizerError.invalid("Missing byte token") }
            byteIDs[byte] = id
        }
        let addedIDs = Set(added.map(\.id))
        for (text, id) in vocabulary {
            if addedIDs.contains(id) { tokenBytes[id] = Array(text.utf8) }
            else {
                let raw = text.unicodeScalars.compactMap { inverse[$0] }
                guard raw.count == text.unicodeScalars.count else { throw NativeTokenizerError.invalid("Invalid byte vocabulary") }
                tokenBytes[id] = raw
            }
        }
        for (rank, entry) in mergeEntries.enumerated() {
            let pair: [String]
            if let text = entry as? String {
                pair = text.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
            } else if let tuple = entry as? [String], tuple.count == 2 {
                pair = tuple
            } else { throw NativeTokenizerError.invalid("Invalid BPE merge") }
            guard pair.count == 2, let left = vocab[String(pair[0])], let right = vocab[String(pair[1])],
                  let result = vocab[String(pair[0]) + String(pair[1])] else { throw NativeTokenizerError.invalid("Invalid BPE merge") }
            merges[Self.key(left, right)] = Merge(rank: rank, token: result)
        }
        let templateData = try Data(contentsOf: modelDirectory.appendingPathComponent("chat_template.jinja"))
        templateVerified = SHA256.hash(data: templateData).map { String(format: "%02x", $0) }.joined() == Self.templateSHA256
            && config["chat_template"] as? String == String(data: templateData, encoding: .utf8)
    }

    private static func key(_ left: Int, _ right: Int) -> UInt64 { UInt64(left) << 32 | UInt64(right) }

    private func bpe(_ bytes: [UInt8]) -> [Int] {
        guard bytes.count > 1 else { return bytes.map { byteIDs[Int($0)] } }
        var ids = bytes.map { byteIDs[Int($0)] }
        var previous = ids.indices.map { $0 - 1 }; var next = ids.indices.map { $0 + 1 }
        next[next.count - 1] = -1
        var alive = [Bool](repeating: true, count: ids.count)
        var heap: [Candidate] = []
        func less(_ a: Candidate, _ b: Candidate) -> Bool { a.rank == b.rank ? a.index < b.index : a.rank < b.rank }
        func push(_ index: Int) {
            guard index >= 0, alive[index], next[index] >= 0, let merge = merges[Self.key(ids[index], ids[next[index]])] else { return }
            heap.append(Candidate(rank: merge.rank, index: index, left: ids[index], right: ids[next[index]], token: merge.token))
            var at = heap.count - 1
            while at > 0 {
                let parent = (at - 1) / 2
                if !less(heap[at], heap[parent]) { break }; heap.swapAt(at, parent); at = parent
            }
        }
        func pop() -> Candidate? {
            guard !heap.isEmpty else { return nil }
            if heap.count == 1 { return heap.removeLast() }
            let result = heap[0]; heap[0] = heap.removeLast(); var at = 0
            while at * 2 + 1 < heap.count {
                var child = at * 2 + 1
                if child + 1 < heap.count && less(heap[child + 1], heap[child]) { child += 1 }
                if !less(heap[child], heap[at]) { break }; heap.swapAt(at, child); at = child
            }
            return result
        }
        for index in ids.indices { push(index) }
        while let pair = pop() {
            let left = pair.index; let right = next[left]
            guard alive[left], right >= 0, ids[left] == pair.left, ids[right] == pair.right else { continue }
            ids[left] = pair.token; alive[right] = false; next[left] = next[right]
            if next[left] >= 0 { previous[next[left]] = left }
            push(previous[left]); push(left)
        }
        return ids.indices.filter { alive[$0] }.map { ids[$0] }
    }

    func encode(_ text: String) throws -> [Int] {
        let source = text as NSString; var offset = 0; var output: [Int] = []
        func plain(_ text: String) throws {
            let normalized = text.precomposedStringWithCanonicalMapping as NSString
            let range = NSRange(location: 0, length: normalized.length)
            var end = 0
            for match in regex.matches(in: normalized as String, range: range) {
                guard match.range.location == end else { throw NativeTokenizerError.invalid("Pre-tokenizer failed to cover input") }
                output += bpe(Array(normalized.substring(with: match.range).utf8)); end = NSMaxRange(match.range)
            }
            guard end == normalized.length else { throw NativeTokenizerError.invalid("Pre-tokenizer failed to cover input") }
        }
        for match in specialRegex.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            try plain(source.substring(with: NSRange(location: offset, length: match.range.location - offset)))
            guard let id = vocabulary[source.substring(with: match.range)] else { throw NativeTokenizerError.invalid("Unknown added token") }
            output.append(id); offset = NSMaxRange(match.range)
        }
        try plain(source.substring(from: offset))
        return output
    }

    func bytes(for token: Int, skipSpecialTokens: Bool = true) throws -> [UInt8] {
        guard let bytes = tokenBytes[token] else { throw NativeTokenizerError.invalid("Unknown token ID \(token)") }
        return skipSpecialTokens && specialIDs.contains(token) ? [] : bytes
    }

    func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) throws -> String {
        String(decoding: try tokens.flatMap { try bytes(for: $0, skipSpecialTokens: skipSpecialTokens) }, as: UTF8.self)
    }

    /// A stable incremental decode: incomplete UTF-8 is retained until the next token or finish().
    final class StreamDecoder {
        private let tokenizer: QwenTokenizer
        private let skipSpecialTokens: Bool
        private var pending: [UInt8] = []
        init(_ tokenizer: QwenTokenizer, skipSpecialTokens: Bool = true) {
            self.tokenizer = tokenizer; self.skipSpecialTokens = skipSpecialTokens
        }
        func append(_ token: Int) throws -> String {
            pending += try tokenizer.bytes(for: token, skipSpecialTokens: skipSpecialTokens)
            var cut = pending.count
            // Only valid incomplete prefixes are buffered; invalid bytes are emitted as replacements.
            if !pending.isEmpty {
                var start = pending.count - 1
                while start > 0 && pending[start] & 0xC0 == 0x80 { start -= 1 }
                let lead = pending[start]
                let expected = lead >= 0xC2 && lead <= 0xDF ? 2 : lead >= 0xE0 && lead <= 0xEF ? 3 : lead >= 0xF0 && lead <= 0xF4 ? 4 : 0
                let suffix = pending.count - start
                var valid = expected > suffix
                if suffix > 1 {
                    let second = pending[start + 1]
                    valid = valid && second & 0xC0 == 0x80 && !(lead == 0xE0 && second < 0xA0)
                        && !(lead == 0xED && second >= 0xA0) && !(lead == 0xF0 && second < 0x90) && !(lead == 0xF4 && second >= 0x90)
                }
                if valid { cut = start }
            }
            let result = String(decoding: pending[..<cut], as: UTF8.self)
            pending.removeFirst(cut); return result
        }
        func finish() -> String {
            let result = String(decoding: pending, as: UTF8.self); pending.removeAll(); return result
        }
    }

    private static func trim(_ text: String) -> String {
        let scalars = text.unicodeScalars
        func whitespace(_ scalar: Unicode.Scalar) -> Bool { scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value) }
        let head = scalars.drop(while: whitespace)
        let tail = head.reversed().drop(while: whitespace).reversed()
        return String(String.UnicodeScalarView(tail))
    }

    /// Product text-chat subset of the pinned Jinja template. Unsupported features fail explicitly.
    func chatTemplate(_ messages: [[String: Any]], addGenerationPrompt: Bool = true, enableThinking: Bool = false) throws -> String {
        guard templateVerified else { throw NativeTokenizerError.invalid("Chat template differs from verified Qwen3.5 template") }
        guard !messages.isEmpty else { throw NativeTokenizerError.invalid("No messages provided") }
        func content(_ message: [String: Any]) throws -> String {
            guard message["tool_calls"] == nil || (message["tool_calls"] as? [Any])?.isEmpty == true else { throw NativeTokenizerError.invalid("tool_calls are not supported") }
            if message["content"] == nil || message["content"] is NSNull { return "" }
            guard let text = message["content"] as? String else { throw NativeTokenizerError.invalid("Only string chat content is supported; vision/text arrays require the multimodal processor") }
            return Self.trim(text)
        }
        let contents = try messages.map(content)
        guard let lastQuery = messages.indices.reversed().first(where: {
            messages[$0]["role"] as? String == "user" && !(contents[$0].hasPrefix("<tool_response>") && contents[$0].hasSuffix("</tool_response>"))
        }) else { throw NativeTokenizerError.invalid("No user query found in messages") }
        var output = ""
        for (index, message) in messages.enumerated() {
            guard let role = message["role"] as? String else { throw NativeTokenizerError.invalid("Missing message role") }
            var text = contents[index]
            switch role {
            case "system":
                guard index == 0 else { throw NativeTokenizerError.invalid("System message must be at the beginning") }
                output += "<|im_start|>system\n\(text)<|im_end|>\n"
            case "user": output += "<|im_start|>user\n\(text)<|im_end|>\n"
            case "assistant":
                var reasoning = message["reasoning_content"] as? String ?? ""
                if message["reasoning_content"] as? String == nil, text.contains("</think>") {
                    let splits = text.components(separatedBy: "</think>")
                    reasoning = splits[0].trimmingCharacters(in: CharacterSet(charactersIn: "\n")).components(separatedBy: "<think>").last ?? ""
                    text = (splits.last ?? "").drop(while: { $0 == "\n" }).description
                }
                output += "<|im_start|>assistant\n"
                if index > lastQuery { output += "<think>\n\(Self.trim(reasoning))\n</think>\n\n" }
                output += "\(text)<|im_end|>\n"
            case "tool":
                // The upstream template does not open a user block for a first-position tool.
                if index > 0 && messages[index - 1]["role"] as? String != "tool" { output += "<|im_start|>user" }
                output += "\n<tool_response>\n\(text)\n</tool_response>"
                if index + 1 == messages.count || messages[index + 1]["role"] as? String != "tool" { output += "<|im_end|>\n" }
            default: throw NativeTokenizerError.invalid("Unexpected message role \(role)")
            }
        }
        if addGenerationPrompt {
            output += "<|im_start|>assistant\n<think>\n"
            if !enableThinking { output += "\n</think>\n\n" }
        }
        return output
    }
}
