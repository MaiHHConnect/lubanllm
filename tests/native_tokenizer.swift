// JSONL CPU comparison harness; compile together with native/Tokenizer.swift.
import Foundation

@main struct NativeTokenizerTest {
    static func main() throws {
        guard CommandLine.arguments.count >= 2 else { throw NativeTokenizerError.invalid("Pass model directory") }
        let tokenizer = try QwenTokenizer(modelDirectory: URL(fileURLWithPath: CommandLine.arguments[1]),
                                          compatibility: CommandLine.arguments.contains("--raw-json") ? .tokenizerJSON : .transformersQwen2)
        while let line = readLine() {
            do {
                guard let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { throw NativeTokenizerError.invalid("Invalid request") }
                var response: [String: Any] = [:]
                if let messages = request["messages"] as? [[String: Any]] {
                    guard request["tools"] == nil else { throw NativeTokenizerError.invalid("Tool schemas are unsupported") }
                    let text = try tokenizer.chatTemplate(messages, addGenerationPrompt: request["add_generation_prompt"] as? Bool ?? true,
                                                          enableThinking: request["enable_thinking"] as? Bool ?? false)
                    response["prompt"] = text; response["ids"] = try tokenizer.encode(text)
                } else if let text = request["text"] as? String { response["ids"] = try tokenizer.encode(text) }
                if let ids = request["ids"] as? [Int] {
                    let skip = request["skip_special_tokens"] as? Bool ?? true
                    response["decoded"] = try tokenizer.decode(ids, skipSpecialTokens: skip)
                    let stream = QwenTokenizer.StreamDecoder(tokenizer, skipSpecialTokens: skip)
                    response["chunks"] = try ids.map { try stream.append($0) } + [stream.finish()]
                }
                let data = try JSONSerialization.data(withJSONObject: response, options: [.sortedKeys, .fragmentsAllowed])
                print(String(decoding: data, as: UTF8.self))
            } catch {
                let data = try JSONSerialization.data(withJSONObject: ["error": String(describing: error)], options: [.sortedKeys])
                print(String(decoding: data, as: UTF8.self))
            }
        }
    }
}
