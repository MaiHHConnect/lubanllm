import Foundation
let languages = ["ar":"Arabic","yue":"Cantonese","zh":"Chinese","cs":"Czech","da":"Danish","nl":"Dutch","en":"English","fil":"Filipino","fi":"Finnish","fr":"French","de":"German","el":"Greek","hi":"Hindi","hu":"Hungarian","id":"Indonesian","it":"Italian","ja":"Japanese","ko":"Korean","mk":"Macedonian","ms":"Malay","fa":"Persian","pl":"Polish","pt":"Portuguese","ro":"Romanian","ru":"Russian","es":"Spanish","sv":"Swedish","th":"Thai","tr":"Turkish","vi":"Vietnamese"]
func language(_ value: Any?) throws -> String? {
    if value == nil || value is NSNull { return nil }
    guard let text = value as? String else { throw NativeTokenizerError.invalid("语种必须是语言代码或名称") }
    if let canonical = languages[text.lowercased()] { return canonical }
    if let canonical = languages.values.first(where: { $0.lowercased() == text.lowercased() }) { return canonical }
    throw NativeTokenizerError.invalid("Unsupported language")
}
do {
    guard CommandLine.arguments.count == 2 else { throw NativeTokenizerError.invalid("Model directory required") }
    let tokenizer = try QwenTokenizer(modelDirectory: URL(fileURLWithPath: CommandLine.arguments[1]))
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard let cases = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { throw NativeTokenizerError.invalid("Cases required") }
    var rows: [[String: Any]] = []
    for item in cases {
        do {
            var text = item["text"] as? String
            var canonical: String? = nil
            if item["mode"] as? String == "prompt" {
                canonical = try language(item["language"])
                guard let tokens = item["audio_tokens"] as? Int, tokens > 0, tokens <= 390 else { throw NativeTokenizerError.invalid("Audio tokens out of bounds") }
                text = "<|im_start|>system\n" + (canonical ?? "") + "<|im_end|>\n<|im_start|>user\n<|audio_start|>" + String(repeating: "<|audio_pad|>", count: tokens) + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
            }
            let ids: [Int]
            if let supplied = item["ids"] as? [Int] { ids = supplied }
            else if let value = text { ids = try tokenizer.encode(value) }
            else { throw NativeTokenizerError.invalid("Text or IDs required") }
            let stream = QwenTokenizer.StreamDecoder(tokenizer)
            var streamed = ""
            for token in ids { streamed += try stream.append(token) }
            streamed += stream.finish()
            rows.append(["ok":true,"language":canonical as Any? ?? NSNull(),"text":text as Any? ?? NSNull(),"ids":ids,"decode":try tokenizer.decode(ids, skipSpecialTokens:false),"skip_special":try tokenizer.decode(ids),"stream_skip_special":streamed])
        } catch { rows.append(["ok":false,"error":String(describing:error)]) }
    }
    let output = try JSONSerialization.data(withJSONObject: rows, options:[.sortedKeys])
    FileHandle.standardOutput.write(output)
} catch { FileHandle.standardError.write(Data((String(describing:error)+"\n").utf8)); exit(2) }
