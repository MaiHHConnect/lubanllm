import AppKit
import PDFKit

// A separate bounded process parses only explicitly selected source bytes.
// No shell, external resources, macros, or model/teacher calls are used.
let limit = 65536
func fail() -> Never { exit(1) }
let raw = FileHandle.standardInput.readData(ofLength: 8 * 1024 * 1024 + 1)
guard !raw.isEmpty, raw.count <= 8 * 1024 * 1024, CommandLine.arguments.count == 2 else { fail() }
let kind = CommandLine.arguments[1]
var text = ""
if kind == "pdf" {
    guard let pdf = PDFDocument(data: raw), !pdf.isLocked, pdf.pageCount <= 1000 else { fail() }
    for index in 0..<pdf.pageCount {
        text += (pdf.page(at: index)?.string ?? "") + "\n"
        guard text.utf8.count <= limit else { exit(2) }
    }
} else {
    let types: [String: NSAttributedString.DocumentType] = ["doc": .docFormat, "rtf": .rtf, "odt": .openDocument]
    guard let type = types[kind], let document = try? NSAttributedString(data: raw,
        options: [.documentType: type], documentAttributes: nil) else { fail() }
    text = document.string
}
guard text.utf8.count <= limit else { exit(2) }
guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      let output = text.data(using: .utf8) else { fail() }
FileHandle.standardOutput.write(output)
