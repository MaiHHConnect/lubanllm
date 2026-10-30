"""Run the actual Swift clipboard file reader without starting the desktop app."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class NativeTextAttachmentTests(unittest.TestCase):
    def test_media_rejected_before_read_and_text_document_bounds_preserved(self):
        source = Path("native/Luban.swift").read_text()
        start = source.index("                func appendFile(_ url: URL, name: String) throws {")
        end = source.index("                for url in urls {", start)
        actual = source[start:end]
        paste = source[source.index("    @objc func pasteChatAttachment"):source.index("    func chooseFeedingFiles")]
        self.assertNotIn("NSImage(pasteboard:", paste)
        self.assertNotIn("NSBitmapImageRep", paste)
        with tempfile.TemporaryDirectory(prefix="luban-native-text-") as temporary:
            directory = Path(temporary)
            main = directory / "main.swift"
            main.write_text('''import Foundation
import UniformTypeIdentifiers
var files: [[String: String]] = []
var total = 0
''' + actual + '''
let root = URL(fileURLWithPath: CommandLine.arguments[1])
var checks = 0
for name in ["sample.PNG", "sample.jpg", "sample.gif", "sample.WAV", "sample.mp3", "sample.ogg", "sample.aiff", "sample.svg"] {
    do {
        try appendFile(root.appendingPathComponent(name), name: name)
        fatalError("media accepted")
    } catch let error as NSError {
        precondition(error.domain == "Luban" && error.code == 4)
        checks += 1
    }
}
for name in ["notes.txt", "notes.md", "qa.csv", "qa.json", "qa.jsonl", "paper.docx", "paper.pdf"] {
    let url = root.appendingPathComponent(name)
    try Data("fixture text".utf8).write(to: url)
    try appendFile(url, name: name)
    precondition(files.last?["name"] == name && files.last?["base64"] == Data("fixture text".utf8).base64EncodedString())
    checks += 1
}
let linked = root.appendingPathComponent("linked.txt")
try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: root.appendingPathComponent("notes.txt"))
do { try appendFile(linked, name: "linked.txt"); fatalError("symlink accepted") }
catch let error as NSError { precondition(error.domain == "Luban" && error.code == 1); checks += 1 }
let large = root.appendingPathComponent("large.txt")
try Data(count: 8 * 1024 * 1024 + 1).write(to: large)
do { try appendFile(large, name: "large.txt"); fatalError("oversize accepted") }
catch let error as NSError { precondition(error.domain == "Luban" && error.code == 1); checks += 1 }
print("{\\"passed\\":true,\\"checks\\":\\(checks),\\"media_files_created\\":0,\\"model_loaded\\":false}")
''')
            sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            executable = directory / "guard"
            subprocess.run(["/Library/Developer/CommandLineTools/usr/bin/swiftc", "-sdk", sdk,
                            str(main), "-o", str(executable)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(executable), str(directory)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {"passed": True, "checks": 17,
                             "media_files_created": 0, "model_loaded": False})
