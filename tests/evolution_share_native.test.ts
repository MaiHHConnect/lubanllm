import { expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

test("Mac copies valid PNG to a private pasteboard, rejects malformed images, and decodes the poster QR", async () => {
  const source = await readFile(resolve(import.meta.dir, "../native/Luban.swift"), "utf8");
  const method = source.slice(source.indexOf("    func copyEvolutionImage("), source.indexOf("    func publishOperation("))
    .replaceAll("NSPasteboard.general", "pasteboard");
  expect(method.startsWith("    func copyEvolutionImage(")).toBe(true);
  const temporary = await mkdtemp(join(tmpdir(), "luban-share-test-"));
  try {
    const harness = `import AppKit
import Vision
class Web {
 var script = ""
 func evaluateJavaScript(_ value: String, completionHandler: Any?) { script = value }
}
class Copier {
 let web = Web()
 let pasteboard = NSPasteboard(name: NSPasteboard.Name("LubanShareTest-" + UUID().uuidString))
 ${method}
}
let copier = Copier()
defer { copier.pasteboard.releaseGlobally() }
let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
let id = UUID().uuidString
copier.copyEvolutionImage(["id": id, "png": data.base64EncodedString()])
precondition(copier.web.script.contains("\\\"success\\\":true"))
precondition(copier.pasteboard.data(forType: .png) == data)
let count = copier.pasteboard.changeCount
var largeDimensions = data
largeDimensions.replaceSubrange(16..<20, with: [0x7f,0xff,0xff,0xff])
for invalid in [Data("not png".utf8), largeDimensions, Data(repeating: 1, count: 3 * 1024 * 1024 + 1)] {
 copier.copyEvolutionImage(["id": id, "png": invalid.base64EncodedString()])
 precondition(copier.web.script.contains("\\\"success\\\":false"))
 precondition(copier.pasteboard.changeCount == count)
}
copier.web.script = ""
copier.copyEvolutionImage(["id": "invalid", "png": data.base64EncodedString()])
precondition(copier.web.script.isEmpty)
precondition(copier.pasteboard.changeCount == count)
let request = VNDetectBarcodesRequest()
try VNImageRequestHandler(data: data).perform([request])
precondition(request.results?.contains(where: { $0.payloadStringValue == "https://ai.maihh.net/" }) == true)
print("Private PNG pasteboard: valid copy + 4 rejected cases; QR: https://ai.maihh.net/")
`;
    await writeFile(join(temporary, "test.swift"), harness);
    const child = Bun.spawn(["/usr/bin/swift", join(temporary, "test.swift"), resolve(import.meta.dir, "../frontend/assets/share/evolution-success.png")], { stdout: "pipe", stderr: "pipe" });
    const [code, stdout, stderr] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()]);
    expect({ code, stderr }).toEqual({ code: 0, stderr: "" });
    expect(stdout).toContain("QR: https://ai.maihh.net/");
  } finally { await rm(temporary, { recursive: true, force: true }); }
}, 30000);
