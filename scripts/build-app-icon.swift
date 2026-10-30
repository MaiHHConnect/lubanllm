import AppKit

// Native macOS icon, derived from the website's nested-L / joinery mark.
// Regenerate without downloading assets or loading a model: swift scripts/build-app-icon.swift
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let assets = root.appendingPathComponent("native/assets", isDirectory: true)
let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
let iconset = temporary.appendingPathComponent("LubanIcon.iconset", isDirectory: true)
try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: temporary) }

let svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="red" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ed3b4c"/><stop offset="1" stop-color="#c7192b"/></linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="14" flood-color="#38131a" flood-opacity=".2"/></filter>
  </defs>
  <rect x="96" y="96" width="832" height="832" rx="188" fill="url(#red)" filter="url(#shadow)"/>
  <rect x="98" y="98" width="828" height="828" rx="186" fill="none" stroke="#fff" stroke-opacity=".16" stroke-width="3"/>
  <path d="M349.5 304V720H674.5M505.5 304V564H674.5" fill="none" stroke="#fff" stroke-width="78" stroke-linejoin="round"/>
</svg>
"""
try (svg + "\n").write(to: assets.appendingPathComponent("LubanIcon.svg"), atomically: true, encoding: .utf8)

func render(_ pixels: Int) throws -> Data {
    guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .calibratedRGB, bytesPerRow: pixels * 4, bitsPerPixel: 32),
        let graphics = NSGraphicsContext(bitmapImageRep: bitmap) else { throw CocoaError(.fileWriteUnknown) }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = graphics
    defer { NSGraphicsContext.restoreGraphicsState() }
    let context = graphics.cgContext
    context.translateBy(x: 0, y: CGFloat(pixels))
    context.scaleBy(x: CGFloat(pixels) / 1024, y: -CGFloat(pixels) / 1024)
    let tile = NSBezierPath(roundedRect: NSRect(x: 96, y: 96, width: 832, height: 832), xRadius: 188, yRadius: 188)
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: 12), blur: 28,
        color: NSColor(srgbRed: 0.22, green: 0.075, blue: 0.1, alpha: 0.2).cgColor)
    NSColor(srgbRed: 0.78, green: 0.098, blue: 0.169, alpha: 1).setFill()
    tile.fill()
    context.restoreGState()
    NSGradient(starting: NSColor(srgbRed: 237/255, green: 59/255, blue: 76/255, alpha: 1),
        ending: NSColor(srgbRed: 199/255, green: 25/255, blue: 43/255, alpha: 1))!.draw(in: tile, angle: 45)
    let edge = NSBezierPath(roundedRect: NSRect(x: 98, y: 98, width: 828, height: 828), xRadius: 186, yRadius: 186)
    NSColor.white.withAlphaComponent(0.16).setStroke()
    edge.lineWidth = 3
    edge.stroke()
    let mark = NSBezierPath()
    mark.move(to: NSPoint(x: 349.5, y: 304))
    mark.line(to: NSPoint(x: 349.5, y: 720))
    mark.line(to: NSPoint(x: 674.5, y: 720))
    mark.move(to: NSPoint(x: 505.5, y: 304))
    mark.line(to: NSPoint(x: 505.5, y: 564))
    mark.line(to: NSPoint(x: 674.5, y: 564))
    mark.lineWidth = 78
    mark.lineJoinStyle = .round
    NSColor.white.setStroke()
    mark.stroke()
    guard let png = bitmap.representation(using: .png, properties: [:]) else { throw CocoaError(.fileWriteUnknown) }
    return png
}

for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let png = try render(size * scale)
        let suffix = scale == 1 ? "" : "@2x"
        try png.write(to: iconset.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
        if size == 512 && scale == 2 { try png.write(to: assets.appendingPathComponent("LubanIcon.png")) }
    }
}
let conversion = Process()
conversion.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
conversion.arguments = ["-c", "icns", iconset.path, "-o", assets.appendingPathComponent("LubanIcon.icns").path]
try conversion.run()
conversion.waitUntilExit()
guard conversion.terminationStatus == 0 else { throw CocoaError(.fileWriteUnknown) }
print("Generated native/assets/LubanIcon.svg, LubanIcon.png and LubanIcon.icns")
