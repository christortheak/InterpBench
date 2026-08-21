// SteerLab app icon generator — reproducible source for SteerLab.icns.
//
// Design: the residual stream as horizontal flow lines inside the standard
// macOS squircle; one line is deflected upward at an injection point, marked
// by a small vector arrow — extraction/injection in one glyph. Deep indigo
// field, cyan-white steered line, muted stream lines. No text (illegible at
// 16 px, and the bundle name already says SteerLab).
//
// Regenerate with:
//   swiftc -O scripts/app-bundle/make-icon.swift -o /tmp/make-icon
//   /tmp/make-icon /tmp/steerlab-icon-1024.png
//   then the iconset/iconutil steps in scripts/app-bundle/README-icon.md
//
// Plain AppKit/CoreGraphics — no Metal, safe under `swiftc` with no package.

import AppKit

let size: CGFloat = 1024
guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: make-icon <out.png>\n".utf8))
    exit(64)
}
let outPath = CommandLine.arguments[1]

let image = NSImage(size: NSSize(width: size, height: size))
image.lockFocus()

guard let ctx = NSGraphicsContext.current?.cgContext else { exit(70) }

// ── The squircle plate (Apple icon grid: ~10% margin, ~22.5% corner radius) ──
let margin = size * 0.100
let plate = CGRect(x: margin, y: margin,
                   width: size - 2 * margin, height: size - 2 * margin)
let corner = plate.width * 0.225
let platePath = NSBezierPath(roundedRect: plate, xRadius: corner, yRadius: corner)

// Vertical gradient: near-black indigo floor to a deep blue-violet crown.
let gradient = NSGradient(colors: [
    NSColor(calibratedRed: 0.055, green: 0.055, blue: 0.130, alpha: 1.0),
    NSColor(calibratedRed: 0.150, green: 0.130, blue: 0.360, alpha: 1.0),
])!
gradient.draw(in: platePath, angle: 90)

// Everything else clips to the plate.
platePath.addClip()

// ── The residual stream: five lines flowing left → right ─────────────────────
// Four stay flat (muted); the third from the top is the steered one — it
// bends upward after the injection point and brightens.
let left = plate.minX + plate.width * 0.10
let right = plate.maxX - plate.width * 0.10
let streamWidth = size * 0.022
let steerWidth = size * 0.030

func flatLine(y: CGFloat, alpha: CGFloat) {
    let p = NSBezierPath()
    p.move(to: CGPoint(x: left, y: y))
    p.line(to: CGPoint(x: right, y: y))
    p.lineWidth = streamWidth
    p.lineCapStyle = .round
    NSColor(calibratedRed: 0.55, green: 0.62, blue: 0.85, alpha: alpha).setStroke()
    p.stroke()
}

let ys: [CGFloat] = [0.255, 0.370, 0.485, 0.600, 0.715].map {
    plate.minY + plate.height * $0
}
flatLine(y: ys[0], alpha: 0.38)
flatLine(y: ys[1], alpha: 0.46)
// ys[2] is the steered line, drawn below.
flatLine(y: ys[3], alpha: 0.46)
flatLine(y: ys[4], alpha: 0.38)

// ── The steered line: flat, then a smooth upward deflection ──────────────────
let baseY = ys[2]
let bendX = plate.minX + plate.width * 0.46      // the injection point
let endY = baseY + plate.height * 0.240           // deflected height at exit

let steered = NSBezierPath()
steered.move(to: CGPoint(x: left, y: baseY))
steered.line(to: CGPoint(x: bendX, y: baseY))
steered.curve(to: CGPoint(x: right, y: endY),
              controlPoint1: CGPoint(x: bendX + plate.width * 0.22, y: baseY),
              controlPoint2: CGPoint(x: bendX + plate.width * 0.26, y: endY))
steered.lineWidth = steerWidth
steered.lineCapStyle = .round

// Soft glow first, then the bright core.
ctx.saveGState()
ctx.setShadow(offset: .zero, blur: size * 0.030,
              color: NSColor(calibratedRed: 0.35, green: 0.90, blue: 1.0,
                             alpha: 0.85).cgColor)
NSColor(calibratedRed: 0.62, green: 0.95, blue: 1.0, alpha: 1.0).setStroke()
steered.stroke()
ctx.restoreGState()
NSColor(calibratedRed: 0.86, green: 0.99, blue: 1.0, alpha: 1.0).setStroke()
steered.stroke()

// ── The injected vector: a short upward arrow meeting the bend point ─────────
let arrowX = bendX
let arrowBottom = baseY - plate.height * 0.185
let arrowTop = baseY - steerWidth * 0.9
let shaft = NSBezierPath()
shaft.move(to: CGPoint(x: arrowX, y: arrowBottom))
shaft.line(to: CGPoint(x: arrowX, y: arrowTop))
shaft.lineWidth = steerWidth * 0.85
shaft.lineCapStyle = .round

let headSpan = size * 0.036
let head = NSBezierPath()
head.move(to: CGPoint(x: arrowX - headSpan, y: arrowTop - headSpan * 1.15))
head.line(to: CGPoint(x: arrowX, y: arrowTop))
head.line(to: CGPoint(x: arrowX + headSpan, y: arrowTop - headSpan * 1.15))
head.lineWidth = steerWidth * 0.85
head.lineCapStyle = .round
head.lineJoinStyle = .round

ctx.saveGState()
ctx.setShadow(offset: .zero, blur: size * 0.022,
              color: NSColor(calibratedRed: 0.55, green: 1.0, blue: 0.85,
                             alpha: 0.9).cgColor)
NSColor(calibratedRed: 0.55, green: 1.0, blue: 0.80, alpha: 1.0).setStroke()
shaft.stroke()
head.stroke()
ctx.restoreGState()

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let png = rep.representation(using: .png, properties: [:])
else { exit(70) }
do {
    try png.write(to: URL(filePath: outPath))
} catch {
    FileHandle.standardError.write(Data("write failed: \(error)\n".utf8))
    exit(70)
}
print("wrote \(outPath)")
