import AppKit
import Metal

if CommandLine.arguments.contains("--benchmark")
    || CommandLine.arguments.contains("--render-test")
    || CommandLine.arguments.contains("--render-benchmark") {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Metal is unavailable\n", stderr)
        exit(2)
    }
    guard let assets = RuntimeAssets.directory() else {
        fputs("runtime assets not found; run scripts/export_flesh_runtime.py\n", stderr)
        exit(2)
    }
    do {
        if let index = CommandLine.arguments.firstIndex(of: "--benchmark") {
            let frames = CommandLine.arguments.indices.contains(index + 1)
                ? Int(CommandLine.arguments[index + 1]) ?? 90
                : 90
            try RuntimeBenchmark.run(
                device: device,
                assetDirectory: assets,
                frames: max(frames, 1))
        } else if let index = CommandLine.arguments.firstIndex(
            of: "--render-test") {
            let output = CommandLine.arguments.indices.contains(index + 1)
                ? CommandLine.arguments[index + 1]
                : "flesh-render.png"
            try RuntimeRenderTest.render(
                device: device,
                assetDirectory: assets,
                outputPath: output)
        } else {
            try RuntimeRenderTest.benchmark(
                device: device, assetDirectory: assets)
        }
    } catch {
        fputs("benchmark failed: \(error)\n", stderr)
        exit(1)
    }
} else {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let delegate = AppDelegate()
    app.delegate = delegate
    app.run()
}
