// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FleshAndBoneLab",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "FleshAndBoneLab",
            path: "Sources/FleshAndBoneLab",
            exclude: [
                "AppDelegate.md",
                "Camera.md",
                "ControlPanel.md",
                "FleshMetalView.md",
                "FleshRenderer.md",
                "FleshSimulation.md",
                "MetalShaders.md",
                "PerformanceMonitor.md",
                "RuntimeAsset.md",
                "RuntimeBenchmark.md",
                "RuntimeRenderTest.md",
                "main.md",
            ]
        )
    ]
)
