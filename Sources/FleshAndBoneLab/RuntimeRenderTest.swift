import AppKit
import Foundation
import Metal

enum RuntimeRenderTest {
    private static func runtime(
        device: MTLDevice,
        assetDirectory: String,
        profileIndex: Int
    ) throws -> (
        queue: MTLCommandQueue,
        simulation: FleshSimulation,
        renderer: FleshRenderer
    ) {
        guard let queue = device.makeCommandQueue() else {
            throw RuntimeAssetError.allocation("render-test queue")
        }
        let library = try device.makeLibrary(source: fleshMetalSource, options: nil)
        let model = try RuntimeModel.load(
            path: assetDirectory + "/h7c_seed7.fnm", device: device)
        let profiles = RuntimeAssets.profilePaths(in: assetDirectory)
        guard profiles.indices.contains(profileIndex) else {
            throw RuntimeAssetError.invalid("render-test profile index")
        }
        let body = try RuntimeBody.load(
            path: profiles[profileIndex], device: device)
        let simulation = try FleshSimulation(
            device: device, library: library, body: body, model: model)
        let renderer = try FleshRenderer(device: device, library: library)
        guard let settle = queue.makeCommandBuffer() else {
            throw RuntimeAssetError.allocation("render-test settle command")
        }
        for _ in 0..<30 { simulation.encodeFrame(commandBuffer: settle) }
        settle.commit()
        settle.waitUntilCompleted()
        return (queue, simulation, renderer)
    }

    private static func texture(
        device: MTLDevice, width: Int, height: Int
    ) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false)
        descriptor.usage = [.renderTarget]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw RuntimeAssetError.allocation("render-test texture")
        }
        return texture
    }

    static func render(
        device: MTLDevice,
        assetDirectory: String,
        outputPath: String,
        profileIndex: Int = 2,
        renderCount: Int? = nil,
        radiusMultiplier: Float = 1,
        opacity: Float = 0.72,
        camera: OrbitCamera = OrbitCamera()
    ) throws {
        let runtime = try runtime(
            device: device,
            assetDirectory: assetDirectory,
            profileIndex: profileIndex)
        let width = 1100
        let height = 760
        let target = try texture(device: device, width: width, height: height)
        let count = renderCount ?? runtime.simulation.body.cellCount
        runtime.simulation.body.sortRenderOrder(
            camera: camera, count: count)
        guard let command = runtime.queue.makeCommandBuffer() else {
            throw RuntimeAssetError.allocation("render-test command")
        }
        runtime.renderer.encode(
            commandBuffer: command,
            texture: target,
            simulation: runtime.simulation,
            renderCount: count,
            radiusMultiplier: radiusMultiplier,
            opacity: opacity,
            camera: camera)
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw RuntimeAssetError.invalid(
                command.error?.localizedDescription ?? "render-test GPU failure")
        }

        var bgra = [UInt8](repeating: 0, count: width * height * 4)
        target.getBytes(
            &bgra,
            bytesPerRow: width * 4,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0)
        var rgba = bgra
        for index in stride(from: 0, to: rgba.count, by: 4) {
            rgba.swapAt(index, index + 2)
        }
        guard let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bitmapFormat: .alphaNonpremultiplied,
            bytesPerRow: width * 4,
            bitsPerPixel: 32),
              let destination = bitmap.bitmapData
        else {
            throw RuntimeAssetError.allocation("PNG bitmap")
        }
        _ = rgba.withUnsafeBytes { source in
            memcpy(destination, source.baseAddress!, rgba.count)
        }
        guard let png = bitmap.representation(
            using: .png, properties: [:]) else {
            throw RuntimeAssetError.invalid("PNG encoding")
        }
        try png.write(to: URL(fileURLWithPath: outputPath))
        print("wrote \(outputPath)")
    }

    static func benchmark(
        device: MTLDevice,
        assetDirectory: String,
        frames: Int = 30
    ) throws {
        print("Flesh-and-Bone native render benchmark")
        print("GPU: \(device.name), target: 1100×760 BGRA8")
        print("cells\trendered\tradius×\tms/frame")
        let profiles = RuntimeAssets.profilePaths(in: assetDirectory)
        for profileIndex in profiles.indices {
            let runtime = try runtime(
                device: device,
                assetDirectory: assetDirectory,
                profileIndex: profileIndex)
            let target = try texture(
                device: device, width: 1100, height: 760)
            runtime.simulation.body.sortRenderOrder(
                camera: OrbitCamera(),
                count: runtime.simulation.body.cellCount)
            guard let warmup = runtime.queue.makeCommandBuffer() else {
                throw RuntimeAssetError.allocation(
                    "render benchmark warmup command")
            }
            runtime.renderer.encode(
                commandBuffer: warmup,
                texture: target,
                simulation: runtime.simulation,
                renderCount: runtime.simulation.body.cellCount,
                radiusMultiplier: 1,
                opacity: 0.72,
                camera: OrbitCamera())
            warmup.commit()
            warmup.waitUntilCompleted()
            let fractions: [Float] = profileIndex == profiles.count - 1
                ? [0.25, 0.50, 1.0]
                : [1.0]
            for fraction in fractions {
                for radius: Float in [0.75, 1.0, 1.5] {
                    let count = Int(
                        Float(runtime.simulation.body.cellCount) * fraction)
                    runtime.simulation.body.sortRenderOrder(
                        camera: OrbitCamera(), count: count)
                    guard let command = runtime.queue.makeCommandBuffer() else {
                        throw RuntimeAssetError.allocation("render benchmark command")
                    }
                    for _ in 0..<frames {
                        runtime.renderer.encode(
                            commandBuffer: command,
                            texture: target,
                            simulation: runtime.simulation,
                            renderCount: count,
                            radiusMultiplier: radius,
                            opacity: 0.72,
                            camera: OrbitCamera())
                    }
                    command.commit()
                    command.waitUntilCompleted()
                    let seconds = command.gpuEndTime - command.gpuStartTime
                    let milliseconds = seconds * 1000 / Double(frames)
                    print(
                        "\(runtime.simulation.body.cellCount)\t"
                        + "\(Int(Float(runtime.simulation.body.cellCount) * fraction))\t"
                        + String(format: "%.2f\t%.3f", radius, milliseconds))
                }
            }
        }
    }
}
