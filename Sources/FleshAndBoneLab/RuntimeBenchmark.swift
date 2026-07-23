import Foundation
import Metal

enum RuntimeBenchmark {
    static func run(device: MTLDevice, assetDirectory: String,
                    frames: Int = 90) throws {
        guard let queue = device.makeCommandQueue() else {
            throw RuntimeAssetError.allocation("benchmark queue")
        }
        let library = try device.makeLibrary(source: fleshMetalSource, options: nil)
        let model = try RuntimeModel.load(
            path: assetDirectory + "/h7c_seed7.fnm", device: device)
        let backboneOnly = ProcessInfo.processInfo.environment[
            "FLESH_BACKBONE_ONLY"] == "1"
        let debugCell = ProcessInfo.processInfo.environment[
            "FLESH_DEBUG_CELL"].flatMap(Int.init)
        let debugLastSubsteps = ProcessInfo.processInfo.environment[
            "FLESH_DEBUG_LAST_SUBSTEPS"].flatMap(Int.init)
        print("Flesh-and-Bone native compute benchmark")
        print("GPU: \(device.name)")
        print(
            "frames: \(frames), physics: 4 substeps/frame, motion: tracked walk, "
            + "density: \(backboneOnly ? "off" : "on")")
        print("cells\tms/frame\trealtime×\tasset MB\tstate MB\tRMS mm\tmax mm\tbad(first)\tfinite")
        for path in RuntimeAssets.profilePaths(in: assetDirectory) {
            let body = try RuntimeBody.load(path: path, device: device)
            let simulation = try FleshSimulation(
                device: device, library: library, body: body, model: model)
            simulation.densityEnabled = !backboneOnly
            guard let warmup = queue.makeCommandBuffer() else {
                throw RuntimeAssetError.allocation("warmup command")
            }
            for _ in 0..<6 { simulation.encodeFrame(commandBuffer: warmup) }
            warmup.commit()
            warmup.waitUntilCompleted()
            simulation.resetDynamics()
            guard let command = queue.makeCommandBuffer() else {
                throw RuntimeAssetError.allocation("benchmark command")
            }
            for frame in 0..<frames {
                let substeps = frame == frames - 1
                    ? debugLastSubsteps ?? 4
                    : 4
                simulation.encodeFrame(
                    commandBuffer: command, substeps: substeps)
            }
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw RuntimeAssetError.invalid(
                    command.error?.localizedDescription ?? "benchmark GPU failure")
            }
            let elapsed = command.gpuEndTime > command.gpuStartTime
                ? command.gpuEndTime - command.gpuStartTime
                : 0
            let milliseconds = elapsed * 1000 / Double(frames)
            let realtime = milliseconds > 0 ? (1000.0 / 30.0) / milliseconds : 0
            let summary = simulation.stateSummary()
            let assetMB = Double(body.sourceBytes + model.sourceBytes) / 1_000_000
            let stateMB = Double(simulation.dynamicBytes) / 1_000_000
            print(
                "\(body.cellCount)\t"
                + String(format: "%.3f\t%.2f\t%.1f\t%.1f\t%.3f\t%.3f\t%@\t%@",
                         milliseconds, realtime, assetMB, stateMB,
                         summary.residualRMS * 1000,
                         summary.residualMaximum * 1000,
                         summary.nonfiniteCount == 0
                            ? "0"
                            : "\(summary.nonfiniteCount)(\(summary.firstNonfiniteIndex ?? -1))",
                         summary.finite ? "yes" : "NO")
            )
            if let index = summary.firstNonfiniteIndex {
                print("  \(simulation.diagnosticState(cell: index))")
            } else if let debugCell {
                print("  \(simulation.diagnosticState(cell: debugCell))")
            }
        }
    }
}
