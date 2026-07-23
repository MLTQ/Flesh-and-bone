import CoreGraphics
import Foundation
import Metal
import simd

enum PopulationBenchmark {
    private static func activeBufferCount(_ simulation: FleshSimulation) -> Int {
        let values = simulation.population.contents().bindMemory(
            to: Float.self, capacity: simulation.slotCount)
        return (0..<simulation.slotCount).reduce(into: 0) {
            if values[$1] >= 0.5 { $0 += 1 }
        }
    }

    private static func selectedResidual(
        _ simulation: FleshSimulation,
        change: PopulationChange
    ) -> (rms: Float, maximum: Float) {
        let values = simulation.residual.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: simulation.slotCount)
        var squared: Double = 0
        var maximum: Float = 0
        for raw in change.indices {
            let residual = values[Int(raw)]
            let magnitude = simd_length(SIMD3<Float>(
                residual.x, residual.y, residual.z))
            squared += Double(magnitude * magnitude)
            maximum = max(maximum, magnitude)
        }
        return (
            Float(sqrt(squared / Double(max(change.count, 1)))),
            maximum)
    }

    private static func pairSeparation(
        _ simulation: FleshSimulation,
        templates: [UInt32]
    ) -> Float {
        let positions = simulation.lbsCurrent.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: simulation.slotCount)
        let residuals = simulation.residual.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: simulation.slotCount)
        let base = simulation.populationBaseline
        var squared: Double = 0
        for raw in templates {
            let lower = Int(raw)
            let upper = base + lower
            let left = positions[lower] + residuals[lower]
            let right = positions[upper] + residuals[upper]
            let difference = SIMD3<Float>(
                right.x - left.x, right.y - left.y, right.z - left.z)
            squared += Double(simd_length_squared(difference))
        }
        return Float(sqrt(squared / Double(max(templates.count, 1))))
    }

    private static func commit(
        queue: MTLCommandQueue,
        simulation: FleshSimulation,
        change: PopulationChange
    ) throws {
        guard let command = queue.makeCommandBuffer() else {
            throw RuntimeAssetError.allocation("population-test command")
        }
        simulation.encodePopulationChange(
            commandBuffer: command, change: change)
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw RuntimeAssetError.invalid(
                command.error?.localizedDescription
                    ?? "population-test GPU failure")
        }
    }

    private static func advance(
        queue: MTLCommandQueue,
        simulation: FleshSimulation,
        frames: Int,
        label: String
    ) throws {
        guard let command = queue.makeCommandBuffer() else {
            throw RuntimeAssetError.allocation(label)
        }
        for _ in 0..<max(frames, 1) {
            simulation.encodeFrame(commandBuffer: command)
        }
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw RuntimeAssetError.invalid(
                command.error?.localizedDescription ?? "\(label) GPU failure")
        }
    }

    static func run(
        device: MTLDevice,
        assetDirectory: String,
        recoveryFrames: Int = 180
    ) throws {
        guard let queue = device.makeCommandQueue() else {
            throw RuntimeAssetError.allocation("population-test queue")
        }
        let library = try device.makeLibrary(
            source: fleshMetalSource, options: nil)
        let model = try RuntimeModel.load(
            path: assetDirectory + "/h7c_seed7.fnm", device: device)
        var camera = OrbitCamera()
        camera.apply(.front)
        let viewport = CGSize(width: 1100, height: 760)
        let brushCenter = CGPoint(x: 550, y: 380)
        print("Flesh-and-Bone directed population benchmark")
        print(
            "GPU: \(device.name), front brush: 72 px, reserve: 200%, "
            + "wound dwell: 30 frames")
        print(
            "cells\tpainted\tpeak%\tspawn mm\tovercap mm\tpair/control mm\t"
            + "wound%\trestored\tfinite\tverdict")

        for path in RuntimeAssets.profilePaths(in: assetDirectory) {
            let body = try RuntimeBody.load(path: path, device: device)
            let simulation = try FleshSimulation(
                device: device, library: library, body: body, model: model)
            try advance(
                queue: queue,
                simulation: simulation,
                frames: 1,
                label: "population-test prime")

            let templates = PopulationBrush.selectTemplates(
                simulation: simulation,
                camera: camera,
                viewport: viewport,
                location: brushCenter,
                radius: 72,
                mode: .source,
                renderCount: body.cellCount)
            guard !templates.isEmpty,
                  let overfill = simulation.planSource(templates: templates)
            else {
                throw RuntimeAssetError.invalid(
                    "front population brush selected no source niches")
            }
            try commit(queue: queue, simulation: simulation, change: overfill)
            let peakActive = activeBufferCount(simulation)
            let spawn = selectedResidual(simulation, change: overfill)
            try advance(
                queue: queue,
                simulation: simulation,
                frames: recoveryFrames,
                label: "overcapacity recovery")
            let overcap = selectedResidual(simulation, change: overfill)
            let separation = pairSeparation(
                simulation, templates: templates)
            let overcapState = simulation.stateSummary()

            let control = try FleshSimulation(
                device: device, library: library, body: body, model: model)
            control.densityEnabled = false
            try advance(
                queue: queue,
                simulation: control,
                frames: 1,
                label: "overcapacity control prime")
            guard let controlOverfill = control.planSource(
                templates: templates) else {
                throw RuntimeAssetError.invalid(
                    "could not source overcapacity control")
            }
            try commit(
                queue: queue,
                simulation: control,
                change: controlOverfill)
            try advance(
                queue: queue,
                simulation: control,
                frames: recoveryFrames,
                label: "overcapacity backbone control")
            let controlSeparation = pairSeparation(
                control, templates: templates)
            let controlState = control.stateSummary()

            guard let removeReserve = simulation.planVacuum(
                templates: templates) else {
                throw RuntimeAssetError.invalid("could not remove reserve layer")
            }
            try commit(
                queue: queue,
                simulation: simulation,
                change: removeReserve)
            let returnedBaseline = activeBufferCount(simulation)

            guard let wound = simulation.planVacuum(templates: templates) else {
                throw RuntimeAssetError.invalid("could not paint local wound")
            }
            try commit(queue: queue, simulation: simulation, change: wound)
            let woundActive = activeBufferCount(simulation)
            try advance(
                queue: queue,
                simulation: simulation,
                frames: 30,
                label: "population wound dwell")

            guard let refill = simulation.planSource(templates: templates) else {
                throw RuntimeAssetError.invalid("could not refill local wound")
            }
            try commit(queue: queue, simulation: simulation, change: refill)
            try advance(
                queue: queue,
                simulation: simulation,
                frames: recoveryFrames,
                label: "population wound recovery")
            let restored = activeBufferCount(simulation)
            let finalState = simulation.stateSummary()
            let peakPercent = 100 * Double(peakActive) / Double(body.cellCount)
            let woundPercent = 100 * Double(woundActive) / Double(body.cellCount)
            let passed = peakActive == body.cellCount + templates.count
                && returnedBaseline == body.cellCount
                && woundActive == body.cellCount - templates.count
                && restored == body.cellCount
                && overcapState.finite
                && controlState.finite
                && finalState.finite
                && spawn.rms > body.pitch
                && overcap.rms < spawn.rms
                && separation > controlSeparation
            print(
                "\(body.cellCount)\t\(templates.count)\t"
                + String(
                    format:
                        "%.2f\t%.3f\t%.3f\t%.3f/%.3f\t%.2f\t%d\t%@\t%@",
                    peakPercent,
                    spawn.rms * 1000,
                    overcap.rms * 1000,
                    separation * 1000,
                    controlSeparation * 1000,
                    woundPercent,
                    restored,
                    (overcapState.finite && controlState.finite
                        && finalState.finite) ? "yes" : "NO",
                    passed ? "PASS" : "FAIL"))
        }
    }
}
