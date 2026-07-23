import Foundation
import Metal
import simd

struct SimulationUniforms {
    var cellCount: UInt32
    var baseCellCount: UInt32
    var boneCount: UInt32
    var frameCount: UInt32
    var physicsEnabled: UInt32
    var layerCount: UInt32
    var phase: Float
    var phaseDelta: Float
    var motionIntensity: Float
    var fps: Float
    var dt: Float
    var pitch: Float
    var neighborScale: Float
    var accelerationCap: Float
}

struct PopulationUniforms {
    var cellCount: UInt32
    var baseCellCount: UInt32
    var changeCount: UInt32
    var activate: UInt32
}

struct StateSummary {
    let residualRMS: Float
    let residualMaximum: Float
    let finite: Bool
    let nonfiniteCount: Int
    let firstNonfiniteIndex: Int?
}

final class FleshSimulation {
    let device: MTLDevice
    let body: RuntimeBody
    let model: RuntimeModel
    let slotCount: Int
    let dynamicBytes: Int

    private let skinPipeline: MTLComputePipelineState
    private let observePipeline: MTLComputePipelineState
    private let integratePipeline: MTLComputePipelineState
    private let populationPipeline: MTLComputePipelineState
    private let populationController: PopulationController
    private var residualA: MTLBuffer
    private var residualB: MTLBuffer
    private var velocityA: MTLBuffer
    private var velocityB: MTLBuffer
    private(set) var residual: MTLBuffer
    private var velocity: MTLBuffer
    private(set) var lbsPrevious: MTLBuffer
    private(set) var lbsCurrent: MTLBuffer
    private(set) var lbsNext: MTLBuffer
    private let lbsSource: MTLBuffer
    private(set) var population: MTLBuffer
    private let neighborResidual: MTLBuffer
    private let neighborVelocity: MTLBuffer
    private let compressionVector: MTLBuffer
    private let stretchVector: MTLBuffer
    private let densityScalars: MTLBuffer

    var motionSpeed: Float = 1.0
    var motionIntensity: Float = 1.0
    var densityEnabled = true
    var physicsEnabled = true {
        didSet {
            if !physicsEnabled { resetDynamics() }
        }
    }
    private(set) var phase: Float = 0

    init(device: MTLDevice, library: MTLLibrary, body: RuntimeBody,
         model: RuntimeModel) throws {
        self.device = device
        self.body = body
        self.model = model
        slotCount = body.cellCount * 2
        guard let skinFunction = library.makeFunction(name: "skin_motion"),
              let observeFunction = library.makeFunction(name: "observe_neighbors"),
              let integrateFunction = library.makeFunction(name: "integrate_flesh"),
              let populationFunction = library.makeFunction(
                name: "apply_population_change")
        else {
            throw RuntimeAssetError.invalid("missing compute functions")
        }
        skinPipeline = try device.makeComputePipelineState(function: skinFunction)
        observePipeline = try device.makeComputePipelineState(function: observeFunction)
        integratePipeline = try device.makeComputePipelineState(function: integrateFunction)
        populationPipeline = try device.makeComputePipelineState(
            function: populationFunction)
        populationController = PopulationController(baseCount: body.cellCount)

        let vectorBytes = slotCount * MemoryLayout<SIMD4<Float>>.stride
        let scalarBytes = slotCount * MemoryLayout<Float>.stride
        func make(_ label: String) throws -> MTLBuffer {
            guard let value = device.makeBuffer(
                length: vectorBytes, options: .storageModeShared)
            else { throw RuntimeAssetError.allocation(label) }
            value.label = label
            memset(value.contents(), 0, vectorBytes)
            return value
        }
        residualA = try make("residual A")
        residualB = try make("residual B")
        velocityA = try make("velocity A")
        velocityB = try make("velocity B")
        residual = residualA
        velocity = velocityA
        lbsPrevious = try make("LBS previous")
        lbsCurrent = try make("LBS current")
        lbsNext = try make("LBS next")
        lbsSource = try make("LBS source")
        neighborResidual = try make("neighbor residual")
        neighborVelocity = try make("neighbor velocity")
        compressionVector = try make("compression vector")
        stretchVector = try make("stretch vector")
        densityScalars = try make("density scalars")
        guard let population = device.makeBuffer(
            length: scalarBytes, options: .storageModeShared)
        else { throw RuntimeAssetError.allocation("active population") }
        population.label = "active population"
        let populationValues = population.contents().bindMemory(
            to: Float.self, capacity: slotCount)
        populationValues.initialize(repeating: 0, count: slotCount)
        for index in 0..<body.cellCount { populationValues[index] = 1 }
        self.population = population
        dynamicBytes = vectorBytes * 13 + scalarBytes
    }

    private func uniforms() -> SimulationUniforms {
        let scale = model.referencePitch / body.pitch
        return SimulationUniforms(
            cellCount: UInt32(slotCount),
            baseCellCount: UInt32(body.cellCount),
            boneCount: UInt32(body.boneCount),
            frameCount: UInt32(body.frameCount),
            physicsEnabled: physicsEnabled ? (densityEnabled ? 1 : 2) : 0,
            layerCount: UInt32(populationController.layerCount),
            phase: phase,
            phaseDelta: motionSpeed,
            motionIntensity: motionIntensity,
            fps: 30,
            dt: 1.0 / 120.0,
            pitch: body.pitch,
            neighborScale: scale * scale,
            accelerationCap: model.accelerationCap
        )
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          count: Int? = nil) {
        encoder.setComputePipelineState(pipeline)
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count ?? slotCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    var activeCount: Int {
        populationController.activeCount
    }

    var populationCapacity: Int {
        populationController.capacity
    }

    var populationLayerCount: Int {
        populationController.layerCount
    }

    var populationBaseline: Int {
        populationController.baseCount
    }

    var populationPercentage: Double {
        populationController.baselineFraction * 100
    }

    func canSource(template: Int) -> Bool {
        populationController.canSource(template: template)
    }

    func representativeSlot(template: Int) -> Int? {
        populationController.representativeSlot(template: template)
    }

    func planSource(templates: [UInt32]) -> PopulationChange? {
        populationController.planSource(templates: templates)
    }

    func planVacuum(templates: [UInt32]) -> PopulationChange? {
        populationController.planVacuum(templates: templates)
    }

    func encodePopulationChange(
        commandBuffer: MTLCommandBuffer,
        change: PopulationChange
    ) {
        guard change.count > 0,
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return }
        let indexBytes = change.count * MemoryLayout<UInt32>.stride
        let changeBuffer = change.indices.withUnsafeBytes { source in
            source.baseAddress.flatMap {
                device.makeBuffer(
                    bytes: $0,
                    length: indexBytes,
                    options: .storageModeShared)
            }
        }
        guard let changeBuffer else { return }
        changeBuffer.label = "population brush indices"
        var values = PopulationUniforms(
            cellCount: UInt32(slotCount),
            baseCellCount: UInt32(body.cellCount),
            changeCount: UInt32(change.count),
            activate: change.activate ? 1 : 0)
        encoder.label = change.activate ? "NCA source" : "NCA vacuum"
        encoder.setBuffer(changeBuffer, offset: 0, index: 0)
        encoder.setBuffer(population, offset: 0, index: 1)
        encoder.setBuffer(lbsCurrent, offset: 0, index: 2)
        encoder.setBuffer(lbsSource, offset: 0, index: 3)
        encoder.setBuffer(residualA, offset: 0, index: 4)
        encoder.setBuffer(residualB, offset: 0, index: 5)
        encoder.setBuffer(velocityA, offset: 0, index: 6)
        encoder.setBuffer(velocityB, offset: 0, index: 7)
        encoder.setBytes(
            &values, length: MemoryLayout<PopulationUniforms>.stride, index: 8)
        dispatch(encoder, pipeline: populationPipeline, count: change.count)
        encoder.endEncoding()
    }

    func encodeFrame(commandBuffer: MTLCommandBuffer, substeps: Int = 4) {
        var values = uniforms()
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "skin motion"
            encoder.setBuffer(body.points, offset: 0, index: 0)
            encoder.setBuffer(body.sourceAnchors, offset: 0, index: 1)
            encoder.setBuffer(body.skinIndices, offset: 0, index: 2)
            encoder.setBuffer(body.skinWeights, offset: 0, index: 3)
            encoder.setBuffer(body.skinMatrices, offset: 0, index: 4)
            encoder.setBuffer(lbsPrevious, offset: 0, index: 5)
            encoder.setBuffer(lbsCurrent, offset: 0, index: 6)
            encoder.setBuffer(lbsNext, offset: 0, index: 7)
            encoder.setBuffer(lbsSource, offset: 0, index: 8)
            encoder.setBytes(
                &values, length: MemoryLayout<SimulationUniforms>.stride, index: 9)
            dispatch(encoder, pipeline: skinPipeline)
            encoder.endEncoding()
        }
        if physicsEnabled {
            for _ in 0..<max(substeps, 0) {
                if let observer = commandBuffer.makeComputeCommandEncoder() {
                    observer.label = "observe fixed neighbors"
                    observer.setBuffer(residual, offset: 0, index: 0)
                    observer.setBuffer(velocity, offset: 0, index: 1)
                    observer.setBuffer(lbsCurrent, offset: 0, index: 2)
                    observer.setBuffer(body.material, offset: 0, index: 3)
                    observer.setBuffer(body.neighbors, offset: 0, index: 4)
                    observer.setBuffer(population, offset: 0, index: 5)
                    observer.setBuffer(neighborResidual, offset: 0, index: 6)
                    observer.setBuffer(neighborVelocity, offset: 0, index: 7)
                    observer.setBuffer(compressionVector, offset: 0, index: 8)
                    observer.setBuffer(stretchVector, offset: 0, index: 9)
                    observer.setBuffer(densityScalars, offset: 0, index: 10)
                    observer.setBytes(
                        &values,
                        length: MemoryLayout<SimulationUniforms>.stride,
                        index: 11)
                    dispatch(observer, pipeline: observePipeline)
                    observer.endEncoding()
                }
                let nextResidual = residual === residualA ? residualB : residualA
                let nextVelocity = velocity === velocityA ? velocityB : velocityA
                if let integrator = commandBuffer.makeComputeCommandEncoder() {
                    integrator.label = "integrate H6C plus H7C"
                    integrator.setBuffer(residual, offset: 0, index: 0)
                    integrator.setBuffer(velocity, offset: 0, index: 1)
                    integrator.setBuffer(lbsPrevious, offset: 0, index: 2)
                    integrator.setBuffer(lbsCurrent, offset: 0, index: 3)
                    integrator.setBuffer(lbsNext, offset: 0, index: 4)
                    integrator.setBuffer(body.material, offset: 0, index: 5)
                    integrator.setBuffer(neighborResidual, offset: 0, index: 6)
                    integrator.setBuffer(neighborVelocity, offset: 0, index: 7)
                    integrator.setBuffer(compressionVector, offset: 0, index: 8)
                    integrator.setBuffer(stretchVector, offset: 0, index: 9)
                    integrator.setBuffer(densityScalars, offset: 0, index: 10)
                    integrator.setBuffer(population, offset: 0, index: 11)
                    integrator.setBuffer(model.values, offset: 0, index: 12)
                    integrator.setBuffer(nextResidual, offset: 0, index: 13)
                    integrator.setBuffer(nextVelocity, offset: 0, index: 14)
                    integrator.setBytes(
                        &values,
                        length: MemoryLayout<SimulationUniforms>.stride,
                        index: 15)
                    dispatch(integrator, pipeline: integratePipeline)
                    integrator.endEncoding()
                }
                residual = nextResidual
                velocity = nextVelocity
            }
        }
        phase = (phase + motionSpeed).truncatingRemainder(
            dividingBy: Float(body.frameCount))
    }

    func resetDynamics() {
        phase = 0
        let bytes = slotCount * MemoryLayout<SIMD4<Float>>.stride
        for buffer in [residualA, residualB, velocityA, velocityB] {
            memset(buffer.contents(), 0, bytes)
        }
        residual = residualA
        velocity = velocityA
    }

    func stateSummary() -> StateSummary {
        let values = residual.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: slotCount)
        let active = population.contents().bindMemory(
            to: Float.self, capacity: slotCount)
        var squared: Double = 0
        var maximum: Float = 0
        var finiteCount = 0
        var nonfiniteCount = 0
        var firstNonfiniteIndex: Int?
        for index in 0..<slotCount {
            if active[index] < 0.5 { continue }
            let vector = SIMD3<Float>(
                values[index].x, values[index].y, values[index].z)
            let magnitude = simd_length(vector)
            if magnitude.isFinite {
                squared += Double(magnitude * magnitude)
                maximum = max(maximum, magnitude)
                finiteCount += 1
            } else {
                nonfiniteCount += 1
                if firstNonfiniteIndex == nil { firstNonfiniteIndex = index }
            }
        }
        return StateSummary(
            residualRMS: Float(sqrt(squared / Double(max(finiteCount, 1)))),
            residualMaximum: maximum,
            finite: nonfiniteCount == 0,
            nonfiniteCount: nonfiniteCount,
            firstNonfiniteIndex: firstNonfiniteIndex
        )
    }

    func diagnosticState(cell: Int) -> String {
        guard (0..<slotCount).contains(cell) else { return "cell out of range" }
        func vector(_ buffer: MTLBuffer) -> SIMD4<Float> {
            buffer.contents().bindMemory(
                to: SIMD4<Float>.self, capacity: slotCount)[cell]
        }
        let activeValue = population.contents().bindMemory(
            to: Float.self, capacity: slotCount)[cell]
        return "cell \(cell) residual=\(vector(residual)) "
            + "velocity=\(vector(velocity)) scalars=\(vector(densityScalars)) "
            + "compression=\(vector(compressionVector)) "
            + "stretch=\(vector(stretchVector)) active=\(activeValue)"
    }
}
