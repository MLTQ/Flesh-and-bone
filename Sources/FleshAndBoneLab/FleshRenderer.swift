import Foundation
import Metal
import simd

struct RenderUniforms {
    var viewProjection: simd_float4x4
    var cameraRight: SIMD4<Float>
    var cameraUp: SIMD4<Float>
    var baseRadius: Float
    var radiusMultiplier: Float
    var opacity: Float
    var baseCellCount: UInt32
    var renderCount: UInt32
    var layerCount: UInt32
}

final class FleshRenderer {
    private let pipeline: MTLRenderPipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let vertex = library.makeFunction(name: "splat_vertex"),
              let fragment = library.makeFunction(name: "splat_fragment")
        else {
            throw RuntimeAssetError.invalid("missing render functions")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Gaussian flesh splats"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        let attachment = descriptor.colorAttachments[0]!
        attachment.pixelFormat = .bgra8Unorm
        attachment.isBlendingEnabled = true
        attachment.sourceRGBBlendFactor = .one
        attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        attachment.sourceAlphaBlendFactor = .one
        attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        texture: MTLTexture,
        simulation: FleshSimulation,
        renderCount: Int,
        radiusMultiplier: Float,
        opacity: Float,
        camera: OrbitCamera
    ) {
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = texture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColor(
            red: 0.018, green: 0.024, blue: 0.032, alpha: 1)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(
            descriptor: descriptor) else { return }
        encoder.label = "render Gaussian cells"
        encoder.setRenderPipelineState(pipeline)
        let body = simulation.body
        encoder.setVertexBuffer(simulation.lbsCurrent, offset: 0, index: 0)
        encoder.setVertexBuffer(simulation.residual, offset: 0, index: 1)
        encoder.setVertexBuffer(body.material, offset: 0, index: 2)
        encoder.setVertexBuffer(body.colors, offset: 0, index: 3)
        encoder.setVertexBuffer(body.renderOrder, offset: 0, index: 4)
        encoder.setVertexBuffer(simulation.population, offset: 0, index: 5)
        let aspect = Float(texture.width) / Float(max(texture.height, 1))
        let cameraValues = camera.matrices(aspect: aspect)
        let boundedRenderCount = min(max(renderCount, 1), body.cellCount)
        var uniforms = RenderUniforms(
            viewProjection: cameraValues.viewProjection,
            cameraRight: SIMD4<Float>(cameraValues.right, 0),
            cameraUp: SIMD4<Float>(cameraValues.up, 0),
            baseRadius: body.baseRadius,
            radiusMultiplier: radiusMultiplier,
            opacity: opacity,
            baseCellCount: UInt32(body.cellCount),
            renderCount: UInt32(boundedRenderCount),
            layerCount: UInt32(simulation.populationLayerCount)
        )
        encoder.setVertexBytes(
            &uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 6)
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: 6,
            instanceCount: boundedRenderCount * simulation.populationLayerCount
        )
        encoder.endEncoding()
    }
}
