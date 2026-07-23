import CoreGraphics
import simd

enum InteractionMode: Int {
    case orbit
    case source
    case vacuum
}

enum PopulationBrush {
    private struct Candidate {
        let template: UInt32
        let cameraDepth: Float
    }

    static func selectTemplates(
        simulation: FleshSimulation,
        camera: OrbitCamera,
        viewport: CGSize,
        location: CGPoint,
        radius: Float,
        mode: InteractionMode,
        renderCount: Int
    ) -> [UInt32] {
        guard mode != .orbit, viewport.width > 0, viewport.height > 0 else {
            return []
        }
        let bounded = min(max(renderCount, 1), simulation.body.cellCount)
        let order = simulation.body.baseRenderOrder.prefix(bounded)
        let positions = simulation.lbsCurrent.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: simulation.slotCount)
        let residuals = simulation.residual.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: simulation.slotCount)
        let aspect = Float(viewport.width / viewport.height)
        let matrix = camera.matrices(aspect: aspect).viewProjection
        let width = Float(viewport.width)
        let height = Float(viewport.height)
        let center = SIMD2<Float>(Float(location.x), Float(location.y))
        let radiusSquared = radius * radius
        var candidates: [Candidate] = []
        candidates.reserveCapacity(min(bounded, 4096))

        for raw in order {
            let template = Int(raw)
            let slot: Int
            switch mode {
            case .source:
                guard simulation.canSource(template: template) else { continue }
                slot = template
            case .vacuum:
                guard let active = simulation.representativeSlot(
                    template: template) else { continue }
                slot = active
            case .orbit:
                continue
            }
            let target = positions[slot]
            let offset = mode == .vacuum ? residuals[slot] : SIMD4<Float>.zero
            let world = SIMD4<Float>(
                target.x + offset.x,
                target.y + offset.y,
                target.z + offset.z,
                1)
            let clip = matrix * world
            guard clip.w > 1.0e-6 else { continue }
            let inverseW = 1 / clip.w
            let screen = SIMD2<Float>(
                (clip.x * inverseW * 0.5 + 0.5) * width,
                (clip.y * inverseW * 0.5 + 0.5) * height)
            guard simd_distance_squared(screen, center) <= radiusSquared else {
                continue
            }
            candidates.append(Candidate(
                template: raw,
                cameraDepth: simd_dot(
                    SIMD3<Float>(world.x, world.y, world.z),
                    camera.depthDirection)))
        }

        guard let front = candidates.map(\.cameraDepth).max() else { return [] }
        let depthTolerance = max(2.5 * simulation.body.pitch, 0.035)
        return candidates.compactMap {
            front - $0.cameraDepth <= depthTolerance ? $0.template : nil
        }
    }
}
