import CoreGraphics
import simd

enum CameraPreset: String, CaseIterable {
    case front
    case left
    case back
    case right

    var title: String {
        rawValue.capitalized
    }

    var azimuth: Float {
        switch self {
        case .front: return 0
        case .left: return -.pi / 2
        case .back: return .pi
        case .right: return .pi / 2
        }
    }
}

struct OrbitCamera {
    var azimuth: Float = 0.28
    var elevation: Float = 0.08
    var distance: Float = 4.15
    var target = SIMD3<Float>(0, 1.02, 0)

    var depthDirection: SIMD3<Float> {
        let cosineElevation = cos(elevation)
        return simd_normalize(SIMD3<Float>(
            sin(azimuth) * cosineElevation,
            sin(elevation),
            cos(azimuth) * cosineElevation
        ))
    }

    mutating func orbit(deltaX: Float, deltaY: Float) {
        azimuth += deltaX * 0.008
        elevation = min(max(elevation + deltaY * 0.008, -1.1), 1.1)
    }

    mutating func zoom(delta: Float) {
        distance = min(max(distance * exp(delta * 0.025), 1.5), 7.0)
    }

    mutating func apply(_ preset: CameraPreset) {
        azimuth = preset.azimuth
        elevation = 0.08
    }

    func matrices(aspect: Float) -> (
        viewProjection: simd_float4x4,
        right: SIMD3<Float>,
        up: SIMD3<Float>
    ) {
        let eye = target + distance * depthDirection
        let backward = simd_normalize(eye - target)
        let right = simd_normalize(simd_cross(
            SIMD3<Float>(0, 1, 0), backward))
        let up = simd_cross(backward, right)
        let view = simd_float4x4(columns: (
            SIMD4<Float>(right.x, up.x, backward.x, 0),
            SIMD4<Float>(right.y, up.y, backward.y, 0),
            SIMD4<Float>(right.z, up.z, backward.z, 0),
            SIMD4<Float>(
                -simd_dot(right, eye),
                -simd_dot(up, eye),
                -simd_dot(backward, eye),
                1)
        ))
        let fieldOfView: Float = 36 * .pi / 180
        let y = 1 / tan(fieldOfView * 0.5)
        let x = y / max(aspect, 0.01)
        let near: Float = 0.05
        let far: Float = 20
        let z = far / (near - far)
        let projection = simd_float4x4(columns: (
            SIMD4<Float>(x, 0, 0, 0),
            SIMD4<Float>(0, y, 0, 0),
            SIMD4<Float>(0, 0, z, -1),
            SIMD4<Float>(0, 0, z * near, 0)
        ))
        return (projection * view, right, up)
    }
}
