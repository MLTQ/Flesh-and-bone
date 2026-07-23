import Foundation
import Metal
import simd

enum RuntimeAssetError: Error, CustomStringConvertible {
    case unreadable(String)
    case invalid(String)
    case allocation(String)

    var description: String {
        switch self {
        case .unreadable(let path): return "cannot read runtime asset: \(path)"
        case .invalid(let reason): return "invalid runtime asset: \(reason)"
        case .allocation(let name): return "Metal buffer allocation failed: \(name)"
        }
    }
}

private final class BinaryReader {
    let data: Data
    private(set) var offset = 0

    init(path: String) throws {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw RuntimeAssetError.unreadable(path)
        }
        self.data = data
    }

    func magic() throws -> String {
        guard offset + 4 <= data.count else { throw RuntimeAssetError.invalid("truncated magic") }
        defer { offset += 4 }
        return String(decoding: data[offset..<(offset + 4)], as: UTF8.self)
    }

    func value<T>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard offset + size <= data.count else { throw RuntimeAssetError.invalid("truncated header") }
        defer { offset += size }
        return data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
    }

    func buffer(device: MTLDevice, bytes: Int, label: String) throws -> MTLBuffer {
        guard bytes >= 0, offset + bytes <= data.count else {
            throw RuntimeAssetError.invalid("truncated \(label)")
        }
        let result: MTLBuffer? = data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base.advanced(by: offset),
                length: bytes,
                options: .storageModeShared
            )
        }
        offset += bytes
        guard let result else { throw RuntimeAssetError.allocation(label) }
        result.label = label
        return result
    }
}

struct RuntimeBody {
    let name: String
    let cellCount: Int
    let boneCount: Int
    let frameCount: Int
    let pitch: Float
    let baseRadius: Float
    let sourceBytes: Int
    let pointValues: [SIMD4<Float>]
    let baseRenderOrder: [UInt32]
    let points: MTLBuffer
    let sourceAnchors: MTLBuffer
    let skinIndices: MTLBuffer
    let skinWeights: MTLBuffer
    let material: MTLBuffer
    let neighbors: MTLBuffer
    let colors: MTLBuffer
    let renderOrder: MTLBuffer
    let skinMatrices: MTLBuffer

    static func load(path: String, device: MTLDevice) throws -> RuntimeBody {
        let reader = try BinaryReader(path: path)
        guard try reader.magic() == "FNB1" else {
            throw RuntimeAssetError.invalid("body magic")
        }
        let version = try reader.value(UInt32.self)
        let cellCount = Int(try reader.value(UInt32.self))
        let boneCount = Int(try reader.value(UInt32.self))
        let frameCount = Int(try reader.value(UInt32.self))
        let pitch = try reader.value(Float.self)
        let baseRadius = try reader.value(Float.self)
        _ = try reader.value(Float.self)
        guard (version == 1 || version == 2),
              cellCount > 0, boneCount > 0, frameCount > 2,
              pitch > 0, baseRadius > 0 else {
            throw RuntimeAssetError.invalid("body header values")
        }
        let points = try reader.buffer(
            device: device, bytes: cellCount * 16, label: "rest points")
        let sourceAnchors = version >= 2
            ? try reader.buffer(
                device: device,
                bytes: cellCount * 16,
                label: "bone source anchors")
            : points
        let indices = try reader.buffer(
            device: device, bytes: cellCount * 8 * 2, label: "skin indices")
        let weights = try reader.buffer(
            device: device, bytes: cellCount * 8 * 4, label: "skin weights")
        let material = try reader.buffer(
            device: device, bytes: cellCount * 16, label: "material")
        let neighbors = try reader.buffer(
            device: device, bytes: cellCount * 8 * 4, label: "neighbors")
        let colors = try reader.buffer(
            device: device, bytes: cellCount * 4, label: "colors")
        let order = try reader.buffer(
            device: device, bytes: cellCount * 4, label: "render order")
        let matrices = try reader.buffer(
            device: device,
            bytes: frameCount * boneCount * 16 * 4,
            label: "skin matrices")
        guard reader.offset == reader.data.count else {
            throw RuntimeAssetError.invalid("unexpected trailing body bytes")
        }
        let pointPointer = points.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: cellCount)
        let orderPointer = order.contents().bindMemory(
            to: UInt32.self, capacity: cellCount)
        return RuntimeBody(
            name: "\(cellCount.formatted()) cells · \(pitch * 1000) mm",
            cellCount: cellCount,
            boneCount: boneCount,
            frameCount: frameCount,
            pitch: pitch,
            baseRadius: baseRadius,
            sourceBytes: reader.data.count,
            pointValues: Array(UnsafeBufferPointer(
                start: pointPointer, count: cellCount)),
            baseRenderOrder: Array(UnsafeBufferPointer(
                start: orderPointer, count: cellCount)),
            points: points,
            sourceAnchors: sourceAnchors,
            skinIndices: indices,
            skinWeights: weights,
            material: material,
            neighbors: neighbors,
            colors: colors,
            renderOrder: order,
            skinMatrices: matrices
        )
    }

    func sortRenderOrder(camera: OrbitCamera, count: Int) {
        let bounded = min(max(count, 1), cellCount)
        let direction = camera.depthDirection
        var selected = Array(baseRenderOrder.prefix(bounded))
        selected.sort { left, right in
            let leftPoint = pointValues[Int(left)]
            let rightPoint = pointValues[Int(right)]
            return simd_dot(
                SIMD3<Float>(leftPoint.x, leftPoint.y, leftPoint.z),
                direction
            ) < simd_dot(
                SIMD3<Float>(rightPoint.x, rightPoint.y, rightPoint.z),
                direction
            )
        }
        let destination = renderOrder.contents().bindMemory(
            to: UInt32.self, capacity: cellCount)
        selected.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: bounded)
        }
    }
}

struct RuntimeModel {
    let hiddenCount: Int
    let inputCount: Int
    let accelerationCap: Float
    let referencePitch: Float
    let sourceBytes: Int
    let values: MTLBuffer

    static func load(path: String, device: MTLDevice) throws -> RuntimeModel {
        let reader = try BinaryReader(path: path)
        guard try reader.magic() == "FNM1" else {
            throw RuntimeAssetError.invalid("model magic")
        }
        let version = try reader.value(UInt32.self)
        let hidden = Int(try reader.value(UInt32.self))
        let input = Int(try reader.value(UInt32.self))
        let cap = try reader.value(Float.self)
        let referencePitch = try reader.value(Float.self)
        _ = try reader.value(Float.self)
        guard version == 1, hidden == 32, input == 5, cap > 0,
              referencePitch > 0 else {
            throw RuntimeAssetError.invalid("unsupported model architecture")
        }
        let floatCount = 5 + 2 + hidden * input + hidden
            + hidden * hidden + hidden + 2 * hidden + 2
        let values = try reader.buffer(
            device: device, bytes: floatCount * 4, label: "H7C model")
        guard reader.offset == reader.data.count else {
            throw RuntimeAssetError.invalid("unexpected trailing model bytes")
        }
        return RuntimeModel(
            hiddenCount: hidden,
            inputCount: input,
            accelerationCap: cap,
            referencePitch: referencePitch,
            sourceBytes: reader.data.count,
            values: values
        )
    }
}

enum RuntimeAssets {
    static func directory() -> String? {
        var candidates: [String] = []
        if let value = ProcessInfo.processInfo.environment["FLESH_ASSETS_DIR"] {
            candidates.append(value)
        }
        candidates.append(
            FileManager.default.currentDirectoryPath + "/runtime/Assets")
        let executable = URL(fileURLWithPath: CommandLine.arguments[0])
            .deletingLastPathComponent()
        candidates.append(
            executable.appendingPathComponent("../Resources").standardized.path)
        return candidates.first { candidate in
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(
                atPath: candidate, isDirectory: &isDirectory
            ) && isDirectory.boolValue
        }
    }

    static func profilePaths(in directory: String) -> [String] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: URL(fileURLWithPath: directory),
            includingPropertiesForKeys: nil
        )) ?? []
        return urls
            .filter { $0.pathExtension == "fnb" }
            .sorted { left, right in
                let l = Int(left.deletingPathExtension().lastPathComponent
                    .split(separator: "_").last ?? "") ?? 0
                let r = Int(right.deletingPathExtension().lastPathComponent
                    .split(separator: "_").last ?? "") ?? 0
                return l < r
            }
            .map(\.path)
    }
}
