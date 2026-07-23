import Foundation
import Metal

struct PerformanceSnapshot {
    let computeMilliseconds: Double
    let renderMilliseconds: Double
    let framesPerSecond: Double
    let computeRealtimeMultiple: Double
}

final class PerformanceMonitor: @unchecked Sendable {
    private let lock = NSLock()
    private var computeMilliseconds = 0.0
    private var renderMilliseconds = 0.0
    private var recentFrames: [TimeInterval] = []

    private func blend(_ old: Double, _ new: Double) -> Double {
        old == 0 ? new : old * 0.88 + new * 0.12
    }

    func recordCompute(_ commandBuffer: MTLCommandBuffer) {
        guard commandBuffer.gpuEndTime > commandBuffer.gpuStartTime else { return }
        let milliseconds = (
            commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        lock.lock()
        computeMilliseconds = blend(computeMilliseconds, milliseconds)
        lock.unlock()
    }

    func recordRender(_ commandBuffer: MTLCommandBuffer) {
        guard commandBuffer.gpuEndTime > commandBuffer.gpuStartTime else { return }
        let milliseconds = (
            commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        lock.lock()
        renderMilliseconds = blend(renderMilliseconds, milliseconds)
        lock.unlock()
    }

    func recordPresentedFrame() {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        recentFrames.append(now)
        recentFrames.removeAll { now - $0 > 1.0 }
        lock.unlock()
    }

    func snapshot() -> PerformanceSnapshot {
        lock.lock()
        defer { lock.unlock() }
        let fps: Double
        if recentFrames.count > 1, let first = recentFrames.first,
           let last = recentFrames.last, last > first {
            fps = Double(recentFrames.count - 1) / (last - first)
        } else {
            fps = 0
        }
        return PerformanceSnapshot(
            computeMilliseconds: computeMilliseconds,
            renderMilliseconds: renderMilliseconds,
            framesPerSecond: fps,
            computeRealtimeMultiple: computeMilliseconds > 0
                ? (1000.0 / 30.0) / computeMilliseconds
                : 0
        )
    }
}
