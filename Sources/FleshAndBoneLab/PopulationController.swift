import Foundation

struct PopulationChange {
    let indices: [UInt32]
    let activate: Bool

    var count: Int {
        indices.count
    }
}

final class PopulationController {
    let baseCount: Int
    let layerCount: Int
    let capacity: Int
    private(set) var activeCount: Int
    private var occupancy: [UInt8]

    init(baseCount: Int, layerCount: Int = 2) {
        self.baseCount = max(baseCount, 1)
        self.layerCount = max(layerCount, 1)
        capacity = self.baseCount * self.layerCount
        occupancy = [UInt8](repeating: 1, count: self.baseCount)
        activeCount = self.baseCount
    }

    var baselineFraction: Double {
        Double(activeCount) / Double(baseCount)
    }

    func occupancyCount(template: Int) -> Int {
        guard (0..<baseCount).contains(template) else { return 0 }
        return Int(occupancy[template])
    }

    func representativeSlot(template: Int) -> Int? {
        let count = occupancyCount(template: template)
        guard count > 0 else { return nil }
        return (count - 1) * baseCount + template
    }

    func canSource(template: Int) -> Bool {
        occupancyCount(template: template) < layerCount
    }

    func planSource(templates: [UInt32]) -> PopulationChange? {
        var seen = Set<UInt32>()
        var indices: [UInt32] = []
        for raw in templates where seen.insert(raw).inserted {
            let template = Int(raw)
            guard (0..<baseCount).contains(template) else { continue }
            let layer = Int(occupancy[template])
            guard layer < layerCount else { continue }
            indices.append(UInt32(layer * baseCount + template))
            occupancy[template] += 1
        }
        guard !indices.isEmpty else { return nil }
        activeCount += indices.count
        return PopulationChange(indices: indices, activate: true)
    }

    func planVacuum(templates: [UInt32]) -> PopulationChange? {
        var seen = Set<UInt32>()
        var indices: [UInt32] = []
        for raw in templates where seen.insert(raw).inserted {
            let template = Int(raw)
            guard (0..<baseCount).contains(template) else { continue }
            let count = Int(occupancy[template])
            guard count > 0 else { continue }
            indices.append(UInt32((count - 1) * baseCount + template))
            occupancy[template] -= 1
        }
        guard !indices.isEmpty else { return nil }
        activeCount -= indices.count
        return PopulationChange(indices: indices, activate: false)
    }
}
