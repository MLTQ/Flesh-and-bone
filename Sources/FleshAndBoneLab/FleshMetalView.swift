import AppKit
import Metal
import QuartzCore

final class FleshMetalView: NSView {
    let device: MTLDevice
    let profilePaths: [String]
    let monitor = PerformanceMonitor()
    private let queue: MTLCommandQueue
    private let library: MTLLibrary
    private let renderer: FleshRenderer
    private let model: RuntimeModel
    private var timer: Timer?
    private var previousDrag: NSPoint?
    private(set) var simulation: FleshSimulation
    private(set) var profileIndex: Int
    private(set) var renderCount: Int
    var radiusMultiplier: Float = 1.0
    var opacity: Float = 0.72
    var paused = false
    var camera = OrbitCamera()

    private var metalLayer: CAMetalLayer {
        layer as! CAMetalLayer
    }

    override func makeBackingLayer() -> CALayer {
        let result = CAMetalLayer()
        result.device = device
        result.pixelFormat = .bgra8Unorm
        result.framebufferOnly = true
        result.isOpaque = true
        return result
    }

    init(frame: NSRect, device: MTLDevice, assetDirectory: String) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw RuntimeAssetError.allocation("command queue")
        }
        self.queue = queue
        self.library = try device.makeLibrary(source: fleshMetalSource, options: nil)
        self.renderer = try FleshRenderer(device: device, library: library)
        self.model = try RuntimeModel.load(
            path: assetDirectory + "/h7c_seed7.fnm", device: device)
        let paths = RuntimeAssets.profilePaths(in: assetDirectory)
        guard !paths.isEmpty else {
            throw RuntimeAssetError.invalid("no FNB1 body profiles")
        }
        profilePaths = paths
        profileIndex = paths.count - 1
        let body = try RuntimeBody.load(path: paths[profileIndex], device: device)
        simulation = try FleshSimulation(
            device: device, library: library, body: body, model: model)
        renderCount = body.cellCount
        super.init(frame: frame)
        wantsLayer = true
        body.sortRenderOrder(camera: camera, count: renderCount)
    }

    required init?(coder: NSCoder) {
        fatalError("FleshMetalView is programmatic")
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        metalLayer.contentsScale = window?.backingScaleFactor ?? 2
        updateDrawableSize()
        if timer == nil {
            timer = Timer.scheduledTimer(
                withTimeInterval: 1.0 / 30.0, repeats: true
            ) { [weak self] _ in self?.tick() }
            if let timer { RunLoop.main.add(timer, forMode: .common) }
        }
    }

    override func layout() {
        super.layout()
        updateDrawableSize()
    }

    private func updateDrawableSize() {
        let scale = metalLayer.contentsScale
        metalLayer.drawableSize = CGSize(
            width: max(bounds.width * scale, 1),
            height: max(bounds.height * scale, 1))
    }

    private func tick() {
        guard !paused else { return }
        if let compute = queue.makeCommandBuffer() {
            compute.label = "Flesh NCA frame"
            simulation.encodeFrame(commandBuffer: compute)
            compute.addCompletedHandler { [monitor] command in
                monitor.recordCompute(command)
            }
            compute.commit()
        }
        guard let drawable = metalLayer.nextDrawable(),
              let render = queue.makeCommandBuffer() else { return }
        render.label = "Flesh splat frame"
        renderer.encode(
            commandBuffer: render,
            texture: drawable.texture,
            simulation: simulation,
            renderCount: renderCount,
            radiusMultiplier: radiusMultiplier,
            opacity: opacity,
            camera: camera)
        render.present(drawable)
        render.addCompletedHandler { [monitor] command in
            monitor.recordRender(command)
            monitor.recordPresentedFrame()
        }
        render.commit()
    }

    func loadProfile(index: Int) throws {
        guard profilePaths.indices.contains(index) else { return }
        let body = try RuntimeBody.load(path: profilePaths[index], device: device)
        simulation = try FleshSimulation(
            device: device, library: library, body: body, model: model)
        profileIndex = index
        renderCount = body.cellCount
        body.sortRenderOrder(camera: camera, count: renderCount)
    }

    func setRenderCount(_ value: Int) {
        renderCount = min(max(value, 1), simulation.body.cellCount)
        simulation.body.sortRenderOrder(camera: camera, count: renderCount)
    }

    func reset() {
        simulation.resetDynamics()
    }

    override func mouseDown(with event: NSEvent) {
        previousDrag = convert(event.locationInWindow, from: nil)
    }

    override func mouseDragged(with event: NSEvent) {
        let current = convert(event.locationInWindow, from: nil)
        if let previousDrag {
            camera.orbit(
                deltaX: Float(current.x - previousDrag.x),
                deltaY: Float(current.y - previousDrag.y))
            simulation.body.sortRenderOrder(camera: camera, count: renderCount)
        }
        previousDrag = current
    }

    override func mouseUp(with event: NSEvent) {
        previousDrag = nil
    }

    override func scrollWheel(with event: NSEvent) {
        camera.zoom(delta: Float(event.scrollingDeltaY))
    }
}
