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
    private var lastRenderCommand: MTLCommandBuffer?
    private var paintedTemplates = Set<UInt32>()
    private var trackingAreaReference: NSTrackingArea?
    private let brushLayer = CAShapeLayer()
    private(set) var simulation: FleshSimulation
    private(set) var profileIndex: Int
    private(set) var renderCount: Int
    var radiusMultiplier: Float = 1.0
    var opacity: Float = 0.72
    var paused = false
    var camera = OrbitCamera()
    var interactionMode: InteractionMode = .orbit {
        didSet { updateBrushAppearance() }
    }
    var brushRadiusPixels: Float = 44

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
        brushLayer.fillColor = NSColor.clear.cgColor
        brushLayer.lineWidth = 1.5
        brushLayer.isHidden = true
        layer?.addSublayer(brushLayer)
        updateBrushAppearance()
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
        brushLayer.frame = bounds
        updateDrawableSize()
    }

    override func updateTrackingAreas() {
        if let trackingAreaReference {
            removeTrackingArea(trackingAreaReference)
        }
        let tracking = NSTrackingArea(
            rect: .zero,
            options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow,
                      .inVisibleRect],
            owner: self,
            userInfo: nil)
        addTrackingArea(tracking)
        trackingAreaReference = tracking
        super.updateTrackingAreas()
    }

    private func updateDrawableSize() {
        let scale = metalLayer.contentsScale
        metalLayer.drawableSize = CGSize(
            width: max(bounds.width * scale, 1),
            height: max(bounds.height * scale, 1))
    }

    private func tick() {
        if !paused, let compute = queue.makeCommandBuffer() {
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
        lastRenderCommand = render
        render.commit()
    }

    func loadProfile(index: Int) throws {
        guard profilePaths.indices.contains(index) else { return }
        lastRenderCommand?.waitUntilCompleted()
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

    func setCameraPreset(_ preset: CameraPreset) {
        camera.apply(preset)
        simulation.body.sortRenderOrder(camera: camera, count: renderCount)
    }

    private func paint(at location: NSPoint) {
        guard interactionMode != .orbit else { return }
        lastRenderCommand?.waitUntilCompleted()
        let templates = PopulationBrush.selectTemplates(
            simulation: simulation,
            camera: camera,
            viewport: bounds.size,
            location: location,
            radius: brushRadiusPixels,
            mode: interactionMode,
            renderCount: renderCount
        ).filter { !paintedTemplates.contains($0) }
        guard !templates.isEmpty else { return }
        let change = interactionMode == .source
            ? simulation.planSource(templates: templates)
            : simulation.planVacuum(templates: templates)
        guard let change, let command = queue.makeCommandBuffer() else { return }
        command.label = interactionMode == .source
            ? "Paint NCA source"
            : "Paint NCA vacuum"
        simulation.encodePopulationChange(
            commandBuffer: command, change: change)
        command.commit()
        paintedTemplates.formUnion(templates)
    }

    func reset() {
        let previous = simulation
        do {
            lastRenderCommand?.waitUntilCompleted()
            simulation = try FleshSimulation(
                device: device,
                library: library,
                body: previous.body,
                model: model)
            simulation.motionSpeed = previous.motionSpeed
            simulation.motionIntensity = previous.motionIntensity
            simulation.physicsEnabled = previous.physicsEnabled
            simulation.densityEnabled = previous.densityEnabled
        } catch {
            previous.resetDynamics()
        }
    }

    override func mouseDown(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        updateBrushIndicator(at: location)
        if interactionMode == .orbit {
            previousDrag = location
        } else {
            paintedTemplates.removeAll(keepingCapacity: true)
            paint(at: location)
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let current = convert(event.locationInWindow, from: nil)
        updateBrushIndicator(at: current)
        if interactionMode == .orbit, let previousDrag {
            camera.orbit(
                deltaX: Float(current.x - previousDrag.x),
                deltaY: Float(current.y - previousDrag.y))
            simulation.body.sortRenderOrder(camera: camera, count: renderCount)
        } else if interactionMode != .orbit {
            paint(at: current)
        }
        previousDrag = current
    }

    override func mouseUp(with event: NSEvent) {
        previousDrag = nil
        paintedTemplates.removeAll(keepingCapacity: true)
    }

    override func mouseMoved(with event: NSEvent) {
        updateBrushIndicator(at: convert(event.locationInWindow, from: nil))
    }

    override func mouseEntered(with event: NSEvent) {
        updateBrushIndicator(at: convert(event.locationInWindow, from: nil))
    }

    override func mouseExited(with event: NSEvent) {
        brushLayer.isHidden = true
    }

    override func scrollWheel(with event: NSEvent) {
        camera.zoom(delta: Float(event.scrollingDeltaY))
    }

    private func updateBrushAppearance() {
        switch interactionMode {
        case .orbit:
            brushLayer.isHidden = true
        case .source:
            brushLayer.strokeColor = NSColor(
                calibratedRed: 0.35, green: 0.95, blue: 0.58, alpha: 0.95
            ).cgColor
        case .vacuum:
            brushLayer.strokeColor = NSColor(
                calibratedRed: 1.0, green: 0.40, blue: 0.34, alpha: 0.95
            ).cgColor
        }
    }

    private func updateBrushIndicator(at location: NSPoint) {
        guard interactionMode != .orbit else {
            brushLayer.isHidden = true
            return
        }
        let radius = CGFloat(brushRadiusPixels)
        brushLayer.path = CGPath(
            ellipseIn: CGRect(
                x: location.x - radius,
                y: location.y - radius,
                width: radius * 2,
                height: radius * 2),
            transform: nil)
        brushLayer.isHidden = false
    }
}
