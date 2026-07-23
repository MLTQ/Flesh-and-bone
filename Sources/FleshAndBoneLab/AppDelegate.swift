import AppKit
import Metal

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw RuntimeAssetError.invalid("Metal is unavailable")
            }
            guard let assetDirectory = RuntimeAssets.directory() else {
                throw RuntimeAssetError.invalid(
                    "runtime/Assets not found; run scripts/export_flesh_runtime.py")
            }
            let metalView = try FleshMetalView(
                frame: .zero, device: device, assetDirectory: assetDirectory)
            metalView.translatesAutoresizingMaskIntoConstraints = false
            let controls = ControlPanel(metalView: metalView)
            let root = NSView()
            root.wantsLayer = true
            root.layer?.backgroundColor = NSColor.black.cgColor
            root.addSubview(metalView)
            root.addSubview(controls)
            NSLayoutConstraint.activate([
                metalView.leadingAnchor.constraint(equalTo: root.leadingAnchor),
                metalView.topAnchor.constraint(equalTo: root.topAnchor),
                metalView.bottomAnchor.constraint(equalTo: root.bottomAnchor),
                metalView.trailingAnchor.constraint(equalTo: controls.leadingAnchor),
                controls.trailingAnchor.constraint(equalTo: root.trailingAnchor),
                controls.topAnchor.constraint(equalTo: root.topAnchor),
                controls.bottomAnchor.constraint(equalTo: root.bottomAnchor),
                controls.widthAnchor.constraint(equalToConstant: 330),
            ])
            let window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false)
            window.title = "Flesh & Bone Lab"
            window.contentView = root
            window.minSize = NSSize(width: 900, height: 640)
            window.center()
            window.makeKeyAndOrderFront(nil)
            self.window = window
            NSApp.activate(ignoringOtherApps: true)
        } catch {
            let alert = NSAlert()
            alert.messageText = "Flesh & Bone Lab could not start"
            alert.informativeText = String(describing: error)
            alert.runModal()
            NSApp.terminate(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(
        _ sender: NSApplication
    ) -> Bool {
        true
    }
}
