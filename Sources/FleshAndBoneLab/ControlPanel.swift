import AppKit

final class ControlPanel: NSView {
    private let metalView: FleshMetalView
    private let profilePopup = NSPopUpButton()
    private let renderSlider = NSSlider(
        value: 1, minValue: 0.03, maxValue: 1, target: nil, action: nil)
    private let renderValue = NSTextField(labelWithString: "")
    private let radiusSlider = NSSlider(
        value: 1, minValue: 0.35, maxValue: 2, target: nil, action: nil)
    private let radiusValue = NSTextField(labelWithString: "1.00×")
    private let opacitySlider = NSSlider(
        value: 0.72, minValue: 0.1, maxValue: 1, target: nil, action: nil)
    private let opacityValue = NSTextField(labelWithString: "0.72")
    private let speedSlider = NSSlider(
        value: 1, minValue: 0, maxValue: 2, target: nil, action: nil)
    private let speedValue = NSTextField(labelWithString: "1.00×")
    private let intensitySlider = NSSlider(
        value: 1, minValue: 0, maxValue: 1.5, target: nil, action: nil)
    private let intensityValue = NSTextField(labelWithString: "1.00×")
    private let physicsCheck = NSButton(
        checkboxWithTitle: "H6C + H7C physics", target: nil, action: nil)
    private let densityCheck = NSButton(
        checkboxWithTitle: "H7C density residual", target: nil, action: nil)
    private let pauseButton = NSButton(
        title: "Pause", target: nil, action: nil)
    private let stats = NSTextField(wrappingLabelWithString: "")
    private var statsTimer: Timer?

    init(metalView: FleshMetalView) {
        self.metalView = metalView
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = NSColor(
            calibratedWhite: 0.075, alpha: 1).cgColor
        translatesAutoresizingMaskIntoConstraints = false
        build()
        updateProfileControls()
        statsTimer = Timer.scheduledTimer(
            withTimeInterval: 0.25, repeats: true
        ) { [weak self] _ in self?.updateStats() }
        if let statsTimer { RunLoop.main.add(statsTimer, forMode: .common) }
    }

    required init?(coder: NSCoder) {
        fatalError("ControlPanel is programmatic")
    }

    private func heading(_ value: String, size: CGFloat, weight: NSFont.Weight)
        -> NSTextField {
        let result = NSTextField(labelWithString: value)
        result.font = .systemFont(ofSize: size, weight: weight)
        result.textColor = .white
        return result
    }

    private func caption(_ value: String) -> NSTextField {
        let result = NSTextField(wrappingLabelWithString: value)
        result.font = .systemFont(ofSize: 11)
        result.textColor = .secondaryLabelColor
        return result
    }

    private func row(title: String, slider: NSSlider, value: NSTextField)
        -> NSStackView {
        slider.isContinuous = true
        let titleLabel = NSTextField(labelWithString: title)
        titleLabel.font = .systemFont(ofSize: 12, weight: .medium)
        value.font = .monospacedDigitSystemFont(ofSize: 11, weight: .regular)
        value.alignment = .right
        value.setContentHuggingPriority(.required, for: .horizontal)
        let labels = NSStackView(views: [titleLabel, value])
        labels.orientation = .horizontal
        labels.distribution = .fill
        let result = NSStackView(views: [labels, slider])
        result.orientation = .vertical
        result.spacing = 4
        return result
    }

    private func build() {
        let title = heading("Flesh & Bone Lab", size: 22, weight: .bold)
        let subtitle = caption(
            "Live sparse-NCA body · fixed skeleton motion · native Metal")
        profilePopup.target = self
        profilePopup.action = #selector(profileChanged)
        for path in metalView.profilePaths {
            let count = Int(
                URL(fileURLWithPath: path).deletingPathExtension()
                    .lastPathComponent.split(separator: "_").last ?? "") ?? 0
            profilePopup.addItem(withTitle: "\(count.formatted()) cells")
        }

        renderSlider.target = self
        renderSlider.action = #selector(renderChanged)
        radiusSlider.target = self
        radiusSlider.action = #selector(radiusChanged)
        opacitySlider.target = self
        opacitySlider.action = #selector(opacityChanged)
        speedSlider.target = self
        speedSlider.action = #selector(speedChanged)
        intensitySlider.target = self
        intensitySlider.action = #selector(intensityChanged)
        physicsCheck.state = .on
        physicsCheck.target = self
        physicsCheck.action = #selector(physicsChanged)
        densityCheck.state = .on
        densityCheck.target = self
        densityCheck.action = #selector(densityChanged)

        pauseButton.bezelStyle = .rounded
        pauseButton.target = self
        pauseButton.action = #selector(togglePause)
        let reset = NSButton(title: "Cold reset", target: self, action: #selector(reset))
        reset.bezelStyle = .rounded
        let buttons = NSStackView(views: [pauseButton, reset])
        buttons.orientation = .horizontal
        buttons.distribution = .fillEqually

        let warning = caption(
            "Physical resolution changes the body graph and NCA workload. "
            + "Render count, radius, and opacity change visuals only.")
        warning.textColor = NSColor(
            calibratedRed: 0.76, green: 0.72, blue: 0.48, alpha: 1)
        stats.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        stats.textColor = NSColor(
            calibratedRed: 0.66, green: 0.90, blue: 0.76, alpha: 1)

        let stack = NSStackView(views: [
            title,
            subtitle,
            heading("Physical profile", size: 12, weight: .semibold),
            profilePopup,
            row(title: "Rendered cells", slider: renderSlider, value: renderValue),
            row(title: "Splat radius", slider: radiusSlider, value: radiusValue),
            row(title: "Opacity", slider: opacitySlider, value: opacityValue),
            heading("Broadcast motion intent", size: 12, weight: .semibold),
            row(title: "Speed", slider: speedSlider, value: speedValue),
            row(title: "Intensity", slider: intensitySlider, value: intensityValue),
            physicsCheck,
            densityCheck,
            buttons,
            warning,
            heading("Live performance", size: 12, weight: .semibold),
            stats,
        ])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        for view in stack.views {
            view.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        }
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 22),
        ])
    }

    private func updateProfileControls() {
        profilePopup.selectItem(at: metalView.profileIndex)
        renderSlider.doubleValue = 1
        renderChanged()
    }

    @objc private func profileChanged() {
        do {
            try metalView.loadProfile(index: profilePopup.indexOfSelectedItem)
            metalView.simulation.motionSpeed = Float(speedSlider.doubleValue)
            metalView.simulation.motionIntensity = Float(intensitySlider.doubleValue)
            metalView.simulation.physicsEnabled = physicsCheck.state == .on
            metalView.simulation.densityEnabled = densityCheck.state == .on
            updateProfileControls()
        } catch {
            let alert = NSAlert()
            alert.messageText = "Could not load body profile"
            alert.informativeText = String(describing: error)
            alert.runModal()
        }
    }

    @objc private func renderChanged() {
        let count = max(
            1,
            Int((renderSlider.doubleValue
                 * Double(metalView.simulation.body.cellCount)).rounded()))
        metalView.setRenderCount(count)
        renderValue.stringValue = count.formatted()
    }

    @objc private func radiusChanged() {
        metalView.radiusMultiplier = Float(radiusSlider.doubleValue)
        radiusValue.stringValue = String(format: "%.2f×", radiusSlider.doubleValue)
    }

    @objc private func opacityChanged() {
        metalView.opacity = Float(opacitySlider.doubleValue)
        opacityValue.stringValue = String(format: "%.2f", opacitySlider.doubleValue)
    }

    @objc private func speedChanged() {
        metalView.simulation.motionSpeed = Float(speedSlider.doubleValue)
        speedValue.stringValue = String(format: "%.2f×", speedSlider.doubleValue)
    }

    @objc private func intensityChanged() {
        metalView.simulation.motionIntensity = Float(intensitySlider.doubleValue)
        intensityValue.stringValue = String(
            format: "%.2f×", intensitySlider.doubleValue)
    }

    @objc private func physicsChanged() {
        metalView.simulation.physicsEnabled = physicsCheck.state == .on
    }

    @objc private func densityChanged() {
        metalView.simulation.densityEnabled = densityCheck.state == .on
    }

    @objc private func togglePause() {
        metalView.paused.toggle()
        pauseButton.title = metalView.paused ? "Resume" : "Pause"
    }

    @objc private func reset() {
        metalView.reset()
    }

    private func updateStats() {
        let snapshot = metalView.monitor.snapshot()
        let body = metalView.simulation.body
        let staticMB = Double(
            body.sourceBytes + metalView.simulation.model.sourceBytes) / 1_000_000
        let dynamicMB = Double(metalView.simulation.dynamicBytes) / 1_000_000
        stats.stringValue = String(
            format:
                "compute  %6.2f ms  (%4.1f× real-time)\\n"
                + "render   %6.2f ms  (%4.1f fps)\\n"
                + "static   %6.1f MB\\n"
                + "dynamic  %6.1f MB\\n"
                + "pitch    %6.1f mm",
            snapshot.computeMilliseconds,
            snapshot.computeRealtimeMultiple,
            snapshot.renderMilliseconds,
            snapshot.framesPerSecond,
            staticMB,
            dynamicMB,
            body.pitch * 1000
        )
    }
}
