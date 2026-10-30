import AppKit
import ApplicationServices
import ScreenCaptureKit
import Carbon.HIToolbox

/// Explicit, window-scoped demonstration recorder. No executor, model, clipboard or background start.
/// The desktop bridge must invoke present() only from its authenticated main-frame user action.
@available(macOS 14.0, *)
final class OperationRecorder: NSObject {
    private var target: SCWindow?
    private var session: OperationRecordingSession?
    private var monitor: Any?
    private var timer: Timer?
    private var typingTimer: Timer?
    private var panel: NSPanel?
    private var status = NSTextField(labelWithString: "")
    private var paused = true
    private var capturing = false
    private var choosing = false
    private var generation = UUID()
    private var latest: OperationFrame?
    private var pending: (frame: OperationFrame, action: [String: Any])?
    private var restartRequired = false
    private let eventEpoch = Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
    var onExport: ((Data) -> Void)?
    var onStatus: ((String) -> Void)?

    /// Does not request screen/AX access automatically. A denied permission cannot start recording.
    func present() {
        guard session == nil else { panel?.makeKeyAndOrderFront(nil); return }
        guard !choosing else { return }
        guard CGPreflightScreenCaptureAccess(), AXIsProcessTrusted() else {
            alert("请先在系统设置中允许鲁班的屏幕录制和辅助功能，再返回开始。当前没有录制。")
            return
        }
        choosing = true
        Task { @MainActor in
            defer { self.choosing = false }
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
                let windows = content.windows.filter {
                    $0.windowLayer == 0 && $0.frame.width > 40 && $0.frame.height > 40
                    && $0.owningApplication?.processID != ProcessInfo.processInfo.processIdentifier
                    && $0.owningApplication?.bundleIdentifier.isEmpty == false
                }
                guard !windows.isEmpty else { throw OperationRecordingError("没有可选择的应用窗口。") }
                self.choose(windows)
            } catch { self.alert(error.localizedDescription) }
        }
    }

    private func choose(_ windows: [SCWindow]) {
        let dialog = NSAlert()
        dialog.messageText = "录制操作示范"
        dialog.informativeText = "只记录所选窗口。请使用不含私人资料的示范窗口；密码、未知输入和快捷键会暂停。不会读取剪贴板。"
        dialog.addButton(withTitle: "开始录制")
        dialog.addButton(withTitle: "取消")
        let view = NSStackView()
        view.orientation = .vertical; view.alignment = .leading; view.spacing = 8
        let picker = NSPopUpButton(frame: NSRect(x: 0, y: 0, width: 460, height: 28))
        windows.forEach { picker.addItem(withTitle: "\($0.owningApplication!.applicationName)：\($0.title ?? "未命名窗口")") }
        let goal = NSTextField(string: "")
        goal.placeholderString = "这次要完成的任务"
        [picker, goal].forEach { view.addArrangedSubview($0); $0.widthAnchor.constraint(equalToConstant: 460).isActive = true }
        dialog.accessoryView = view
        view.frame = NSRect(x: 0, y: 0, width: 460, height: view.fittingSize.height)
        guard dialog.runModal() == .alertFirstButtonReturn else { return }
        do {
            let selected = windows[picker.indexOfSelectedItem]
            // An ambiguous duplicate cannot bind the focused AX window to a unique capture target.
            guard windows.filter({ $0.owningApplication?.processID == selected.owningApplication?.processID && $0.frame == selected.frame }).count == 1 else {
                throw OperationRecordingError("窗口位置不唯一，请移动所选窗口后重试。")
            }
            session = try OperationRecordingSession(appID: selected.owningApplication!.bundleIdentifier,
                goal: goal.stringValue, family: selected.owningApplication!.bundleIdentifier, now: now())
            target = selected
            paused = true
            restartRequired = false
            showControls()
            status.stringValue = "准备所选窗口；若未自动开始，请切到所选窗口再继续。"
            onStatus?("录制已开始；停止并核对后仅保存示范，尚不训练或执行。")
            // This follows the explicit Start confirmation, never app launch or tab selection.
            let token = generation
            NSRunningApplication(processIdentifier: selected.owningApplication!.processID)?.activate(options: [])
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                guard let self, self.session != nil, self.generation == token else { return }
                self.resume()
            }
        } catch { session = nil; target = nil; alert(error.localizedDescription) }
    }

    private func showControls() {
        let box = NSStackView(); box.orientation = .vertical; box.alignment = .leading; box.spacing = 10
        status = NSTextField(labelWithString: "")
        status.maximumNumberOfLines = 3; status.preferredMaxLayoutWidth = 360
        box.addArrangedSubview(status)
        let row = NSStackView()
        for (title, selector) in [("继续", #selector(resume)), ("暂停", #selector(pauseClicked)),
                                  ("停止并核对", #selector(stopClicked)), ("丢弃", #selector(discardClicked))] {
            row.addArrangedSubview(NSButton(title: title, target: self, action: selector))
        }
        box.addArrangedSubview(row)
        let value = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 440, height: 120),
                            styleMask: [.titled, .nonactivatingPanel], backing: .buffered, defer: false)
        value.title = "鲁班 · 操作示范录制"; value.level = .floating
        value.isFloatingPanel = true; value.hidesOnDeactivate = false
        box.frame = NSRect(x: 15, y: 15, width: 410, height: 90)
        value.contentView?.addSubview(box); value.center(); value.orderFrontRegardless(); panel = value
    }

    @objc private func resume() {
        guard session != nil, CGPreflightScreenCaptureAccess(), AXIsProcessTrusted() else {
            pause("权限已失效；没有继续录制。"); return
        }
        guard safeFocus(typing: false) else { pause("请先切到所选窗口，并离开密码或未知输入区域。") ; return }
        if restartRequired, let old = session {
            // Never join an unobserved interval to an apparently continuous demonstration.
            session = try? OperationRecordingSession(appID: old.appID, goal: old.goal, family: old.family, now: now())
            restartRequired = false
        }
        paused = false; latest = nil; pending = nil; generation = UUID()
        status.stringValue = "正在录制所选窗口；暂停后继续会重新开始完整示范。"
        if monitor == nil {
            monitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .leftMouseUp, .rightMouseDown,
                .rightMouseUp, .leftMouseDragged, .rightMouseDragged, .keyDown, .scrollWheel]) { [weak self] event in self?.observe(event) }
        }
        guard monitor != nil else { pause("无法监听示范操作；当前已暂停。") ; return }
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.20, repeats: true) { [weak self] _ in self?.capture() }
        capture()
    }

    @objc private func pauseClicked() { pause("已暂停；不会采集画面或按键。") }
    @objc private func discardClicked() { close(); onStatus?("示范已丢弃，没有保存或训练。") }
    @objc private func stopClicked() {
        if pending?.action["type"] as? String == "type" { flushTyping() }
        guard var value = session else { return }
        let interrupted = pending != nil || restartRequired
        close(); session = nil; target = nil
        guard !interrupted else { alert("停止时有未完成动作，本次示范已丢弃。请重新录制完整任务。") ; return }
        do { try value.stop(at: now()); review(value) } catch { alert(error.localizedDescription) }
    }

    private func pause(_ reason: String) {
        if !paused { restartRequired = true }
        paused = true; latest = nil; pending = nil; generation = UUID()
        timer?.invalidate(); timer = nil
        typingTimer?.invalidate(); typingTimer = nil
        if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil }
        status.stringValue = reason
    }

    /// Call from desktop window/application termination too. Unreviewed data remains memory-only.
    func close() {
        pause("已停止。"); panel?.orderOut(nil); panel = nil
        session = nil; target = nil
    }

    private func safeFocus(typing: Bool) -> Bool {
        guard let target, let pid = target.owningApplication?.processID,
              NSWorkspace.shared.frontmostApplication?.processIdentifier == pid,
              CGPreflightScreenCaptureAccess(), AXIsProcessTrusted() else { return false }
        let app = AXUIElementCreateApplication(pid)
        guard let window = ax(app, kAXFocusedWindowAttribute) as! AXUIElement?,
              let position = ax(window, kAXPositionAttribute) as! AXValue?,
              let size = ax(window, kAXSizeAttribute) as! AXValue? else { return false }
        var point = CGPoint.zero, extent = CGSize.zero
        guard AXValueGetValue(position, .cgPoint, &point), AXValueGetValue(size, .cgSize, &extent),
              abs(point.x - target.frame.minX) <= 1, abs(point.y - target.frame.minY) <= 1,
              abs(extent.width - target.frame.width) <= 1, abs(extent.height - target.frame.height) <= 1,
              let element = ax(app, kAXFocusedUIElementAttribute) as! AXUIElement?,
              let role = ax(element, kAXRoleAttribute) as? String else { return false }
        let subrole = ax(element, kAXSubroleAttribute) as? String ?? ""
        let protected = ax(element, "AXProtectedContent") as? Bool ?? false
        return OperationRecordingSession.safeRole(role, subrole: subrole, protected: protected, typing: typing)
    }

    private func ax(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
    }

    private func capture() {
        guard !paused, !capturing, let target else { return }
        guard safeFocus(typing: false) else { pause("已离开所选窗口，或焦点含敏感/未知内容；已暂停。") ; return }
        capturing = true
        let token = generation
        let filter = SCContentFilter(desktopIndependentWindow: target)
        let config = SCStreamConfiguration()
        config.width = Int(target.frame.width * 2); config.height = Int(target.frame.height * 2)
        config.showsCursor = false; config.capturesAudio = false; config.ignoreShadowsSingleWindow = true
        Task { @MainActor in
            defer { self.capturing = false }
            do {
                let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
                guard !self.paused, self.generation == token else { return }
                guard self.safeFocus(typing: false) else { self.pause("窗口或输入焦点改变；已暂停。") ; return }
                self.latest = try OperationFrame(image: image, capturedAt: self.now(), bounds: target.frame)
            } catch { self.pause("画面采集失败；已暂停：\(error.localizedDescription)") }
        }
    }

    private func observe(_ event: NSEvent) {
        guard !paused, var value = session else { return }
        if (event.type == .leftMouseDown || event.type == .rightMouseDown) && event.clickCount != 1 {
            // NSEvent counts cumulatively: retaining count 1 and then count 2 would replay three clicks.
            pause("连续多击尚不支持；本次轨迹已丢弃，继续会重新开始示范。")
            session = try? OperationRecordingSession(appID: value.appID, goal: value.goal, family: value.family, now: now())
            return
        }
        guard event.type != .leftMouseDragged && event.type != .rightMouseDragged else {
            pause("拖拽动作尚不支持；已暂停，继续将重新开始示范。"); return
        }
        let typing = event.type == .keyDown && !OperationRecordingSession.navigationKeys.keys.contains(Int(event.keyCode))
        guard safeFocus(typing: typing) else { pause("敏感/未知输入或离开目标窗口；已暂停。") ; return }
        let modifiers = event.modifierFlags.intersection([.command, .control, .option, .function])
        guard modifiers.isEmpty else { pause("快捷键可能访问剪贴板或其他窗口；已暂停且未读取内容。") ; return }
        guard typing || !event.modifierFlags.contains(.shift) else { pause("带修饰键的动作尚不支持；已暂停。") ; return }
        if typing {
            let source = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
            guard let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceType),
                  Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() == kTISTypeKeyboardLayout else {
                pause("输入法组合输入尚不能准确录制；已暂停且未读取按键文字。"); return
            }
        }
        let at = Int64((eventEpoch + event.timestamp) * 1000)
        do {
            if typing, var item = pending, item.action["type"] as? String == "type" {
                guard let text = event.characters, !text.isEmpty,
                      let old = item.action["text"] as? String else { throw OperationRecordingError("输入无法确定；已暂停。") }
                item.action["text"] = old + text; item.action["finished_at_ms"] = at
                try value.validate(frame: item.frame, action: item.action)
                pending = item; scheduleTyping(); return
            }
            if pending?.action["type"] as? String == "type" {
                flushTyping()
                guard !paused, let fresh = session else { return }
                value = fresh
            }
            if event.type == .leftMouseUp || event.type == .rightMouseUp {
                guard var action = pending?.action, let frame = pending?.frame else { return }
                action["finished_at_ms"] = at
                try value.append(frame: frame, action: action)
                session = value; pending = nil; latest = nil
                status.stringValue = "正在录制；已记录 \(value.count) 个完整动作。"
                return
            }
            guard pending == nil else { throw OperationRecordingError("动作重叠；本次未完成动作已丢弃。") }
            guard let frame = latest else { throw OperationRecordingError("尚无动作前画面；请继续后等待画面就绪。") }
            var action: [String: Any] = ["type": "", "started_at_ms": at, "finished_at_ms": at]
            switch event.type {
            case .leftMouseDown, .rightMouseDown:
                let point = event.cgEvent?.location ?? CGPoint(x: -1, y: -1)
                let x = Int((point.x - frame.bounds.minX) * Double(frame.width) / frame.bounds.width)
                let y = Int((point.y - frame.bounds.minY) * Double(frame.height) / frame.bounds.height)
                action.merge(["type": "click", "x": x, "y": y,
                              "button": event.type == .leftMouseDown ? "left" : "right", "count": event.clickCount]) { _, b in b }
                try value.validate(frame: frame, action: action)
                pending = (frame, action); return
            case .scrollWheel:
                guard !event.hasPreciseScrollingDeltas else {
                    // Fractional gesture deltas cannot be silently relabeled as wheel units.
                    throw OperationRecordingError("触控板连续滚动尚不支持；已暂停。")
                }
                guard event.scrollingDeltaX.isFinite, event.scrollingDeltaY.isFinite,
                      event.scrollingDeltaX.rounded() == event.scrollingDeltaX,
                      event.scrollingDeltaY.rounded() == event.scrollingDeltaY else {
                    throw OperationRecordingError("非整数滚动单位尚不支持；已暂停。")
                }
                let point = event.cgEvent?.location ?? CGPoint(x: -1, y: -1)
                action.merge(["type": "scroll", "x": Int((point.x - frame.bounds.minX) * Double(frame.width) / frame.bounds.width),
                    "y": Int((point.y - frame.bounds.minY) * Double(frame.height) / frame.bounds.height),
                    "delta_x": Int(event.scrollingDeltaX), "delta_y": Int(event.scrollingDeltaY)]) { _, b in b }
            case .keyDown:
                if let key = OperationRecordingSession.navigationKeys[Int(event.keyCode)] {
                    action.merge(["type": "key", "key": key]) { _, b in b }
                } else {
                    // Reading characters happens only after positive editable-role and nonsecure checks.
                    guard let text = event.characters, !text.isEmpty else { throw OperationRecordingError("输入无法确定；已暂停。") }
                    action.merge(["type": "type", "text": text, "sensitive": false]) { _, b in b }
                    try value.validate(frame: frame, action: action)
                    pending = (frame, action); scheduleTyping(); return
                }
            default: return
            }
            try value.append(frame: frame, action: action)
            session = value; latest = nil
            status.stringValue = "正在录制；已记录 \(value.count) 个完整动作。"
        } catch { pause(error.localizedDescription) }
    }

    private func scheduleTyping() {
        typingTimer?.invalidate()
        typingTimer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: false) { [weak self] _ in self?.flushTyping() }
    }

    private func flushTyping() {
        typingTimer?.invalidate(); typingTimer = nil
        guard let item = pending, item.action["type"] as? String == "type", var value = session else { return }
        do {
            try value.append(frame: item.frame, action: item.action)
            session = value; pending = nil
            // A newer periodic frame is usable only if it was captured after the completed typing.
            if let latest, latest.capturedAt < (item.action["finished_at_ms"] as! Int64) { self.latest = nil }
            status.stringValue = "正在录制；已记录 \(value.count) 个完整动作。"
        } catch { pause(error.localizedDescription) }
    }

    private func review(_ value: OperationRecordingSession) {
        let dialog = NSAlert(); dialog.messageText = "核对示范后允许训练"
        dialog.informativeText = "逐项核对画面与动作，确认没有密码或私人信息，并确认任务确实完成。取消会丢弃全部内存数据。"
        dialog.addButton(withTitle: "确认并导出示范"); dialog.addButton(withTitle: "丢弃")
        let stack = NSStackView(); stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 10
        for step in value.steps {
            let image = NSImage(cgImage: step.frame.cgImage!, size: NSSize(width: step.frame.width, height: step.frame.height))
            let view = NSImageView(image: image); view.imageScaling = .scaleProportionallyUpOrDown
            view.widthAnchor.constraint(equalToConstant: 420).isActive = true
            view.heightAnchor.constraint(equalToConstant: 240).isActive = true
            stack.addArrangedSubview(view)
            let data = try! JSONSerialization.data(withJSONObject: step.action, options: [.sortedKeys])
            let label = NSTextField(wrappingLabelWithString: String(decoding: data, as: UTF8.self))
            label.widthAnchor.constraint(equalToConstant: 420).isActive = true; stack.addArrangedSubview(label)
        }
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 460, height: 420))
        scroll.widthAnchor.constraint(equalToConstant: 460).isActive = true
        scroll.heightAnchor.constraint(equalToConstant: 420).isActive = true
        scroll.hasVerticalScroller = true; scroll.documentView = stack
        stack.frame = NSRect(x: 0, y: 0, width: 440, height: stack.fittingSize.height)
        let wrap = NSStackView(); wrap.orientation = .vertical; wrap.alignment = .leading; wrap.spacing = 8
        wrap.addArrangedSubview(scroll)
        let reviewed = NSButton(checkboxWithTitle: "全部画面和输入已核对，无敏感信息", target: nil, action: nil)
        let success = NSButton(checkboxWithTitle: "我核对了实际结果，任务已成功完成", target: nil, action: nil)
        let permitted = NSButton(checkboxWithTitle: "我有权将这些资料用于本地模型训练", target: nil, action: nil)
        let evidence = NSTextField(string: ""), rights = NSTextField(string: "")
        evidence.placeholderString = "实际完成结果（例如：目标文件已保存并打开核对）"
        rights.placeholderString = "训练许可依据（例如：本人自有示范与资料）"
        [reviewed, success, permitted, evidence, rights].forEach { wrap.addArrangedSubview($0) }
        dialog.accessoryView = wrap
        wrap.frame = NSRect(x: 0, y: 0, width: 460, height: wrap.fittingSize.height)
        guard dialog.runModal() == .alertFirstButtonReturn else { onStatus?("示范已丢弃，没有保存或训练。"); return }
        do {
            let data = try value.export(reviewed: reviewed.state == .on, success: success.state == .on,
                                        permitted: permitted.state == .on, evidence: evidence.stringValue,
                                        rights: rights.stringValue, at: now())
            guard let onExport else { throw OperationRecordingError("示范接收入口尚未连接；没有保存或训练。") }
            onExport(data)
        } catch { alert(error.localizedDescription) }
    }

    private func alert(_ text: String) { onStatus?(text); let value = NSAlert(); value.messageText = text; value.runModal() }
    private func now() -> Int64 { Int64(Date().timeIntervalSince1970 * 1000) }
    deinit { timer?.invalidate(); if let monitor { NSEvent.removeMonitor(monitor) } }
}

struct OperationRecordingError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// Stores only the recorder's reviewed export. Training/sample admission remains a separate step.
enum OperationDemonstrationStore {
    static let maximumBytes = 272 * 1024 * 1024 // Existing 64MiB RGB contract in bounded decimal JSON, plus metadata.

    static func save(_ data: Data, home: URL) throws -> [String: Any] {
        guard !data.isEmpty, data.count <= maximumBytes, home.isFileURL,
              let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              value["schema"] as? String == "luban-operation-demonstration-v1",
              value["source"] as? String == "user_demonstration",
              value["approved"] as? Bool == true, value["training_allowed"] as? Bool == true,
              value["sensitive_reviewed"] as? Bool == true, value["sensitive"] as? Bool == false,
              value["user_data_scope_confirmed"] as? Bool == true,
              let outcome = value["outcome"] as? [String: Any], outcome["success"] as? Bool == true,
              outcome["verified_by"] as? String == "user",
              let steps = value["steps"] as? [[String: Any]], !steps.isEmpty, steps.count <= 256 else {
            throw OperationRecordingError("示范未完成核对、许可或超过保存上限，没有保存。")
        }
        var parent = open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parent >= 0 else { throw OperationRecordingError("无法打开私有示范目录。") }
        defer { close(parent) }
        // Walk only this write destination; never follow an ancestor symlink or inspect unrelated files.
        guard !home.pathComponents.contains(".."), !home.pathComponents.contains(".") else {
            throw OperationRecordingError("私有示范目录不能含相对路径。")
        }
        for component in home.pathComponents where component != "/" {
            let next = openat(parent, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard next >= 0 else { throw OperationRecordingError("私有示范目录不可用或包含链接。") }
            close(parent); parent = next
        }
        var homeStat = stat()
        guard fstat(parent, &homeStat) == 0, homeStat.st_uid == getuid() else {
            throw OperationRecordingError("私有示范目录不属于当前用户。")
        }
        let name = "operation-demonstrations"
        let created = mkdirat(parent, name, 0o700) == 0
        guard created || errno == EEXIST else { throw OperationRecordingError("无法建立私有示范目录。") }
        let directory = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directory >= 0 else { throw OperationRecordingError("私有示范目录不能是链接。") }
        defer { close(directory) }
        var attributes = stat()
        guard fstat(directory, &attributes) == 0, attributes.st_uid == getuid(), attributes.st_mode & 0o777 == 0o700 else {
            throw OperationRecordingError("示范目录需要当前用户独占的0700权限。")
        }
        if created, fsync(parent) != 0 { throw OperationRecordingError("无法保存示范目录。") }
        let id = UUID().uuidString.lowercased(), filename = id + ".json"
        let file = openat(directory, filename, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard file >= 0 else { throw OperationRecordingError("无法创建新的私有示范文件。") }
        var complete = false
        defer { close(file); if !complete { unlinkat(directory, filename, 0) } }
        guard fchmod(file, 0o600) == 0 else { throw OperationRecordingError("无法保护示范文件权限。") }
        try data.withUnsafeBytes { bytes in
            var done = 0
            while done < bytes.count {
                let count = Darwin.write(file, bytes.baseAddress!.advanced(by: done), bytes.count - done)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw OperationRecordingError("示范保存不完整，临时文件已丢弃。") }
                done += count
            }
        }
        guard fsync(file) == 0, fsync(directory) == 0 else { throw OperationRecordingError("示范保存未完成。") }
        complete = true
        return ["state": "saved", "id": id, "actions": steps.count,
                "message": "示范已私有保存；可在学习页训练候选，自动执行尚未接通。"]
    }
}

struct OperationFrame {
    let width: Int, height: Int
    let rgb: [UInt8]
    let capturedAt: Int64
    let bounds: CGRect
    let cgImage: CGImage?

    init(width: Int, height: Int, rgb: [UInt8], capturedAt: Int64, bounds: CGRect) throws {
        guard width > 0, height > 0, width <= 8192, height <= 8192, width * height <= 16_000_000,
              rgb.count == width * height * 3, capturedAt >= 0, bounds.width > 0, bounds.height > 0 else {
            throw OperationRecordingError("画面格式或尺寸不符合录制范围。")
        }
        self.width = width; self.height = height; self.rgb = rgb; self.capturedAt = capturedAt; self.bounds = bounds; self.cgImage = nil
    }

    init(image: CGImage, capturedAt: Int64, bounds: CGRect) throws {
        let width = image.width, height = image.height
        guard width > 0, height > 0, width <= 8192, height <= 8192, width * height <= 16_000_000 else {
            throw OperationRecordingError("窗口过大，请缩小窗口后重新录制。")
        }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height)); return true
        }
        guard ok else { throw OperationRecordingError("无法读取窗口 RGB。") }
        var rgb = [UInt8](); rgb.reserveCapacity(width * height * 3)
        for index in stride(from: 0, to: rgba.count, by: 4) {
            guard rgba[index + 3] == 255 else { throw OperationRecordingError("透明窗口画面暂不支持。") }
            rgb.append(contentsOf: rgba[index ..< index + 3])
        }
        self.width = width; self.height = height; self.rgb = rgb; self.capturedAt = capturedAt
        self.bounds = bounds; self.cgImage = image
    }
}

/// The testable admission/state portion uses the existing demonstration schema, not a new training API.
struct OperationRecordingSession {
    struct Step { let frame: OperationFrame; let action: [String: Any] }
    static let navigationKeys = [36: "Enter", 48: "Tab", 53: "Escape", 51: "Backspace", 126: "ArrowUp", 125: "ArrowDown", 123: "ArrowLeft", 124: "ArrowRight"]
    let id = UUID().uuidString.lowercased()
    let appID: String, goal: String, family: String, startedAt: Int64
    private(set) var stoppedAt: Int64?
    private(set) var steps: [Step] = []
    private var bytes = 0
    var count: Int { steps.count }

    init(appID: String, goal: String, family: String, now: Int64) throws {
        guard Self.validID(appID), Self.validID(family), Self.validText(goal, max: 4096), now >= 0 else {
            throw OperationRecordingError("请填写任务及合法同类任务标识。")
        }
        self.appID = appID; self.goal = goal; self.family = family; self.startedAt = now
    }

    static func safeRole(_ role: String, subrole: String, protected: Bool, typing: Bool) -> Bool {
        guard !protected, !role.localizedCaseInsensitiveContains("secure"), !subrole.localizedCaseInsensitiveContains("secure") else { return false }
        if typing { return ["AXTextField", "AXTextArea"].contains(role) }
        return ["AXWebArea", "AXButton", "AXLink", "AXTable", "AXRow", "AXCell", "AXScrollArea", "AXGroup",
                "AXStaticText", "AXWindow", "AXTextField", "AXTextArea", "AXComboBox", "AXPopUpButton", "AXMenuButton",
                "AXList", "AXOutline", "AXImage", "AXToolbar"].contains(role)
    }

    func validate(frame: OperationFrame, action: [String: Any]) throws {
        guard stoppedAt == nil, count < 256, bytes + frame.rgb.count <= 64 * 1024 * 1024,
              frame.capturedAt >= (steps.last?.action["finished_at_ms"] as? Int64 ?? startedAt),
              let start = action["started_at_ms"] as? Int64, let end = action["finished_at_ms"] as? Int64,
              start > frame.capturedAt, start - frame.capturedAt <= 2000, end >= start,
              let type = action["type"] as? String else { throw OperationRecordingError("画面过期、动作未完成或超出录制预算；已暂停。") }
        var expected = Set(["type", "started_at_ms", "finished_at_ms"])
        switch type {
        case "click", "scroll":
            guard let x = action["x"] as? Int, let y = action["y"] as? Int, x >= 0, y >= 0,
                  x < frame.width, y < frame.height else { throw OperationRecordingError("动作超出所选窗口。") }
            expected.formUnion(["x", "y"])
            if type == "click" {
                expected.formUnion(["button", "count"])
                guard ["left", "right"].contains(action["button"] as? String ?? ""), [1, 2].contains(action["count"] as? Int ?? 0) else { throw OperationRecordingError("不支持的点击。") }
            } else {
                expected.formUnion(["delta_x", "delta_y"])
                guard let dx = action["delta_x"] as? Int, let dy = action["delta_y"] as? Int,
                      (-8192...8192).contains(dx), (-8192...8192).contains(dy), dx != 0 || dy != 0 else { throw OperationRecordingError("不支持的滚动。") }
            }
        case "type":
            expected.formUnion(["text", "sensitive"])
            guard action["sensitive"] as? Bool == false, let text = action["text"] as? String,
                  Self.validText(text, max: 8192) else { throw OperationRecordingError("敏感或无效输入。") }
        case "key":
            expected.insert("key")
            guard Self.navigationKeys.values.contains(action["key"] as? String ?? "") else { throw OperationRecordingError("不支持的按键。") }
        default: throw OperationRecordingError("不支持的动作。")
        }
        guard Set(action.keys) == expected else { throw OperationRecordingError("动作含未知字段。") }
    }

    mutating func append(frame: OperationFrame, action: [String: Any]) throws {
        try validate(frame: frame, action: action)
        steps.append(Step(frame: frame, action: action)); bytes += frame.rgb.count
    }

    mutating func stop(at: Int64) throws {
        guard stoppedAt == nil, !steps.isEmpty, at > startedAt,
              at >= (steps.last!.action["finished_at_ms"] as! Int64) else { throw OperationRecordingError("没有完整示范，或停止时间无效。") }
        stoppedAt = at
    }

    func export(reviewed: Bool, success: Bool, permitted: Bool, evidence: String, rights: String, at: Int64) throws -> Data {
        guard reviewed, success, permitted, let stoppedAt, at >= stoppedAt,
              Self.validText(evidence, max: 4096), Self.validText(rights, max: 4096) else {
            throw OperationRecordingError("必须核对全部画面、实际成功结果及训练许可；未导出。")
        }
        let result: [String: Any] = ["schema": "luban-operation-demonstration-v1", "id": "demo-\(id)",
            "session_id": id, "source": "user_demonstration", "source_family": family, "approved": true,
            "training_allowed": true, "rights_basis": rights, "user_data_scope_confirmed": true,
            "sensitive_reviewed": true, "sensitive": false,
            "recording": ["start": ["explicit": true, "at_ms": startedAt], "stop": ["explicit": true, "at_ms": stoppedAt]],
            "task": ["id": "task-\(id)", "family": family, "goal": goal, "app_ids": [appID], "allowed_actions": ["click", "type", "key", "scroll"]],
            "outcome": ["task_id": "task-\(id)", "verified_by": "user", "success": true, "at_ms": at, "evidence": evidence],
            "steps": steps.enumerated().map { index, step -> [String: Any] in
                var action = step.action
                action["id"] = "action-\(index)"; action["task_id"] = "task-\(id)"; action["app_id"] = appID
                let observation: [String: Any] = ["id": "frame-\(index)", "task_id": "task-\(id)", "app_id": appID,
                    "session_id": id, "captured_at_ms": step.frame.capturedAt, "phase": "before_action",
                    "previous_action_id": index == 0 ? NSNull() : "action-\(index - 1)",
                    "width": step.frame.width, "height": step.frame.height, "rgb": step.frame.rgb,
                    "reviewed": true, "sensitive": false]
                return ["observation": observation, "action": action]
            }]
        return try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    }

    private static func validID(_ text: String) -> Bool {
        !text.isEmpty && text.count <= 160 && text.range(of: "^[A-Za-z0-9._:-]+$", options: .regularExpression) != nil
    }
    private static func validText(_ text: String, max: Int) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && text.utf16.count <= max
        && !text.unicodeScalars.contains { $0.value < 32 && ![9, 10, 13].contains($0.value) || $0.value == 127 }
    }
}
