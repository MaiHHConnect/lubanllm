import AppKit
import WebKit
import UniformTypeIdentifiers

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler, WKDownloadDelegate {
    enum CloseAction: String {
        case close
        case minimize
    }

    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var terminationSignal: DispatchSourceSignal?
    var attempt = 0
    var started = false
    var sessionToken: String?
    var preparingUpdate = false
    var checkingClose = false
    var bypassClosePrompt = false
    var choosingFeeding = false
    var operationRecorder: AnyObject?
    lazy var updater: DesktopUpdater = {
        let value = DesktopUpdater(home: home, version: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0",
                                   build: Int(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0") ?? 0)
        value.publish = { [weak self] status in self?.publishUpdate(status) }
        return value
    }()
    let port = 8765
    let home = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Luban", isDirectory: true)

    func applicationDidFinishLaunching(_ notification: Notification) {
        signal(SIGTERM, SIG_IGN)
        let signalSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        // NSApp.terminate enters a nested run loop; leave the main dispatch callback first
        // so the asynchronous close-status/pause replies can return on the main queue.
        signalSource.setEventHandler { RunLoop.main.perform { self.terminateWithoutClosePrompt() } }
        signalSource.resume()
        terminationSignal = signalSource
        let menu = NSMenu()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "退出鲁班零号", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let resetClose = appMenu.addItem(withTitle: "恢复关闭提示", action: #selector(resetCloseActionPreference), keyEquivalent: "")
        resetClose.target = self
        let menuItem = NSMenuItem()
        menuItem.submenu = appMenu
        menu.addItem(menuItem)
        let edit = NSMenu(title: "编辑")
        for (title, action, key) in [("撤销", "undo:", "z"), ("剪切", "cut:", "x"),
                                     ("复制", "copy:", "c"), ("粘贴", "paste:", "v"),
                                     ("全选", "selectAll:", "a")] {
            let item = edit.addItem(withTitle: title, action: NSSelectorFromString(key == "v" ? "pasteChatAttachment:" : action), keyEquivalent: key)
            if key == "v" { item.target = self }
        }
        let editItem = NSMenuItem(title: "编辑", action: nil, keyEquivalent: "")
        editItem.submenu = edit
        menu.addItem(editItem)
        NSApp.mainMenu = menu
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1160, height: 660),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "鲁班零号 · 本地智能"
        window.delegate = self
        window.minSize = NSSize(width: 760, height: 600)
        window.center()
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(self, name: "lubanUpdate")
        configuration.userContentController.add(self, name: "lubanFeeding")
        configuration.userContentController.add(self, name: "lubanShare")
        web = WKWebView(frame: window.contentView!.bounds, configuration: configuration)
        web.autoresizingMask = [.width, .height]
        web.navigationDelegate = self
        web.uiDelegate = self
        window.contentView!.addSubview(web)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        web.loadHTMLString("<html lang='zh-CN'><body style='background:#fff;color:#152033;font-family:system-ui;text-align:center;padding:160px 32px'><h1 style='color:#c7192b'>鲁班零号</h1><p>正在打开应用…</p></body></html>", baseURL: nil)
        probe()
    }

    func startServer() throws {
        let project = Bundle.main.resourceURL!
        let service = Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("luban-service")
        guard FileManager.default.isExecutableFile(atPath: service.path) else {
            throw NSError(domain: "Luban", code: 1, userInfo: [NSLocalizedDescriptionKey: "应用运行文件缺失，请重新安装鲁班零号。"])
        }
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let logURL = home.appendingPathComponent("app.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
        let log = try FileHandle(forWritingTo: logURL)
        let task = Process()
        task.executableURL = service
        task.currentDirectoryURL = home
        let packagedBackend = Bundle.main.resourceURL!.appendingPathComponent("backend/src", isDirectory: true)
        let backend = packagedBackend
        task.arguments = ["--project", project.path, "--backend", backend.path,
                          "--home", home.path, "--port", String(port)]
        var env = ProcessInfo.processInfo.environment
        env["LUBAN_DESKTOP_VERSION"] = updater.currentVersion
        env["LUBAN_DESKTOP_BUILD"] = String(updater.currentBuild)
        env["TOKENIZERS_PARALLELISM"] = "false"
        task.environment = env
        task.standardOutput = log
        task.standardError = log
        try task.run()
        server = task
    }

    func probe() {
        let url = URL(string: "http://127.0.0.1:\(port)/api/health")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let data = data, let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   value["app"] as? String == "luban" {
                    do {
                        let launch = try JSONSerialization.jsonObject(with: Data(contentsOf: self.home.appendingPathComponent("launch.json"))) as! [String: Any]
                        let token = launch["token"] as! String
                        self.sessionToken = token
                        var components = URLComponents(string: "http://127.0.0.1:\(self.port)/")!
                        components.fragment = "token=" + token
                        self.web.load(URLRequest(url: components.url!))
                        self.updater.start()
                    } catch { self.fail("无法读取本地启动会话，请重新启动应用。") }
                    return
                }
                if let response = response as? HTTPURLResponse {
                    self.fail("本地端口 \(self.port) 已被其他服务占用（HTTP \(response.statusCode)）。")
                    return
                }
                if !self.started {
                    self.started = true
                    do { try self.startServer() } catch { self.fail(error.localizedDescription); return }
                }
                self.attempt += 1
                if self.attempt > 600 || (self.server != nil && !self.server!.isRunning) {
                    self.fail("服务启动失败。请查看 ~/Library/Application Support/Luban/app.log。")
                    return
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.probe() }
            }
        }.resume()
    }

    func fail(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "鲁班启动失败"
        alert.informativeText = message
        alert.addButton(withTitle: "重试")
        alert.addButton(withTitle: "退出")
        if alert.runModal() == .alertFirstButtonReturn {
            attempt = 0
            if server != nil && !server!.isRunning { started = false }
            probe()
        } else { self.terminateWithoutClosePrompt() }
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url else { decisionHandler(.cancel); return }
        if isLocalPage(url) || url.scheme == "about" {
            decisionHandler(navigationAction.shouldPerformDownload && isLocalPage(url) ? .download : .allow)
        } else {
            let feedback = url.absoluteString == "mailto:lubanlinghao@agent.qq.com"
                && navigationAction.navigationType == .linkActivated && navigationAction.sourceFrame.isMainFrame
                && webView.url.map(isLocalPage) == true
            if ["https", "http"].contains(url.scheme ?? "") || feedback { NSWorkspace.shared.open(url) }
            decisionHandler(.cancel)
        }
    }

    @objc func pasteChatAttachment(_ sender: Any?) {
        guard let page = web?.url, isLocalPage(page) else {
            NSApp.sendAction(NSSelectorFromString("paste:"), to: nil, from: sender)
            return
        }
        web.evaluateJavaScript("document.activeElement?.id === 'chat-input'") { value, _ in
            guard value as? Bool == true else {
                NSApp.sendAction(NSSelectorFromString("paste:"), to: nil, from: sender)
                return
            }
            let clipboard = NSPasteboard.general
            let urls = clipboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
            do {
                var files: [[String: String]] = []
                var total = 0
                func appendFile(_ url: URL, name: String) throws {
                    let type = UTType(filenameExtension: url.pathExtension)
                    guard type?.conforms(to: .image) != true, type?.conforms(to: .audio) != true,
                          name.lowercased().range(of: "\\.(png|jpe?g|gif|webp|bmp|heic|avif|wav|mp3|m4a|aac|ogg|flac|aiff?|opus)$", options: .regularExpression) == nil else {
                        throw NSError(domain: "Luban", code: 4, userInfo: [NSLocalizedDescriptionKey: "当前对话仅支持文字，不接收图片或音频；请选择文本文件或文档。"])
                    }
                    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
                    guard values.isRegularFile == true, values.isSymbolicLink != true,
                          let size = values.fileSize, size > 0, size <= 8 * 1024 * 1024,
                          files.count < 32, total + size <= 16 * 1024 * 1024 else {
                        throw NSError(domain: "Luban", code: 1, userInfo: [NSLocalizedDescriptionKey: "剪贴板文件最多32个，共16 MiB，每个不超过8 MiB；不读取符号链接。"])
                    }
                    let data = try Data(contentsOf: url)
                    guard data.count == size else { throw NSError(domain: "Luban", code: 2, userInfo: [NSLocalizedDescriptionKey: "文件读取期间发生变化，请重新复制。"] ) }
                    total += data.count
                    files.append(["name": name, "type": UTType(filenameExtension: url.pathExtension)?.preferredMIMEType ?? "application/octet-stream", "base64": data.base64EncodedString()])
                }
                for url in urls {
                    let access = url.startAccessingSecurityScopedResource()
                    defer { if access { url.stopAccessingSecurityScopedResource() } }
                    let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                    guard values.isSymbolicLink != true else { throw NSError(domain: "Luban", code: 3, userInfo: [NSLocalizedDescriptionKey: "请复制实际文件或文件夹，不使用符号链接。"] ) }
                    if values.isDirectory == true {
                        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { continue }
                        for case let child as URL in enumerator {
                            let info = try child.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                            if info.isSymbolicLink == true { enumerator.skipDescendants(); continue }
                            if info.isRegularFile == true { try appendFile(child, name: url.lastPathComponent + "/" + String(child.path.dropFirst(url.path.count + 1))) }
                        }
                    } else { try appendFile(url, name: url.lastPathComponent) }
                }
                if urls.isEmpty, clipboard.string(forType: .string) == nil,
                   clipboard.types?.contains(where: { type in
                       let value = UTType(type.rawValue)
                       return value?.conforms(to: .image) == true || value?.conforms(to: .audio) == true
                   }) == true {
                    throw NSError(domain: "Luban", code: 4, userInfo: [NSLocalizedDescriptionKey: "当前对话仅支持文字，不接收图片或音频；请选择文本文件或文档。"])
                }
                guard !files.isEmpty else {
                    NSApp.sendAction(NSSelectorFromString("paste:"), to: nil, from: sender)
                    return
                }
                let json = String(data: try JSONSerialization.data(withJSONObject: files), encoding: .utf8)!
                self.web.evaluateJavaScript("window.dispatchEvent(new CustomEvent('luban-chat-files',{detail:" + json + "}));", completionHandler: nil)
            } catch {
                let json = String(data: (try? JSONSerialization.data(withJSONObject: ["message": error.localizedDescription])) ?? Data("{}".utf8), encoding: .utf8)!
                self.web.evaluateJavaScript("window.dispatchEvent(new CustomEvent('luban-chat-paste-error',{detail:" + json + "}));", completionHandler: nil)
            }
        }
    }

    func chooseFeedingFiles() {
        guard !choosingFeeding else { return }
        choosingFeeding = true
        let panel = NSOpenPanel()
        panel.title = "选择投喂资料"
        panel.prompt = "选择"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.beginSheetModal(for: window) { response in
            guard response == .OK else { self.choosingFeeding = false; return }
            let urls = panel.urls
            DispatchQueue.global(qos: .userInitiated).async {
                let manager = FileManager.default
                let keys: Set<URLResourceKey> = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
                let privateNames: Set<String> = ["internal", "node_modules", "credentials", "credentials.json", "settings", "settings.json", "memory.sqlite3", "模型流程脑图.html"]
                var files = [[String: String]](), rejected = [[String: Any]](), total = 0, visited = 0
                func reject(_ name: String, _ reason: String) {
                    if rejected.count < 100 { rejected.append(["name": name, "supported": false, "reason": reason]) }
                }
                func inspect(_ url: URL, name: String) -> Bool {
                    visited += 1
                    if visited > 1000 { return false }
                    let parts = name.precomposedStringWithCompatibilityMapping.lowercased().split(separator: "/")
                    guard !parts.contains(where: { $0.hasPrefix(".") || privateNames.contains(String($0)) || $0.range(of: "^(settings|credentials)([._-]|$)", options: .regularExpression) != nil }),
                          name.count <= 300, name.rangeOfCharacter(from: .controlCharacters) == nil,
                          !name.contains(":"), !name.contains("\\") else { reject(name, "私有或内部文件不能投喂"); return false }
                    guard let values = try? url.resourceValues(forKeys: keys), values.isSymbolicLink != true else { reject(name, "不能读取链接或此文件"); return false }
                    if values.isDirectory == true { return true }
                    guard values.isRegularFile == true else { reject(name, "不是普通文件"); return false }
                    let kind = url.pathExtension.lowercased()
                    let document = ["pdf", "doc", "docx", "rtf", "odt", "html", "htm"].contains(kind)
                    guard document || ["txt", "md", "csv", "json", "jsonl", "tsv", "log", "rst"].contains(kind) else { reject(name, "不支持此格式"); return false }
                    let limit = document ? 8 * 1024 * 1024 : 65536
                    guard (values.fileSize ?? limit + 1) <= limit, files.count < 20, total + (values.fileSize ?? limit + 1) <= 16 * 1024 * 1024 else { reject(name, "超过文件数量或大小限制"); return false }
                    do {
                        let handle = try FileHandle(forReadingFrom: url)
                        defer { try? handle.close() }
                        let data = try handle.read(upToCount: limit + 1) ?? Data()
                        guard data.count <= limit, total + data.count <= 16 * 1024 * 1024 else { reject(name, "文件过大"); return false }
                        if document { files.append(["name": name, "base64": data.base64EncodedString()]) }
                        else if let content = String(data: data, encoding: .utf8) { files.append(["name": name, "content": content]) }
                        else { reject(name, "无法读取有效 UTF-8 文本"); return false }
                        total += data.count
                    } catch { reject(name, "无法读取文件") }
                    return false
                }
                for root in urls {
                    if visited >= 1000 { break }
                    let name = root.lastPathComponent
                    guard inspect(root, name: name), let iterator = manager.enumerator(at: root, includingPropertiesForKeys: Array(keys), options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { continue }
                    while let url = iterator.nextObject() as? URL {
                        if visited >= 1000 { reject(name, "文件夹过大，仅检查前1000项"); break }
                        let relative = name + "/" + String(url.path.dropFirst(root.path.count + 1))
                        if !inspect(url, name: relative) { iterator.skipDescendants() }
                    }
                }
                let result: [String: Any] = ["files": files, "rejected": rejected]
                DispatchQueue.main.async {
                    self.choosingFeeding = false
                    guard let page = self.web.url, self.isLocalPage(page),
                          let data = try? JSONSerialization.data(withJSONObject: result), let json = String(data: data, encoding: .utf8) else { return }
                    self.web.evaluateJavaScript("window.dispatchEvent(new CustomEvent('luban-feeding-files',{detail:" + json + "}));", completionHandler: nil)
                }
            }
        }
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        guard let url = webView.url, isLocalPage(url) else { completionHandler(nil); return }
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.canChooseFiles = !parameters.allowsDirectories
        panel.beginSheetModal(for: window) { response in completionHandler(response == .OK ? panel.urls : nil) }
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        guard let url = navigationResponse.response.url, isLocalPage(url) else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(navigationResponse.response.mimeType == "application/zip" ? .download : .allow)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse,
                  suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let templates = ["/assets/templates/source.txt", "/assets/templates/qa.csv", "/assets/templates/qa.jsonl"]
        guard let url = response.url, isLocalPage(url),
              url.path.hasPrefix("/api/model-packages/download/") || templates.contains(url.path) else {
            completionHandler(nil)
            return
        }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedFilename
        panel.allowedContentTypes = url.path.hasPrefix("/api/model-packages/download/") ? [.zip] : [UTType(filenameExtension: url.pathExtension) ?? .plainText]
        panel.canCreateDirectories = true
        panel.beginSheetModal(for: window) { result in completionHandler(result == .OK ? panel.url : nil) }
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        let alert = NSAlert()
        alert.messageText = "文件保存失败"
        alert.informativeText = "请重新下载，或选择有足够空间的保存位置。"
        alert.addButton(withTitle: "知道了")
        alert.beginSheetModal(for: window)
    }

    func isLocalPage(_ url: URL) -> Bool {
        url.scheme == "http" && url.host == "127.0.0.1" && url.port == port && url.user == nil && url.password == nil
    }

    func publishUpdate(_ status: [String: Any]) {
        guard let url = web?.url, isLocalPage(url),
              let data = try? JSONSerialization.data(withJSONObject: status),
              let json = String(data: data, encoding: .utf8) else { return }
        web.evaluateJavaScript("window.dispatchEvent(new CustomEvent('luban-update',{detail:" + json + "}));", completionHandler: nil)
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        let origin = message.frameInfo.securityOrigin
        guard ["lubanUpdate", "lubanFeeding", "lubanShare"].contains(message.name), message.frameInfo.isMainFrame,
              origin.protocol == "http", origin.host == "127.0.0.1", origin.port == port,
              let page = web.url, isLocalPage(page), let token = sessionToken,
              let body = message.body as? [String: Any], let action = body["action"] as? String,
              (message.name == "lubanUpdate" ? ["status", "check", "install"].contains(action)
                : message.name == "lubanShare" ? action == "copy-image" : ["choose", "record-operation"].contains(action)) else { return }
        web.configuration.websiteDataStore.httpCookieStore.getAllCookies { cookies in
            DispatchQueue.main.async {
                guard let current = self.web.url, self.isLocalPage(current),
                      self.sessionToken == token,
                      cookies.contains(where: { $0.name == "luban_session" && $0.value == token && $0.domain == "127.0.0.1" }) else { return }
                if message.name == "lubanShare" { self.copyEvolutionImage(body); return }
                if message.name == "lubanFeeding" {
                    if action == "record-operation" { self.recordOperation(token: token) }
                    else { self.chooseFeedingFiles() }
                    return
                }
                switch action {
                case "check": self.updater.check()
                case "install": self.installUpdate(token: token)
                default: self.updater.report()
                }
            }
        }
    }

    func copyEvolutionImage(_ body: [String: Any]) {
        guard let id = body["id"] as? String, UUID(uuidString: id) != nil else { return }
        var copied = false
        if let encoded = body["png"] as? String, encoded.utf8.count <= 4 * 1024 * 1024,
           let data = Data(base64Encoded: encoded), !data.isEmpty, data.count <= 3 * 1024 * 1024,
           data.starts(with: [137, 80, 78, 71, 13, 10, 26, 10]), data.count >= 24 {
            // Read PNG dimensions before decoding so compressed input cannot expand without a bound.
            let width = data[16..<20].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            let height = data[20..<24].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            if width > 0, height > 0, width <= 2048, height <= 2048,
               let image = NSBitmapImageRep(data: data), image.pixelsWide == Int(width), image.pixelsHigh == Int(height) {
                let item = NSPasteboardItem()
                if item.setData(data, forType: .png) {
                    NSPasteboard.general.clearContents()
                    copied = NSPasteboard.general.writeObjects([item])
                }
            }
        }
        guard let data = try? JSONSerialization.data(withJSONObject: ["id": id, "success": copied]),
              let json = String(data: data, encoding: .utf8) else { return }
        web.evaluateJavaScript("window.dispatchEvent(new CustomEvent('luban-share-copied',{detail:" + json + "}));", completionHandler: nil)
    }

    func publishOperation(_ value: [String: Any]) {
        guard let page = web?.url, isLocalPage(page),
              let data = try? JSONSerialization.data(withJSONObject: value),
              let json = String(data: data, encoding: .utf8) else { return }
        web.evaluateJavaScript("window.dispatchEvent(new CustomEvent('luban-operation-recording',{detail:" + json + "}));", completionHandler: nil)
    }

    func recordOperation(token: String) {
        guard #available(macOS 14.0, *) else {
            publishOperation(["message": "操作示范录制需要 macOS 14 或更新版本；尚未开始录制。"]); return
        }
        let recorder = (operationRecorder as? OperationRecorder) ?? OperationRecorder()
        operationRecorder = recorder
        recorder.onStatus = { [weak self] message in self?.publishOperation(["message": message]) }
        recorder.onExport = { [weak self] data in
            guard let self else { return }
            guard self.sessionToken == token, let page = self.web.url, self.isLocalPage(page) else {
                self.publishOperation(["message": "本机会话已变更；示范没有保存。"]); return
            }
            self.web.configuration.websiteDataStore.httpCookieStore.getAllCookies { [weak self] cookies in
                DispatchQueue.main.async {
                    guard let self else { return }
                    guard self.sessionToken == token, let current = self.web.url, self.isLocalPage(current),
                          cookies.contains(where: { $0.name == "luban_session" && $0.value == token && $0.domain == "127.0.0.1" }) else {
                        self.publishOperation(["message": "本机会话已变更；示范没有保存。"]); return
                    }
                    do { self.publishOperation(try OperationDemonstrationStore.save(data, home: self.home)) }
                    catch { self.publishOperation(["message": error.localizedDescription]) }
                }
            }
        }
        recorder.present()
    }

    func closeOperationRecording() {
        if #available(macOS 14.0, *) { (operationRecorder as? OperationRecorder)?.close() }
        operationRecorder = nil
    }

    func updateAdmission(_ endpoint: String, token: String, completion: @escaping (String?) -> Void) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/update/" + endpoint)!)
        request.httpMethod = "POST"
        request.timeoutInterval = 5
        request.setValue("luban_session=" + token, forHTTPHeaderField: "Cookie")
        request.setValue("http://127.0.0.1:\(port)", forHTTPHeaderField: "Origin")
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                guard error == nil, (response as? HTTPURLResponse)?.statusCode == 200,
                      let data = data, let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      value["ok"] as? Bool == true else {
                    var message = "暂时无法更新，请等待模型加载、对话或学习任务完成后重试。"
                    if let data = data, let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let detail = value["detail"] as? String { message = detail }
                    completion(message); return
                }
                completion(nil)
            }
        }.resume()
    }

    func installUpdate(token: String) {
        guard !preparingUpdate, updater.state == "ready" else { updater.report(); return }
        guard let ownedServer = server, ownedServer.isRunning else {
            updater.installationError("当前服务由另一个进程启动。请退出所有鲁班窗口后重新启动本应用再更新。")
            return
        }
        preparingUpdate = true
        updateAdmission("prepare", token: token) { error in
            self.preparingUpdate = false
            if let error = error { self.updater.installationError(error); return }
            do {
                guard ownedServer.isRunning else { throw UpdateFailure("本地服务已停止，请重启应用后更新。") }
                try self.updater.install(appPID: ProcessInfo.processInfo.processIdentifier, serverPID: ownedServer.processIdentifier)
                self.terminateWithoutClosePrompt()
            } catch {
                self.updateAdmission("cancel", token: token) { _ in }
                self.updater.installationError(error.localizedDescription)
            }
        }
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        switch closeAction(for: sender) {
        case .minimize:
            sender.miniaturize(nil)
        case .close:
            bypassClosePrompt = true
            NSApp.terminate(nil)
        }
        return false
    }

    private let closeActionPreferenceKey = "closeActionPreference"

    func closeAction(for target: NSWindow?) -> CloseAction {
        if let raw = UserDefaults.standard.string(forKey: closeActionPreferenceKey),
           let saved = CloseAction(rawValue: raw) { return saved }

        let alert = NSAlert()
        alert.messageText = "关闭鲁班零号"
        alert.informativeText = "请选择关闭应用，或缩小窗口继续在后台运行。"
        alert.addButton(withTitle: "直接关闭")
        alert.addButton(withTitle: "缩小")
        alert.showsSuppressionButton = true
        alert.suppressionButton?.title = "记住我的选择"
        let action: CloseAction = alert.runModal() == .alertFirstButtonReturn ? .close : .minimize
        if alert.suppressionButton?.state == .on {
            UserDefaults.standard.set(action.rawValue, forKey: closeActionPreferenceKey)
        } else {
            UserDefaults.standard.removeObject(forKey: closeActionPreferenceKey)
        }
        return action
    }

    @objc func resetCloseActionPreference(_ sender: Any?) {
        UserDefaults.standard.removeObject(forKey: closeActionPreferenceKey)
    }

    func terminateWithoutClosePrompt() {
        bypassClosePrompt = true
        NSApp.terminate(nil)
    }

    func closeRequest(_ endpoint: String, method: String = "GET", completion: @escaping ([String: Any]?) -> Void) {
        guard let token = sessionToken else { completion(nil); return }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/automatic/" + endpoint)!)
        request.httpMethod = method
        request.timeoutInterval = method == "POST" ? 60 : 5
        request.setValue("luban_session=" + token, forHTTPHeaderField: "Cookie")
        request.setValue("http://127.0.0.1:\(port)", forHTTPHeaderField: "Origin")
        URLSession.shared.dataTask(with: request) { data, response, _ in
            let value = (response as? HTTPURLResponse)?.statusCode == 200
                ? data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } : nil
            DispatchQueue.main.async { completion(value) }
        }.resume()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        closeOperationRecording()
        if bypassClosePrompt {
            bypassClosePrompt = false
        } else if closeAction(for: window) == .minimize {
            window?.miniaturize(nil)
            return .terminateCancel
        }
        guard sessionToken != nil, server?.isRunning == true else { return .terminateNow }
        if checkingClose { return .terminateLater }
        checkingClose = true
        closeRequest("close-status") { status in
            if status?["package_busy"] as? Bool == true {
                let alert = NSAlert()
                alert.messageText = "正在处理模型包"
                alert.informativeText = "请等待导入、导出或切换完成后再退出。"
                alert.addButton(withTitle: "留在应用")
                alert.runModal()
                self.checkingClose = false
                NSApp.reply(toApplicationShouldTerminate: false)
                return
            }
            if status?["running"] as? Bool == true {
                let alert = NSAlert()
                alert.messageText = "正在学习"
                alert.informativeText = "保存有效训练进度后退出，下次打开会继续学习。"
                alert.addButton(withTitle: "保存后退出")
                alert.addButton(withTitle: "继续学习")
                if alert.runModal() != .alertFirstButtonReturn {
                    self.checkingClose = false
                    NSApp.reply(toApplicationShouldTerminate: false)
                    return
                }
            }
            self.closeRequest("pause", method: "POST") { result in
                if result?["ok"] as? Bool == true {
                    NSApp.reply(toApplicationShouldTerminate: true)
                } else {
                    let alert = NSAlert()
                    alert.messageText = "训练进度还在保存"
                    alert.informativeText = "请稍后再退出；强制退出后，只能从最近的有效检查点继续。"
                    alert.addButton(withTitle: "留在应用")
                    alert.runModal()
                    self.closeRequest("resume", method: "POST") { _ in }
                    self.checkingClose = false
                    NSApp.reply(toApplicationShouldTerminate: false)
                }
            }
        }
        return .terminateLater
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) {
        closeOperationRecording()
        if let process = server, process.isRunning { process.terminate() }
    }
}

@main
struct LubanApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }
}
