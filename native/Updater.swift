import Foundation

struct UpdateDelta: Decodable {
    let from_version: String
    let from_build: Int
    let url: String
    let sha256: String
    let size: Int64
    let removed: [String]

    func validate(version: String, build: Int, architecture: String) throws {
        guard let previous = UpdateManifest.parts(from_version), let next = UpdateManifest.parts(version),
              previous.lexicographicallyPrecedes(next), from_build > 0, from_build < build,
              let location = URL(string: url), UpdateManifest.trustedURL(location),
              location.path == "/updates/Luban-\(version)-from-\(from_version)-macos-\(architecture)-delta.zip",
              sha256.count == 64, sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
              size > 0, size <= 512 * 1024 * 1024, removed.count <= 30_000 else {
            throw UpdateFailure("模块更新清单无效。")
        }
        var paths: Set<String> = []
        for path in removed {
            let key = try NativeUpdateArchive.deltaPath(path)
            guard paths.insert(key).inserted else { throw UpdateFailure("模块更新包含重复删除路径。") }
        }
    }
}

struct UpdateManifest: Decodable {
    let version: String
    let build: Int
    let url: String
    let sha256: String
    let size: Int64
    let notes: [String]
    let platform: String
    let architecture: String
    let delta: UpdateDelta?

    static func parts(_ version: String) -> [Int]? {
        let fields = version.split(separator: ".", omittingEmptySubsequences: false)
        guard fields.count == 3 else { return nil }
        var result: [Int] = []
        for field in fields {
            guard !field.isEmpty, field.allSatisfy({ $0.isASCII && $0.isNumber }),
                  field == "0" || !field.hasPrefix("0"), let number = Int(field), number < 1_000_000 else { return nil }
            result.append(number)
        }
        return result
    }

    static func trustedURL(_ value: URL) -> Bool {
        value.scheme == "https" && value.host == "ai.maihh.net" && (value.port == nil || value.port == 443)
            && value.user == nil && value.password == nil && value.query == nil && value.fragment == nil
            && value.path.hasPrefix("/updates/") && !value.path.contains("..")
    }

    func validate(currentVersion: String, currentBuild: Int) throws -> Bool {
        #if arch(arm64)
        let nativeArchitecture = "arm64"
        #else
        let nativeArchitecture = "x64"
        #endif
        guard platform == "macos", architecture == nativeArchitecture, let next = Self.parts(version), let current = Self.parts(currentVersion), build > 0,
              let location = URL(string: url), Self.trustedURL(location), location.path.hasSuffix(".zip"),
              sha256.count == 64, sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
              size > 0, size <= 512 * 1024 * 1024, (1...5).contains(notes.count),
              notes.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                  && !$0.unicodeScalars.contains(where: { $0.value < 32 || (127...159).contains($0.value) }) }),
              notes.reduce(0, { $0 + $1.unicodeScalars.count }) <= 300 else {
            throw UpdateFailure("更新清单无效：版本、地址、校验值或更新说明不符合要求。")
        }
        try delta?.validate(version: version, build: build, architecture: architecture)
        return build > currentBuild && current.lexicographicallyPrecedes(next)
    }
}

struct UpdateFailure: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

final class DesktopUpdater: NSObject, URLSessionDownloadDelegate {
    let currentVersion: String
    let currentBuild: Int
    let home: URL
    var publish: (([String: Any]) -> Void)?
    private(set) var state = "idle"
    private var errorMessage: String?
    private var candidate: UpdateManifest?
    private var preparedApp: URL?
    private var deltaSelected = false
    private var timer: Timer?
    private var jobs: [Int: (manifest: UpdateManifest?, limit: Int64)] = [:]
    private let worker = DispatchQueue(label: "local.luban.update-verification", qos: .utility)
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 1800
        config.httpCookieStorage = nil
        config.urlCache = nil
        return URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }()

    init(home: URL, version: String, build: Int) {
        self.home = home.appendingPathComponent("updates", isDirectory: true)
        currentVersion = version
        currentBuild = build
        super.init()
    }

    var status: [String: Any] {
        var value: [String: Any] = ["current_version": currentVersion, "current_build": currentBuild, "state": state]
        if let manifest = candidate {
            value["version"] = manifest.version
            value["build"] = manifest.build
            value["notes"] = manifest.notes
            value["download_kind"] = deltaSelected ? "delta" : "full"
            value["download_size"] = deltaSelected ? manifest.delta!.size : manifest.size
        }
        if let errorMessage = errorMessage { value["error"] = errorMessage }
        return value
    }

    func report() { publish?(status) }
    private func setState(_ value: String, error: String? = nil) {
        state = value
        errorMessage = error
        report()
    }
    func start() {
        guard timer == nil else { return }
        check()
        timer = Timer.scheduledTimer(withTimeInterval: 3600, repeats: true) { [weak self] _ in self?.check() }
    }
    func check() {
        guard !["checking", "downloading", "installing", "ready"].contains(state) else { report(); return }
        candidate = nil
        preparedApp = nil
        deltaSelected = false
        setState("checking")
        download(URL(string: "https://ai.maihh.net/updates/latest.json")!, manifest: nil, limit: 64 * 1024)
    }
    private func download(_ url: URL, manifest: UpdateManifest?, limit: Int64) {
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let task = session.downloadTask(with: request)
        jobs[task.taskIdentifier] = (manifest, limit)
        task.resume()
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(request.url.map(UpdateManifest.trustedURL) == true ? request : nil)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if let job = jobs[downloadTask.taskIdentifier],
           totalBytesWritten > job.limit || totalBytesExpectedToWrite > job.limit { downloadTask.cancel() }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard jobs.removeValue(forKey: task.taskIdentifier) != nil, let error = error else { return }
        setState("error", error: "更新检查或下载失败：\(error.localizedDescription)")
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let job = jobs.removeValue(forKey: downloadTask.taskIdentifier) else { return }
        do {
            guard let response = downloadTask.response as? HTTPURLResponse, response.statusCode == 200,
                  let responseURL = response.url, UpdateManifest.trustedURL(responseURL) else {
                throw UpdateFailure("更新服务器未返回有效文件。")
            }
            let size = try location.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size > 0 && Int64(size) <= job.limit else { throw UpdateFailure("更新文件大小不符合要求。") }
            if job.manifest == nil {
                let manifest = try JSONDecoder().decode(UpdateManifest.self, from: Data(contentsOf: location))
                guard try manifest.validate(currentVersion: currentVersion, currentBuild: currentBuild) else {
                    setState("idle"); return
                }
                candidate = manifest
                deltaSelected = manifest.delta.map { $0.from_version == currentVersion && $0.from_build == currentBuild } ?? false
                setState("downloading")
                download(URL(string: deltaSelected ? manifest.delta!.url : manifest.url)!, manifest: manifest,
                         limit: deltaSelected ? manifest.delta!.size : manifest.size)
                return
            }
            let manifest = job.manifest!
            let useDelta = deltaSelected
            guard Int64(size) == (useDelta ? manifest.delta!.size : manifest.size) else { throw UpdateFailure("更新文件不完整。") }
            let fm = FileManager.default
            try fm.createDirectory(at: home, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let directory = home.appendingPathComponent(UUID().uuidString, isDirectory: true)
            try fm.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
            let archive = directory.appendingPathComponent("update.zip")
            try fm.moveItem(at: location, to: archive)
            try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: archive.path)
            worker.async {
                do {
                    let app = try self.verifyAndExtract(archive, directory: directory, manifest: manifest, useDelta: useDelta)
                    DispatchQueue.main.async { self.preparedApp = app; self.setState("ready") }
                } catch {
                    try? fm.removeItem(at: directory)
                    DispatchQueue.main.async { self.setState("error", error: error.localizedDescription) }
                }
            }
        } catch { setState("error", error: error.localizedDescription) }
    }

    private func verifyAndExtract(_ archive: URL, directory: URL, manifest: UpdateManifest, useDelta: Bool) throws -> URL {
        let payload = directory.appendingPathComponent("payload", isDirectory: true)
        do {
            if useDelta, let delta = manifest.delta {
                let installed = Bundle.main.bundleURL.standardizedFileURL
                try validateApp(installed, version: delta.from_version, build: delta.from_build)
                _ = try NativeUpdateArchive.reconstruct(archive, base: installed, to: payload,
                    expectedSHA256: delta.sha256, removed: delta.removed)
            } else {
                _ = try NativeUpdateArchive.extract(archive, to: payload, expectedSHA256: manifest.sha256)
            }
            let app = payload.appendingPathComponent("Luban.app", isDirectory: true)
            try validateApp(app, manifest: manifest)
            try FileManager.default.removeItem(at: archive)
            return app
        } catch let error as NativeUpdateArchiveError { throw UpdateFailure(error.description) }
    }
    private func validateApp(_ app: URL, manifest: UpdateManifest) throws {
        try validateApp(app, version: manifest.version, build: manifest.build)
    }
    private func validateApp(_ app: URL, version: String, build: Int) throws {
        guard let identifier = Bundle.main.bundleIdentifier, let resources = Bundle.main.resourceURL,
              let launcher = try JSONSerialization.jsonObject(with: Data(contentsOf: resources.appendingPathComponent("launcher.json"))) as? [String: String],
              let project = launcher["project"] else { throw UpdateFailure("启动配置不可用。") }
        do {
            try NativeUpdateArchive.validateApp(app, bundleIdentifier: identifier, version: version,
                                                build: build, currentProject: project)
        } catch let error as NativeUpdateArchiveError { throw UpdateFailure(error.description) }
    }

    static var installationScript: String {
        """
        #!/bin/sh
        app="$1"; staged="$2"; backup="$3"; app_pid="$4"; server_pid="$5"; cache="$6"; service="$7"; version="$8"; build="$9"
        fail() { /bin/rm -rf "$staged"; /bin/rm -f "$service"; /usr/bin/osascript -e 'display alert "鲁班更新失败" message "应用已保留，请重新启动后重试。"'; exit 1; }
        count=0
        while /bin/kill -0 "$app_pid" 2>/dev/null || /bin/kill -0 "$server_pid" 2>/dev/null; do
          count=$((count + 1)); [ "$count" -lt 60 ] || fail
          /bin/sleep 1
        done
        /bin/mv "$app" "$backup" || fail
        if ! /bin/mv "$staged/Luban.app" "$app"; then /bin/mv "$backup" "$app"; fail; fi
        "$app/Contents/MacOS/Luban" >/dev/null 2>&1 &
        replacement_pid=$!
        if ! "$service" health "$version" "$build"
        then
          /bin/kill -TERM "$replacement_pid" 2>/dev/null
          count=0
          while /bin/kill -0 "$replacement_pid" 2>/dev/null; do
            count=$((count + 1)); [ "$count" -lt 60 ] || fail
            /bin/sleep 1
          done
          if ! "$service" quiescence
          then fail; fi
          /bin/rm -rf "$app"; /bin/mv "$backup" "$app" || fail; /usr/bin/open -n -a "$app"; fail
        fi
        /bin/rm -rf "$backup" "$staged" "$cache"
        /bin/rm -f "$service" "$0"
        """
    }

    func installationError(_ message: String) { setState("ready", error: message) }
    func install(appPID: Int32, serverPID: Int32) throws {
        guard state == "ready", let source = preparedApp, let manifest = candidate else {
            throw UpdateFailure("更新尚未下载并通过校验。")
        }
        let started = Date()
        let fm = FileManager.default
        let app = Bundle.main.bundleURL.standardizedFileURL
        let parent = app.deletingLastPathComponent()
        guard app.pathExtension == "app", !app.path.hasPrefix("/Volumes/"),
              app.resolvingSymlinksInPath() == app, fm.isWritableFile(atPath: parent.path),
              fm.isWritableFile(atPath: app.path) else { throw UpdateFailure("应用所在目录不能安全写入，请移到本机可写目录后更新。") }
        let staging = parent.appendingPathComponent(".Luban-update-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let stagedApp = staging.appendingPathComponent("Luban.app", isDirectory: true)
        var detachedService: URL?
        do {
            try fm.copyItem(at: source, to: stagedApp)
            try validateApp(stagedApp, manifest: manifest)
            guard Date().timeIntervalSince(started) < 40 else { throw UpdateFailure("准备更新耗时过长，请重试以保护正在运行的任务。") }
            let backup = parent.appendingPathComponent(".Luban-backup-" + UUID().uuidString + ".app")
            let helper = home.appendingPathComponent("install-" + UUID().uuidString + ".sh")
            guard let bundledService = Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("luban-update-service"),
                  fm.isExecutableFile(atPath: bundledService.path) else { throw UpdateFailure("原生更新检查程序不可用。") }
            let service = home.appendingPathComponent("service-" + UUID().uuidString)
            try fm.copyItem(at: bundledService, to: service)
            detachedService = service
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: service.path)
            let script = Self.installationScript
            try Data(script.utf8).write(to: helper, options: .atomic)
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/sh")
            process.arguments = [helper.path, app.path, staging.path, backup.path, String(appPID), String(serverPID), source.deletingLastPathComponent().deletingLastPathComponent().path, service.path, manifest.version, String(manifest.build)]
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            try process.run()
            setState("installing")
        } catch {
            try? fm.removeItem(at: staging)
            if let detachedService { try? fm.removeItem(at: detachedService) }
            throw error
        }
    }
}
