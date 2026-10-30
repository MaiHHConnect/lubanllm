import Foundation
import Darwin

// Small detached update helper: it survives replacement of the running app bundle.
private final class HealthProbe: NSObject, URLSessionDataDelegate {
    private let done = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var response: HTTPURLResponse?
    private var body = Data()
    private var error: Error?
    private var oversized = false

    struct Result {
        let response: HTTPURLResponse?
        let body: Data
        let error: Error?
        let oversized: Bool
    }

    func request(_ url: URL, timeout: Double) -> Result {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        config.timeoutIntervalForResource = timeout
        config.httpCookieStorage = nil
        config.urlCache = nil
        config.connectionProxyDictionary = [:]
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout)
        request.httpMethod = "GET"
        let task = session.dataTask(with: request)
        task.resume()
        let completed = done.wait(timeout: .now() + timeout + 0.1) == .success
        if !completed { task.cancel() }
        lock.lock(); defer { lock.unlock() }
        return Result(response: response, body: body,
                      error: completed ? error : URLError(.timedOut), oversized: oversized)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        // A local service must never redirect the helper to another origin.
        completionHandler(nil)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        lock.lock()
        self.response = response as? HTTPURLResponse
        oversized = response.expectedContentLength > 64 * 1024
        lock.unlock()
        completionHandler(oversized ? .cancel : .allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        if data.count > 64 * 1024 - body.count { oversized = true }
        else { body.append(data) }
        let cancel = oversized
        lock.unlock()
        if cancel { dataTask.cancel() }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock(); self.error = error; lock.unlock()
        done.signal()
    }
}

@main struct NativeUpdateService {
    private static let endpoint = URL(string: "http://127.0.0.1:8765/api/health")!
    private static func clock() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

    private static func validEndpoint(_ url: URL) -> Bool {
        url.scheme == "http" && url.host == "127.0.0.1" && (1...65535).contains(url.port ?? 80)
            && url.path == "/api/health" && url.user == nil && url.password == nil
            && url.query == nil && url.fragment == nil
    }

    private static func matches(_ result: HealthProbe.Result, version: String, build: Int) -> Bool {
        guard result.error == nil, !result.oversized, result.response?.statusCode == 200,
              let value = try? JSONSerialization.jsonObject(with: result.body) as? [String: Any],
              value["app"] as? String == "luban", value["version"] as? String == version,
              let number = value["build"] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              ["c", "s", "i", "l", "q", "C", "S", "I", "L", "Q"].contains(String(cString: number.objCType)),
              number.int64Value == Int64(build) else { return false }
        return true
    }

    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        if args == ["--help"] {
            print("luban-update-service health VERSION BUILD [LOCAL_URL TIMEOUT_SECONDS]\nluban-update-service quiescence [LOCAL_URL TIMEOUT_SECONDS]")
            return
        }
        let health = args.first == "health"
        let base = health ? 3 : 1
        guard (health || args.first == "quiescence"), args.count == base || args.count == base + 2 else { exit(2) }
        let url = args.count == base ? endpoint : URL(string: args[base])
        let timeout = args.count == base ? (health ? 180.0 : 60.0) : Double(args[base + 1])
        guard let url, validEndpoint(url), let timeout, timeout.isFinite, (0.05...1800).contains(timeout) else { exit(2) }
        var version = "", build = 0
        if health {
            version = args[1]
            guard UpdateManifest.parts(version) != nil, let parsed = Int(args[2]), parsed > 0,
                  String(parsed) == args[2] else { exit(2) }
            build = parsed
        }
        let deadline = clock() + timeout
        var attempts = 0, passed = false
        while clock() < deadline {
            attempts += 1
            let result = HealthProbe().request(url, timeout: min(2, deadline - clock()))
            if health { passed = matches(result, version: version, build: build) }
            else {
                // HTTP errors, invalid JSON, redirects and timeouts still mean the service
                // may be alive. Only a refused local connection admits rollback.
                passed = result.response == nil && (result.error as? URLError)?.code == .cannotConnectToHost
            }
            if passed { break }
            let remaining = deadline - clock()
            if remaining > 0 { Thread.sleep(forTimeInterval: min(1, remaining)) }
        }
        print("{\"mode\":\"\(health ? "health" : "quiescence")\",\"passed\":\(passed),\"attempts\":\(attempts)}")
        if !passed { exit(1) }
    }
}
