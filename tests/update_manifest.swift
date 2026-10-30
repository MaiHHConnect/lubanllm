import Foundation
@main struct ManifestTests {
 static func main() throws {
  var base: [String:Any] = ["version":"0.1.1","build":2,"url":"https://ai.maihh.net/updates/luban-0.1.1.zip","sha256":String(repeating:"a",count:64),"size":1,"notes":["新增设置"],"platform":"macos","architecture":"arm64"]
  #if arch(x86_64)
  base["architecture"] = "x64"
  #endif
  func parse(_ values: [String:Any]) throws -> UpdateManifest {
   try JSONDecoder().decode(UpdateManifest.self, from: JSONSerialization.data(withJSONObject:values))
  }
  var count = 0
  func invalid(_ field: String, _ value: Any) throws {
   var values=base; values[field]=value
   do { _ = try parse(values).validate(currentVersion:"0.1.0",currentBuild:1); fatalError("accepted invalid \(field):\(value)") }
   catch { count += 1 }
  }
  let newer = try parse(base).validate(currentVersion:"0.1.0",currentBuild:1); assert(newer); count += 1
  let sameVersion = try parse(base).validate(currentVersion:"0.1.1",currentBuild:1); assert(!sameVersion); count += 1
  let sameBuild = try parse(base).validate(currentVersion:"0.1.0",currentBuild:2); assert(!sameBuild); count += 1
  for url in ["http://ai.maihh.net/updates/app.zip", "https://evil.test/updates/app.zip", "https://ai.maihh.net:444/updates/app.zip", "https://user@ai.maihh.net/updates/app.zip", "https://ai.maihh.net/updates/../a.zip", "https://ai.maihh.net/app.zip", "https://ai.maihh.net/updates/app.zip?q=x"] { try invalid("url",url) }
  for version in ["1.0", "01.0.1", "1.0.1-beta", "a.0.1", "1.0.1000000"] { try invalid("version",version) }
  try invalid("notes",[]); try invalid("notes",[" "]); try invalid("notes",Array(repeating:"a",count:6))
  try invalid("notes",[String(repeating:"中",count:301)]); try invalid("sha256",String(repeating:"g",count:64))
  try invalid("size",0); try invalid("size",512*1024*1024+1); try invalid("build",0)
  try invalid("architecture","wrong"); try invalid("platform","windows")
  try invalid("notes",["line\nbreak"]); try invalid("notes",["nul\u{0000}"]); try invalid("notes",["c1\u{0085}"])
  let delta: [String:Any] = ["from_version":"0.1.0","from_build":1,
    "url":"https://ai.maihh.net/updates/Luban-0.1.1-from-0.1.0-macos-\(base["architecture"]!)-delta.zip",
    "sha256":String(repeating:"b",count:64),"size":1,"removed":["Contents/Resources/old.txt"]]
  var modular=base;modular["delta"]=delta
  let validDelta = try parse(modular).validate(currentVersion:"0.1.0",currentBuild:1)
  assert(validDelta);count += 1
  for (key,value): (String,Any) in [("from_version","0.1.1"),("from_build",2),("size",0),("sha256","bad"),
    ("url","https://evil.test/updates/module.zip"),("url","https://ai.maihh.net/updates/Luban-0.1.2-from-0.1.0-macos-arm64-delta.zip"),
    ("removed",["../outside"]),("removed",["Contents/../outside"]),("removed",["Contents/Resources/"]),
    ("removed",["Contents/Resources/a","Contents/Resources/A"]),("removed",["Contents/internal/a"])] {
      var bad=delta;bad[key]=value;try invalid("delta",bad)
  }
  try Data(DesktopUpdater.installationScript.utf8).write(to: URL(fileURLWithPath:CommandLine.arguments[1] + "/install.sh"))
  print("Manifest validation: \(count) cases passed")
 }
}
