/** Run the shipped service with developer files denied by macOS, in an isolated app home. */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const [appArgument, homeArgument, outputArgument, mode = "empty"] = Bun.argv.slice(2);
assert(appArgument && homeArgument && outputArgument && ["empty", "chat", "factory"].includes(mode),
  "Usage: bun scripts/verify-standalone-app.ts /candidate/Luban.app /isolated-home /evidence.json [empty|chat|factory]");
const app = resolve(appArgument), home = resolve(homeArgument), output = resolve(outputArgument);
assert(home !== join(homedir(), "Library/Application Support/Luban") && !existsSync(join(home, "launch.json")),
  "Use an isolated home without a running service");
await mkdir(home, { recursive: true });
const temp = join(home, "tmp"); await mkdir(temp, { recursive: true });
const resources = join(app, "Contents/Resources"), log = Bun.file(join(home, "standalone.log"));
const denied = [resolve(import.meta.dir, ".."), join(homedir(), ".local/share/uv"),
  join(homedir(), ".cache/uv"), join(homedir(), "Library/Application Support/Luban")];
assert(!denied.some(path => app.startsWith(path + "/") || home.startsWith(path + "/")), "Candidate must be outside denied developer paths");
const policy = "(version 1) (allow default) " + denied.map(path => `(deny file-read* (subpath ${JSON.stringify(path)}))`).join(" ");
const child = Bun.spawn(["/usr/bin/sandbox-exec", "-p", policy, join(app, "Contents/MacOS/luban-service"),
  "--project", resources, "--backend", join(resources, "backend/src"), "--home", home, "--port", "0"],
  { cwd: home, env: { PATH: "/usr/bin:/bin", LANG: "en_US.UTF-8", TMPDIR: temp }, stdout: log, stderr: log });
const evidence: Record<string, unknown> = { schema: "luban-standalone-smoke-v1", app, home,
  mode, developer_files_denied: denied, teacher_calls: 0, model_autoload: false, passed: false };
const pause = () => new Promise(done => setTimeout(done, 500));
try {
  const launchPath = join(home, "launch.json");
  for (let i = 0; !existsSync(launchPath) && i < 120; i++) {
    assert(child.exitCode === null, "Service exited; inspect isolated standalone.log"); await pause();
  }
  const launch = JSON.parse(await readFile(launchPath, "utf8"));
  const origin = `http://127.0.0.1:${launch.port}`;
  async function request(path: string, body?: unknown) {
    const response = await fetch(origin + path, { method: body === undefined ? "GET" : "POST",
      headers: { cookie: "luban_session=" + launch.token, origin, "content-type": "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), proxy: null,
      signal: AbortSignal.timeout(180_000) });
    const text = await response.text();
    assert(response.ok, `${path}: HTTP ${response.status}: ${text.slice(0, 1000)}`);
    return path === "/" ? text : JSON.parse(text);
  }
  let health;
  for (let i = 0; i < 120; i++) {
    health = await request("/api/health"); if (health.state === "stopped") break; await pause();
  }
  assert.equal(health.state, "stopped", "App must not load the model on open");
  evidence.initial_health = health;
  const packages = await request("/api/model-packages");
  assert.equal(packages.installation.required, mode === "empty");
  evidence.installation = packages.installation;
  assert((await request("/")).includes('id="install-models"'), "Packaged UI must contain first-install action");
  if (mode !== "empty") {
    await request("/api/model/reload", {});
    for (let i = 0; i < 240; i++) {
      health = await request("/api/health"); if (["ready", "error"].includes(health.state)) break; await pause();
    }
    assert.equal(health.state, "ready", "Actual first-install model load failed");
    const response = await fetch(origin + "/api/chat", { method: "POST", headers: {
      cookie: "luban_session=" + launch.token, origin, "content-type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "1加1等于几？只回答数字。" }],
        route: "local", thinking: "off", temperature: 0, max_tokens: 32 }), proxy: null,
      signal: AbortSignal.timeout(180_000) });
    assert(response.ok, "Actual chat HTTP failed");
    const events = (await response.text()).split("\n").filter(line => line.startsWith("data: "))
      .map(line => JSON.parse(line.slice(6)));
    assert(!events.some(event => event.type === "error"), JSON.stringify(events.filter(event => event.type === "error")));
    const answer = events.filter(event => event.type === "delta").map(event => event.content ?? "").join("").trim();
    assert(answer && events.some(event => event.type === "done"), "Actual model did not complete a response");
    evidence.chat = { answer, completed: true, route: "local", correct: answer === "2",
      usage: events.find(event => event.type === "done")?.usage };
    if (mode === "factory") {
      const status = await request("/api/status");
      assert.equal(status.runtime.active_expert, "learned-bafb4050374f");
      assert.equal(status.runtime.core_version, "core-fdfa0ce5a75c");
      assert.equal(status.automatic.learning_count, 0);
      assert.equal(status.automatic.evolution_count, 0);
      assert.equal(status.automatic.fastbrain_updates, 0);
      assert.notEqual(status.automatic.state, "waiting_validation", JSON.stringify(status.automatic));
      const history = await request("/api/experts");
      assert.deepEqual(history.versions, []);
      assert.equal(history.last_attempt, null);
      const learning = await fetch(origin + "/api/learn", { method: "POST", headers: {
        cookie: "luban_session=" + launch.token, origin, "content-type": "application/json" },
        body: "{}", proxy: null, signal: AbortSignal.timeout(180_000) });
      const shortage = await learning.text();
      assert(!learning.ok && /资料不足|训练 0 \/ 16/.test(shortage), shortage);
      const answers = [];
      for (const prompt of ["是谁把你微调成现在这样的？", "你现在用的模型名是什么？"]) {
        const response = await fetch(origin + "/api/chat", { method: "POST", headers: {
          cookie: "luban_session=" + launch.token, origin, "content-type": "application/json" },
          body: JSON.stringify({ messages: [{role: "user", content: prompt}], route: "local", thinking: "off", temperature: 0, max_tokens: 64 }),
          proxy: null, signal: AbortSignal.timeout(180_000) });
        assert(response.ok);
        const events = (await response.text()).split("\n").filter(line => line.startsWith("data: ")).map(line => JSON.parse(line.slice(6)));
        assert(!events.some(event => event.type === "error"), JSON.stringify(events));
        assert(events.some(event => event.type === "done"));
        answers.push({prompt, answer: events.filter(event => event.type === "delta").map(event => event.content ?? "").join("").trim()});
      }
      assert(answers[0].answer.includes("麦好火") && answers[1].answer.includes("luban0"), JSON.stringify(answers));
      evidence.factory = {core: status.runtime.core_version, expert: status.runtime.active_expert,
        automatic: status.automatic, history, shortage, answers, teacher_calls: status.teacher_calls};
    }
    await request("/api/model/stop", {});
  }
  evidence.passed = true;
} catch (error) {
  evidence.error = String(error); throw error;
} finally {
  child.kill("SIGTERM");
  let timer: ReturnType<typeof setTimeout>;
  try { await Promise.race([child.exited, new Promise((_, reject) => {
    timer = setTimeout(() => reject(Error("Standalone service did not stop")), 60_000);
  })]); } finally { clearTimeout(timer!); await writeFile(output, JSON.stringify(evidence, null, 2) + "\n"); }
}
console.log(JSON.stringify(evidence));
