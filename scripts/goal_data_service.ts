/** Private data owner for one isolated teaching run. No compute worker. */
import { randomBytes } from "node:crypto";
import { resolve } from "node:path";
import { Settings } from "../backend/settings.ts";
import { createBackend } from "../backend/server.ts";

const home = Bun.argv[2], token = process.env.LUBAN_BUN_RPC_TOKEN;
if (!home || Bun.argv.length !== 3 || !token || !/^[!-~]+$/.test(token)) throw new Error("Private owner arguments invalid");
const settings = new Settings(resolve(home));
const application = createBackend({ settings, port: 0, computeURL: "http://127.0.0.1:9",
  computeToken: randomBytes(32).toString("base64url"), rpcToken: token,
  webRoot: resolve(home), version: "isolated-teaching", build: 1 });
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, idleTimeout: 255,
  fetch(request) {
    return new URL(request.url).pathname === "/internal/rpc"
      ? application.fetch(request) : new Response("Not Found", { status: 404 });
  } });
const close = () => { application.close(); server.stop(true); process.exit(0); };
process.on("SIGINT", close); process.on("SIGTERM", close);
const parent = Number(process.env.LUBAN_GOAL_PARENT_PID);
if (!Number.isSafeInteger(parent) || parent < 1) throw new Error("Private parent missing");
setInterval(() => { try { process.kill(parent, 0); } catch { close(); } }, 1000).unref();
console.log(`http://127.0.0.1:${server.port}/internal/rpc`);
