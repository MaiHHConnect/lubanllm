import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBackend } from "../backend/server";
import { Settings } from "../backend/settings";

const cleanup: (() => void)[] = [];
afterEach(() => { for (const close of cleanup.splice(0).reverse()) close(); });

test("disconnected queued db_begin cannot acquire an orphan lease after the owner closes", async () => {
  const directory = mkdtempSync(join(tmpdir(), "luban-rpc-cancel-"));
  cleanup.push(() => rmSync(directory, { recursive: true, force: true }));
  const settings = new Settings(join(directory, "home"));
  settings.update({ storage_root: join(directory, "storage") });
  const backend = createBackend({ settings, port: 0, computeURL: "http://127.0.0.1:1",
    computeToken: "compute-fixture", rpcToken: "rpc-fixture", webRoot: directory, version: "test", build: 1 });
  cleanup.push(() => backend.close());
  const entered = Promise.withResolvers<void>(), disconnected = Promise.withResolvers<void>();
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    if (request.headers.has("x-fixture-queued")) {
      request.signal.addEventListener("abort", () => disconnected.resolve(), { once: true });
      entered.resolve();
    }
    return backend.fetch(request);
  } });
  cleanup.push(() => server.stop(true));
  const rpc = async (method: string, transaction?: string, signal?: AbortSignal, queued = false) => {
    const response = await fetch(`http://127.0.0.1:${server.port}/internal/rpc`, { method: "POST", signal,
      headers: { authorization: "Bearer rpc-fixture", ...(queued ? { "x-fixture-queued": "yes" } : {}) },
      body: JSON.stringify({ target: "memory", root: settings.snapshot().storage_root, method, transaction,
        args: [], kwargs: method === "db_execute" ? { sql: "SELECT 1" } : {} }) });
    return response.json() as Promise<any>;
  };
  const owner = (await rpc("db_begin")).result.transaction;
  const controller = new AbortController();
  const abandoned = rpc("db_begin", undefined, controller.signal, true).catch(error => error);
  await entered.promise; await Bun.sleep(20);
  controller.abort(); await abandoned;
  await Promise.race([disconnected.promise, Bun.sleep(1000).then(() => { throw new Error("HTTP disconnect not observed"); })]);
  expect((await rpc("db_execute", owner)).ok).toBe(true);
  expect((await rpc("db_close", owner)).ok).toBe(true);
  const next = new AbortController(); cleanup.push(() => next.abort());
  const result = await Promise.race([rpc("events", undefined, next.signal), Bun.sleep(1000).then(() => "blocked by orphan lease")]);
  expect(result).toEqual({ ok: true, result: [] });
});
