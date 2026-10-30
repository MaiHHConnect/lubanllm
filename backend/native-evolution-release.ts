/** A live Engine owns one admitted package binding; borrowers never reread the same binaries. */
import assert from "node:assert/strict";
import { AsyncLocalStorage } from "node:async_hooks";
import { assertObservationMutationAllowed } from "../.native-migration/research/native-observation-python-parity-v2/scope.ts";

type Binding = { assertUnchanged(): Promise<void>; assertFilesUnchanged?(): Promise<void>; verifyComplete?(): Promise<void>; close(): Promise<void> };
type Entry = { opened: Promise<Binding>; borrowers: number };
type Scope = { entries: Map<string, Entry>; failed: Set<() => Promise<void>>; closing: boolean; closed: boolean };
const contexts = new WeakMap<object, Scope>(), active = new AsyncLocalStorage<Scope>();
function context(owner: object) {
  let value = contexts.get(owner);
  if (!value) { value = { entries: new Map(), failed: new Set(), closing: false, closed: false }; contexts.set(owner, value); }
  assert(!value.closing && !value.closed, "原生来源生命周期已开始关闭"); return value;
}
export function withNativeEvolutionRelease<T>(owner: object, action: () => T): T {
  return active.run(context(owner), action);
}
/** If initial verification itself fails to close an FD, keep that exact cleanup reachable by Engine.close(). */
export function retainNativeReleaseFailure(close: () => Promise<void>) { const scope = active.getStore(); if (scope) scope.failed.add(close); }
function freeze(value: any): void {
  if (value && typeof value === "object" && !Object.isFrozen(value)) { for (const item of Object.values(value)) freeze(item); Object.freeze(value); }
}
export async function borrowNativeEvolutionRelease<T extends Binding>(root: string, authority: unknown,
  verify: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  signal?.throwIfAborted(); const scope = active.getStore();
  // Non-evolution callers retain their original independently owned admission.
  if (!scope) return verify();
  assert(!scope.closing && !scope.closed && !scope.failed.size, "原生来源仍在等待清理");
  const key = root + "\0" + JSON.stringify(authority);
  let entry = scope.entries.get(key);
  if (!entry) {
    entry = { opened: Promise.resolve().then(verify), borrowers: 0 }; scope.entries.set(key, entry);
    const current = entry;
    void current.opened.catch(() => { if (scope.entries.get(key) === current) scope.entries.delete(key); });
  }
  const binding = await entry.opened as T;
  signal?.throwIfAborted(); assert(!scope.closing && !scope.closed);
  await binding.assertUnchanged(); signal?.throwIfAborted(); assert(!scope.closing && !scope.closed);
  for (const name of ["manifest", "commands", "files"]) freeze((binding as any)[name]);
  entry.borrowers++; const held = entry; let closed = false;
  const check = async () => { assert(!closed && !scope.closed); await binding.assertUnchanged(); };
  return { ...binding, assertUnchanged: check,
    ...(binding.assertFilesUnchanged ? { assertFilesUnchanged: check } : {}), ...(binding.verifyComplete ? { verifyComplete: check } : {}),
    close: async function close() { if (closed) return; assertObservationMutationAllowed(close); assert(held.borrowers > 0); held.borrowers--; closed = true; } };
}
/** The Engine is the final owner. Never release a package while a program borrower still holds it. */
export async function closeNativeEvolutionRelease(owner: object) {
  const scope = contexts.get(owner); if (!scope || scope.closed) return;
  scope.closing = true;
  for (const entry of scope.entries.values()) { await entry.opened.catch(() => {}); assert(entry.borrowers === 0, "原生程序仍持有发行来源，不能提前关闭"); }
  const errors: unknown[] = [];
  for (const [key, entry] of scope.entries) try { await (await entry.opened).close(); scope.entries.delete(key); } catch (error) { errors.push(error); }
  for (const close of scope.failed) try { await close(); scope.failed.delete(close); } catch (error) { errors.push(error); }
  if (errors.length) throw new AggregateError(errors, "发行来源FD未全部关闭，保留同一所有者重试");
  scope.closed = true;
}
