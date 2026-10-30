/** New durable generations on the user's volume; strict native writers keep their local scratch. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, unlink, readdir } from "node:fs/promises";
import { dirname, isAbsolute, normalize, join } from "node:path";
import { dlopen, ptr } from "bun:ffi";

const system = dlopen("/usr/lib/libSystem.B.dylib", { statfs64: { args: ["ptr", "ptr"], returns: "i32" } });
/** Match Python TrainingSession.save: preserve the journal's current and previous generations. */
export async function pruneNativeGenerations(directory:string,kept:string[],assertCurrent:()=>void){
  assert(kept.length>=1&&kept.length<=2&&kept.every(name=>/^[a-f0-9]{32}\.pt$/.test(name)));
  const before=await assertNativePersistentDirectory(directory),fd=await open(directory,constants.O_RDONLY|constants.O_NOFOLLOW);
  try{for(const name of await readdir(directory)){
    if(!/^[a-f0-9]{32}\.pt$/.test(name)||kept.includes(name))continue;
    assertCurrent();const current=await assertNativePersistentDirectory(directory);assert(current.dev===before.dev&&current.ino===before.ino);
    const path=join(directory,name),stat=await lstat(path);assert(stat.isFile()&&!stat.isSymbolicLink()&&stat.nlink===1&&stat.uid===process.getuid!(),'Old checkpoint generation is not a regular owned file');
    await unlink(path);
  }await fd.sync();}finally{await fd.close();}
}
export function nativeStorageFilesystem(path: string) {
  const storage = Buffer.alloc(4096), name = Buffer.from(path + "\0");
  assert.equal(system.symbols.statfs64(ptr(name), ptr(storage)), 0, "Storage volume is unavailable");
  return storage.subarray(72, 88).toString().split("\0")[0]!;
}
export async function assertNativePersistentDirectory(path: string) {
  assert(isAbsolute(path) && normalize(path) === path && !path.includes("\0"));
  assert.equal(await realpath(path), path, "Persistent directory contains links");
  const stat = await lstat(path, { bigint: true });
  assert(stat.isDirectory() && !stat.isSymbolicLink() && stat.uid === BigInt(process.getuid!()), "Persistent directory owner/type differs");
  // exFAT has no POSIX privacy bits. This is storage compatibility, never a claim of private permissions.
  assert((stat.mode & 0o7777n) === 0o700n || nativeStorageFilesystem(path) === "exfat", "Private directory required on POSIX storage");
  return stat;
}
export async function persistNativeGeneration(input: { source: string; destination: string; expectedSHA: string; signal?: AbortSignal }) {
  const { source, destination, expectedSHA, signal } = input;
  assert(/^[a-f0-9]{64}$/.test(expectedSHA) && isAbsolute(destination) && normalize(destination) === destination && !destination.includes("\0"));
  signal?.throwIfAborted();
  assert.equal(await realpath(dirname(source)), dirname(source), "Generation source parent contains links");
  const parent = dirname(destination), parentStat = await assertNativePersistentDirectory(parent);
  const directory = await open(parent, constants.O_RDONLY | constants.O_NOFOLLOW);
  let from: Awaited<ReturnType<typeof open>> | undefined, to: Awaited<ReturnType<typeof open>> | undefined;
  let created: Awaited<ReturnType<typeof lstat>> | undefined;
  const sameParent = async () => {
    const held = await directory.stat({ bigint: true }), named = await assertNativePersistentDirectory(parent);
    assert(held.dev === parentStat.dev && held.ino === parentStat.ino && named.dev === held.dev && named.ino === held.ino,
      "Persistent volume or directory changed");
  };
  try {
    from = await open(source, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await from.stat({ bigint: true }), named = await lstat(source, { bigint: true });
    assert(before.isFile() && before.nlink === 1n && before.size > 0n && before.size <= 2n ** 31n
      && before.uid === BigInt(process.getuid!()) && (before.mode & 0o7777n) === 0o600n
      && before.dev === named.dev && before.ino === named.ino, "Strict local generation source required");
    await sameParent();
    to = await open(destination, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
    created = await to.stat();
    assert(created.isFile() && created.nlink === 1, "Fresh regular generation required");
    const bytes = Number(before.size), buffer = Buffer.alloc(1 << 20), hash = createHash("sha256");
    for (let at = 0; at < bytes;) {
      signal?.throwIfAborted();
      const n = (await from.read(buffer, 0, Math.min(buffer.length, bytes - at), at)).bytesRead;
      assert(n > 0, "Generation source truncated"); hash.update(buffer.subarray(0, n));
      for (let offset = 0; offset < n;) {
        signal?.throwIfAborted();
        const written = (await to.write(buffer, offset, n - offset, at + offset)).bytesWritten;
        assert(written > 0, "Generation write incomplete"); offset += written;
      }
      at += n;
    }
    assert.equal(hash.digest("hex"), expectedSHA, "Generation changed during persistent copy");
    const after = await from.stat({ bigint: true }), sourceName = await lstat(source, { bigint: true });
    assert(after.dev === before.dev && after.ino === before.ino && after.size === before.size
      && after.mtimeNs === before.mtimeNs && after.ctimeNs === before.ctimeNs
      && sourceName.dev === before.dev && sourceName.ino === before.ino, "Generation source changed during copy");
    await to.sync();
    const stored = await to.stat(), destinationName = await lstat(destination);
    assert(stored.size === bytes && stored.nlink === 1 && stored.dev === destinationName.dev && stored.ino === destinationName.ino,
      "Persistent generation binding changed");
    await sameParent(); await directory.sync(); signal?.throwIfAborted();
    return { path: destination, bytes, sha256: expectedSHA, file_fsynced: true, directory_fsynced: true };
  } catch (error) {
    if (created) {
      const named = await lstat(destination).catch(() => null);
      if (named && named.dev === created.dev && named.ino === created.ino) { await unlink(destination); await directory.sync(); }
    }
    throw error;
  } finally { await Promise.allSettled([...(from ? [from.close()] : []), directory.close(), ...(to ? [to.close()] : [])]); }
}
