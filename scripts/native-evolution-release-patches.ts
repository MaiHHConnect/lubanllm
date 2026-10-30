/** Keep first full admission; subsequent checks only compare held FD/path metadata. Frozen input sources remain untouched. */
import { join } from "node:path";
export function applyNativeEvolutionReleasePatches({ bodies, target, replace, root }: {
  bodies: Map<string, string>; target(path: string): string; replace(body: string, before: string, after: string): string; root: string;
}) {
  const helper = JSON.stringify(join(root, "backend/native-evolution-release.ts"));
  const admission = target(join(root, ".native-migration/research/native-unified-service/admission-v6.ts"));
  let body = bodies.get(admission)!;
  body = "import {borrowNativeEvolutionRelease,retainNativeReleaseFailure} from " + helper + ";\n" + body;
  body = replace(body, "export async function verifyRelease(root:string,authority:ReleaseAuthority,signal?:AbortSignal){",
    "export function verifyRelease(root:string,authority:ReleaseAuthority,signal?:AbortSignal){return borrowNativeEvolutionRelease(root,authority,()=>verifyFreshRelease(root,authority,signal),signal);}\nasync function verifyFreshRelease(root:string,authority:ReleaseAuthority,signal?:AbortSignal){");
  const guarded = body.includes('close:async function close(){if(!closed){assertObservationMutationAllowed(close);');
  body = replace(body, " let closed=false;", " let closed=false;\n const close=async()=>{if(closed)return;" + (guarded ? "assertObservationMutationAllowed(close);" : "") + "const results=await Promise.allSettled(owners.map(o=>o.handle.fd===-1?Promise.resolve():o.handle.close()));const failures=results.filter((v):v is PromiseRejectedResult=>v.status==='rejected');if(failures.length)throw new AggregateError(failures.map(v=>v.reason),'Release FD close failed');assert(owners.every(o=>o.handle.fd===-1));closed=true;};");
  if (body.includes("  const verifyComplete=async()=>{")) {
    // The current composition already made assertUnchanged stat-only. Its extra complete read must share that binding too.
    body = replace(body, `  const verifyComplete=async()=>{
   await assertUnchanged();
   for(const o of owners){await unchangedNativeFile(o.path,o.handle,o.stat as any);
    const raw=await readNativeSmallFile(o.handle,o.binding.bytes);assert.equal(sha(raw),o.binding.sha256);
    await unchangedNativeFile(o.path,o.handle,o.stat as any);
   }
   await assertUnchanged();
  };`, "  const verifyComplete=assertUnchanged;");
    body = replace(body, "  // Complete byte verification belongs to a lifecycle boundary, never an individual command.",
      "  // Initial admission already verified all bytes; reuse its held files for this Engine lifetime.");
  } else {
    body = replace(body,
      "    assert((await o.handle.stat()).nlink===1);const raw=await readNativeSmallFile(o.handle,o.binding.bytes);assert.equal(sha(raw),o.binding.sha256);await unchangedNativeFile(o.path,o.handle,o.stat as any);",
      "    assert((await o.handle.stat()).nlink===1);");
    body = replace(body, "   for(const[path,image]of Object.entries(images))assert.deepEqual(inspect(path,commands),image,'Runtime linkage/signature changed');", "");
  }
  body = replace(body, (guarded ? 'close:async function close(){if(!closed){assertObservationMutationAllowed(close);' : 'close:async()=>{if(!closed){') + 'const results=await Promise.allSettled(owners.map(o=>o.handle.close()));const failure=results.find(v=>v.status==="rejected");if(failure?.status==="rejected")throw failure.reason;assert(owners.every(o=>o.handle.fd===-1),"Role owner FD closure unconfirmed");closed=true;}}', "close");
  body = replace(body, " }catch(error){await Promise.allSettled(owners.map(o=>o.handle.close()));throw error;}",
    " }catch(error){try{await close();}catch(cleanup){retainNativeReleaseFailure(close);throw new AggregateError([error,cleanup],'Release admission and cleanup failed');}throw error;}");
  bodies.set(admission, body);
  const engine = target(join(root, ".native-migration/research/native-unified-expert/native-engine.ts"));
  body = "import {withNativeEvolutionRelease} from " + helper + ";\n" + bodies.get(engine)!;
  body = replace(body, 'return this.withQuiescent(action,"source",signal) as Promise<T>;',
    'return withNativeEvolutionRelease(this,()=>this.withQuiescent(action,"source",signal) as Promise<T>);');
  bodies.set(engine, body);
}
