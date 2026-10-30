/** Narrow build overlay: effective training identities, unchanged physical CAS and restoration. */
import { join } from "node:path";
export function applyNativeEvolutionTrainingSourcePatches({ bodies, target, replace, root }: {
  bodies: Map<string, string>; target(path: string): string; replace(body: string, before: string, after: string): string; root: string;
}) {
  const helper = JSON.stringify(join(root, "backend/native-evolution-training-source.ts"));
  const start = target(join(root, ".native-migration/backend/native-training-start-v4.ts"));
  let body = bodies.get(start)!;
  body = "import {assertNativeEvolutionTrainingSource,assertNativeTrainingPointerIdentity,type NativeEvolutionTrainingSource} from " + helper + ";\n" + body;
  body = replace(body, " guard:{modelLockFd:number;assertQuiescent():void};assertSource():void};",
    " evolutionSource?:NativeEvolutionTrainingSource;guard:{modelLockFd:number;assertQuiescent():void};assertSource():void};");
  body = replace(body, " r.guard.assertQuiescent();r.assertSource();", " r.guard.assertQuiescent();r.assertSource();assertNativeEvolutionTrainingSource(r);");
  body = replace(body,
    " for(const[name,key]of[['active-core.json','core_version'],['active.json','expert_id']]as const){let value=null;try{const p=ledgerParse(readSafe(r.store,name,4<<20).toString());value=p.version??p.id;}catch(e:any){if(e.code!=='ENOENT')throw e;}need(value===r[key],'Native start accepted pointer identity differs');}",
    " const physicalPointers={core_version:null as unknown,expert_id:null as unknown};for(const[name,key]of[['active-core.json','core_version'],['active.json','expert_id']]as const){try{const p=ledgerParse(readSafe(r.store,name,4<<20).toString());physicalPointers[key]=p.version??p.id;}catch(e:any){if(e.code!=='ENOENT')throw e;}}assertNativeTrainingPointerIdentity(r,physicalPointers);");
  bodies.set(start, body);
  const boot = target(join(root, ".native-migration/research/native-core450-owner-controller/bootstrap.ts"));
  body = bodies.get(boot)!;
  const index = body.indexOf("export async function createCore450ProductFreshCheckpoint(");
  if (index < 0) throw new Error("Actual product bootstrap is missing");
  let product = body.slice(index);
  product = replace(product, "   session=await startNativeTrainingSessionV4(", "   const trainingSource=nativeEvolutionTrainingSource(engine,guard,cohort);\n   session=await startNativeTrainingSessionV4(");
  product = replace(product, "core_version:r.preparation.graphBinding.accepted_core_version,expert_id:borrowNativeEnginePeer(engine,guard).previous?.id??null,",
    "core_version:trainingSource.core_version,expert_id:trainingSource.expert_id,evolutionSource:trainingSource,");
  bodies.set(boot, "import {nativeEvolutionTrainingSource} from " + helper + ";\n" + body.slice(0, index) + product);
}
