/** Compose the existing save lifecycle with local formatting and real external readback; no frozen source edits. */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export function patchNativeCheckpointStorage(body: string, helper: string, replace: (body: string, before: string, after: string) => string) {
  body = `import {retainNativeEvolutionScratch,removeNativeEvolutionScratch} from ${JSON.stringify(join(helper, '..', 'native-evolution-segment-files.ts'))};\n` + body;
  body = `import {assertNativePersistentDirectory,persistNativeGeneration,pruneNativeGenerations} from ${JSON.stringify(helper)};\nimport {rm as removeLocalGeneration} from 'node:fs/promises';\n` + body;
  body = replace(body, 'need(s.isDirectory()&&s.uid===process.getuid!()&&(s.mode&0o7777)===0o700,"Checkpoint run directory privacy invalid");',
    'await assertNativePersistentDirectory(path);');
  body = replace(body, 'source=join(directory,name);', 'source=join(directory,name),stagedSource=join(work,name);');
  body = replace(body, 'const opened:FileHandle[]=[];let readback:VerifiedNativeCheckpoint|undefined;', 'await retainNativeEvolutionScratch(work);let disposed=false;const opened:FileHandle[]=[];let readback:VerifiedNativeCheckpoint|undefined;');
  body = replace(body, 'close:()=>readback!.close()', `close:async()=>{if(disposed)return;await readback!.close();
      const kept=[descriptor.name,...(body.checkpoint?[body.checkpoint.name]:[])];
      await pruneNativeGenerations(directory,kept,()=>{r.guard.assertQuiescent();r.memory.connect(db=>{const actual=db.query('SELECT * FROM training_runs WHERE id=?').get(expected.id) as Row;need(actual&&sameRow(actual,committed),'Checkpoint generation cleanup journal changed');});});
      await removeNativeEvolutionScratch(work);disposed=true;}`);
  body = replace(body, '"--output",source,"--binary-sha"', '"--output",stagedSource,"--binary-sha"');
  const check = 'const saved=await openCheckpointRaw(source);opened.push(saved.handle);need([0o600,0o700].includes(Number(saved.stat.mode)&0o7777)&&saved.stat.uid===BigInt(process.getuid!())&&await checkpointRawSHA(saved.handle,Number(saved.stat.size))===receipt.checkpoint_sha256,"Checkpoint saved private bytes/SHA differ");';
  body = replace(body, check, check.replace('const saved=', 'const staged=').replaceAll('saved.', 'staged.').replace('openCheckpointRaw(source)', 'openCheckpointRaw(stagedSource)')
    + '\n    await persistNativeGeneration({source:stagedSource,destination:source,expectedSHA:receipt.checkpoint_sha256,signal:r.signal});\n    const saved=await openCheckpointRaw(source);opened.push(saved.handle);');
  if(body.includes('finally{await Promise.allSettled(opened.map(f=>f.close()));}'))return replace(body, 'finally{await Promise.allSettled(opened.map(f=>f.close()));}',
    'finally{await Promise.allSettled(opened.map(f=>f.close()));await removeLocalGeneration(stagedSource,{force:true});}');
  return replace(body, "throw new AggregateError(errors,'Checkpoint temporary/readback owner cleanup unconfirmed');}}",
    "throw new AggregateError(errors,'Checkpoint temporary/readback owner cleanup unconfirmed');}await removeLocalGeneration(stagedSource,{force:true});}");
}
export async function applyNativeEvolutionStoragePatches(input: { bodies: Map<string,string>; target(path: string): string;
  replace(body: string, before: string, after: string): string; root: string }) {
  const { bodies, target, replace, root } = input;
  for(const path of new Set([target(join(root, '.native-migration/backend/native-checkpoint-save.ts')),
    target(join(root,'.native-migration/research/native-learning-product/checkpoint-save-product.ts'))]))
    bodies.set(path, patchNativeCheckpointStorage(bodies.get(path) ?? await readFile(path, 'utf8'),
      join(root, 'backend/native-evolution-storage.ts'), replace));
}
