/** Match Python's full TRAIN baseline and original-correct replay subset. */
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
type Replace=(body:string,before:string,after:string)=>string;
export function patchNativeEvolutionTrainingHistory(body:string,replace:Replace){
 return replace(body,
  'const baselineReplay=nativePartialCorrectIDs(r.baseline_training_replay_generation,groups.replay_training,false),replay=groups.replay_training.filter(row=>baselineReplay.has(row.id));',
  "const replayIDs=new Set(groups.replay_training.map(row=>row.id)),baselineReplay=nativePartialCorrectIDs(r.partial_training_baseline.filter((row:any)=>replayIDs.has(row.id)),groups.replay_training,false),replay=groups.replay_training.filter(row=>baselineReplay.has(row.id));const protectedReplay=nativePartialCorrectIDs(r.baseline_training_replay_generation,replay,false);assert.equal(protectedReplay.size,replay.length,'Original-correct replay must pass before training');");
}
export async function applyNativeEvolutionTrainingHistoryPatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(path:string)=>string;replace:Replace;root:string}){
 const path=target(join(root,'.native-migration/research/native-core450-runner/training-safety.ts'));
 bodies.set(path,patchNativeEvolutionTrainingHistory(bodies.get(path)??await readFile(path,'utf8'),replace));
}
