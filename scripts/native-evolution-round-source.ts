/** Native round/RNG commands use the existing worker and do not alter graph mathematics. */
import {join} from 'node:path';
import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
export function patchEvolutionRoundSource(root:string,change:(before:string,after:string)=>void){
 change('    durable(root/"parent_rng.raw",parent.data_ptr(),parent.nbytes());durable(root/"restored_rng.raw",restored.data_ptr(),restored.nbytes());durable(root/"parent_mps_rng.raw",parentMPS.data_ptr(),parentMPS.nbytes());durable(root/"restored_mps_rng.raw",restoredMPS.data_ptr(),restoredMPS.nbytes());syncDir(root);\n','');
 change('#include '+JSON.stringify(join(root,'native/EvolutionTail.h')),'#include '+JSON.stringify(join(root,'native/EvolutionTail.h'))+'\n#include '+JSON.stringify(join(root,'native/EvolutionRound.h')));
 change('bool observePrevious=false;std::string currentCandidateSHA;','bool observePrevious=false;std::string currentCandidateSHA;EvolutionRoundRNG evolutionRound;');
 change('need(!failed||op=="core-observation"','need(!failed||op=="evolution-end"||op=="core-observation"');
 change('if(op=="core-observation"){',`if(op=="evolution-begin"){need(!(in>>extra)&&!currentCore,"Evolution begins from accepted source");evolutionRound.begin();}
   else if(op=="evolution-end"){need(!(in>>extra)&&!currentCore,"Evolution RNG restoration requires accepted source");evolutionRound.restore();failed=false;}
   else if(op=="evolution-rng"){in>>name;need(bool(in)&&!(in>>extra),"Exact checkpoint RNG output required");validName();evolutionRound.checkpoint(root/name);}
   else if(op=="core-observation"){`);
 change('currentCore&&backbone?jsonString(observePrevious?(backbone->initial.config.at("core_version")=="null"?ORIGINAL:backbone->initial.config.at("core_tensor_sha256")):currentCandidateSHA):"null"','backbone?jsonString(currentCore&&!observePrevious?currentCandidateSHA:(backbone->initial.config.at("core_version")=="null"?ORIGINAL:backbone->initial.config.at("core_tensor_sha256"))):"null"');
}

export async function writeEvolutionCoreStateSource(root:string,output:string){
 const source=join(root,'.native-migration/research/native-unified-core-peer/CoreStateV2.h'),original=await readFile(source,'utf8');let body=original;const edits:{before:string;after:string}[]=[];
 const change=(before:string,after:string)=>{assert.equal(body.split(before).length,2,before);edits.push({before,after});body=body.replace(before,after);};
 change('void residual(const fs::path&path){','void residual(const fs::path&path,bool ordinary=false){');
 change('  check();torch::NoGradGuard noGrad;','  if(ordinary)source->check();else check();torch::NoGradGuard noGrad;');
 change('  snapshot(path);const std::array','  if(ordinary)fresh(path);else snapshot(path);const std::array');
 change('syncDir(path);check();need(captureNativeCPURNG()', 'syncDir(path);if(ordinary)source->check();else check();need(captureNativeCPURNG()');
 const path=join(output,'EvolutionCoreState.h');await writeFile(path,body);return{path,record:{source,path,original,body,edits}};
}
