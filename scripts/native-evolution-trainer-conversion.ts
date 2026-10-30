/** Reversible terminal-only overlay; frozen trainer math and original seven-state ABI remain intact. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {dirname,join,resolve} from 'node:path';
export function patchNativeEvolutionTrainerConversion(original:string,source:string,header:string){
 let body=original;const edits:{before:string;after:string}[]=[];
 const replace=(before:string,after:string)=>{assert.equal(body.split(before).length,2,before);body=body.replace(before,after);edits.push({before,after});};
 for(const match of [...body.matchAll(/^#include "([^"]+)"/gm)])replace(match[0],'#include '+JSON.stringify(resolve(dirname(source),match[1]!)));
 replace('#include <torch/torch.h>','#include <torch/torch.h>\n#include '+JSON.stringify(header));
 replace('||key=="rng_mode","Unexpected config");','||key=="rng_mode"||key=="final_conversion"||key=="constraint_old_positions","Unexpected config");');
 replace('  need(offset<=10000&&offset+steps<=10000,"Training offset invalid");',`  need(offset<=10000&&offset+steps<=10000,"Training offset invalid");
  const bool conversion=m.config.count("final_conversion");
  need(conversion==bool(m.config.count("constraint_old_positions"))&&conversion==bool(m.tensors.count("constraint_inputs")),"Incomplete terminal calibration binding");
  if(conversion){
    need(m.config.at("final_conversion")=="gptq-bf16-refine-v1"&&offset+steps<=450,"Unsupported terminal conversion or fixed450 range");
    const auto& spec=m.tensors.at("constraint_inputs");const auto old=integer(m.config.at("constraint_old_positions"));
    need(spec.dtype=="f32"&&spec.shape.size()==2&&spec.shape[1]==m.tensors.at("master").shape.at(1)&&old>0&&old<spec.shape[0]&&spec.shape[0]*spec.shape[1]<=10000000,"Invalid OLD/Q4 calibration matrix");
  }`);
 replace('    master.requires_grad_(true);int64_t stateStep=originalStep;',`    const bool finalConversion=m.config.count("final_conversion");
    const auto oldPositions=finalConversion?integer(m.config.at("constraint_old_positions")):0;
    Tensor conversionInputs;if(finalConversion&&originalStep+int64_t(m.steps.size())==450)conversionInputs=load(m,"constraint_inputs",torch::kCPU);
    std::string conversionReport="null";
    master.requires_grad_(true);int64_t stateStep=originalStep;`);
 replace('    for(const auto& [id,row]:m.rows){','    if(m.config.count("final_conversion"))allowed.insert("constraint_inputs");\n    for(const auto& [id,row]:m.rows){');
 const projection='      {torch::NoGradGuard noGrad;auto delta=master-reference;master.copy_(reference+delta-torch::matmul(torch::matmul(delta,basis),basis.t()));candidate=master.detach().to(candidate.scalar_type()).clone();}';
 replace(projection,projection+`
      if(finalConversion&&stateStep==450){
        luban_conversion::Result conversion;
        luban_conversion::applyAt450(master,candidate,reference,basis,conversionInputs,oldPositions,stateStep,step.index==int64_t(m.steps.size()),[&](){rng.checkCancelled();},&conversion);
        conversionReport=luban_conversion::reportJSON(conversion,oldPositions,conversionInputs.size(0));
      }`);
 replace('<<",\\"load_hash_transfer_seconds\\":"<<loadSeconds','<<",\\"terminal_conversion\\":"<<conversionReport<<",\\"load_hash_transfer_seconds\\":"<<loadSeconds');
 let inverse=body;for(const edit of edits.toReversed()){assert.equal(inverse.split(edit.after).length,2);inverse=inverse.replace(edit.after,edit.before);}assert.equal(inverse,original);
 return {body,edits,inverse_exact:true};
}
export async function writeNativeEvolutionTrainerConversion(root:string,output:string){
 const source=join(root,'.native-migration/research/native-mps-training/Training.v3.cpp'),header=join(root,'native/EvolutionConversion.h');
 const original=await readFile(source,'utf8'),patch=patchNativeEvolutionTrainerConversion(original,source,header),path=join(output,'EvolutionTraining.cpp');
 await writeFile(path,patch.body);return {source,header,original,path,...patch};
}
export function patchNativeEvolutionConversionSummary(body:string,replace:(body:string,before:string,after:string)=>string,helper:string){
 body=`import {validateNativeEvolutionTerminalConversion} from ${JSON.stringify(helper)};\n`+body;
 return replace(body,'      receiptSHA=digest(encoded({tensors:listing.sha256,report:reportFile.sha256,trace:traceFile.sha256},true));','      validateNativeEvolutionTerminalConversion(report!,input);\n      receiptSHA=digest(encoded({tensors:listing.sha256,report:reportFile.sha256,trace:traceFile.sha256},true));');
}
export async function applyNativeEvolutionConversionSummaryPatches(o:{root:string;bodies:Map<string,string>;target:(relative:string)=>string;replace:(body:string,before:string,after:string)=>string}){
 const path=o.target(join(o.root,'.native-migration/backend/native-training-output.ts')),body=o.bodies.get(path)??await readFile(path,'utf8');
 const helper=join(o.root,'backend/native-evolution-conversion.ts');
 o.bodies.set(path,patchNativeEvolutionConversionSummary(body,o.replace,helper));
 const statePath=o.target(join(o.root,'.native-migration/research/native-core450-runner/state.ts'));let state=o.bodies.get(statePath)??await readFile(statePath,'utf8');
 state=`import {validateNativeEvolutionTerminalConversion} from ${JSON.stringify(helper)};\n`+state;
 state=o.replace(state,' if(step===0){',' if(step===450&&output?.fixture.config.final_conversion)report.terminal_conversion=validateNativeEvolutionTerminalConversion(output.report!,output.fixture);\n if(step===0){');
 o.bodies.set(statePath,state);
}
