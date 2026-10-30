/** Static conversion inputs use the existing fixture/segment/resume ownership. */
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
type Context={bodies:Map<string,string>;target:(p:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export function patchConstraintFixtureParser(body:string,replace:Context['replace'],helper:string){
 body='import {validateConstraintFixture} from '+JSON.stringify(helper)+';\n'+body;
 body=replace(body,'"training_offset","learning_rate","distillation_weight",','"training_offset","learning_rate","distillation_weight",...(version===3?["final_conversion","constraint_old_positions"]:[]),');
 body=replace(body,'const names=[...FINAL,"norm","head",','const names=[...FINAL,"norm","head",...(config.final_conversion?["constraint_inputs"]:[]),');
 body=replace(body,'const shape=tensors.candidate!.shape,[hidden,input]=r.core_shape;','validateConstraintFixture(config,tensors,r.core_shape[1]);\n  const shape=tensors.candidate!.shape,[hidden,input]=r.core_shape;');
 return body;
}
export async function applyNativeEvolutionCalibrationPatches({bodies,target,replace,root}:Context){
 const helper=join(root,'backend/native-evolution-constraint-inputs.ts'),calibration=join(root,'backend/native-evolution-calibration.ts');
 const edit=async(path:string,change:(body:string)=>string)=>{path=target(join(root,'.native-migration',path));bodies.set(path,change(bodies.get(path)??await readFile(path,'utf8')));};
 await edit('research/native-evolution-producer/producer.ts',body=>{
  body='import {prepareNativeEvolutionCalibration} from '+JSON.stringify(calibration)+';\n'+body;
  body=replace(body,'const id=randomBytes(6).toString("hex"),version=', 'const conversion=await prepareNativeEvolutionCalibration(r.engine,guard,cohort,control,r.owners,r.work,r.signal);\n  const id=randomBytes(6).toString("hex"),version=');
  body=replace(body,'const graphBinding={fixture_only:true,','const graphBinding={conversion:conversion.metadata,fixture_only:true,');
  body=replace(body,'={...state,norm,head,cpu_rng:', '={...state,norm,head,constraint_inputs:conversion.tensor,cpu_rng:');
  body=replace(body,'expected,state,weights:{norm,head},caches:captures,','expected,state,constraint:{input:conversion.tensor,old_positions:conversion.metadata.old_positions},weights:{norm,head},caches:captures,');
  return body;
 });
 await edit('research/native-evolution-fixture/exporter.ts',body=>{
  body='import {validateConstraintFixture,FINAL_CONVERSION} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,' steps?:number;start_step?:number;learning_rate?:number;distillation_weight?:number;', ' constraint?:{input:NativeFixtureRaw;old_positions:number};\n steps?:number;start_step?:number;learning_rate?:number;distillation_weight?:number;');
  body=replace(body,'const state=Object.fromEntries(', 'const constraint=request.constraint?{input:snapshotInput(request.constraint.input) as NativeFixtureRaw,old_positions:request.constraint.old_positions}:null;\n  const state=Object.fromEntries(');
  body=replace(body,'sources.cpu_rng=rng.cpu;', 'if(constraint)sources.constraint_inputs=constraint.input;\n  sources.cpu_rng=rng.cpu;');
  body=replace(body,'shapes.cpu_rng={dtype:"u8",shape:[rng.cpu.bytes]};','if(constraint){validateConstraintFixture({final_conversion:FINAL_CONVERSION,constraint_old_positions:String(constraint.old_positions)},{constraint_inputs:constraint.input},width);shapes.constraint_inputs={dtype:"f32",shape:[...constraint.input.shape]};}\n  shapes.cpu_rng={dtype:"u8",shape:[rng.cpu.bytes]};');
  body=replace(body,'const config={identity,device,rng_mode:rng.mode,','const config={...(constraint?{final_conversion:FINAL_CONVERSION,constraint_old_positions:String(constraint.old_positions)}:{}),identity,device,rng_mode:rng.mode,');
  return body;
 });
 for(const path of ['research/native-evolution-fixture/parser-readback.ts','backend/native-training-output.ts'])await edit(path,body=>patchConstraintFixtureParser(body,replace,helper));
 await edit('research/native-core450-owner-controller/bootstrap.ts',body=>{
  body='import {preparedOldBaseline} from '+JSON.stringify(helper)+';\n'+body;
  const old='trainingBaseline=await generateCore450Rows(control,trainingRecords,fresh,r.signal),coreProtection=await generateCore450Rows(control,records.protection,fresh,r.signal);';
  const value='coreProtection=preparedOldBaseline(r.preparation.graphBinding.conversion,records.protection),generatedTraining=await generateCore450Rows(control,trainingRecords.filter((row:any)=>!records.protection.some((old:any)=>old.id===row.id)),fresh,r.signal),trainingBaseline:Record<string,any>[]=trainingRecords.map((row:any)=>{const value=[...generatedTraining,...coreProtection].find(v=>v.id===row.id);assert(value,"Bound original baseline row missing");return value;});assert(equal(coreProtection.map(row=>row.id),records.protection.map((row:any)=>row.id)),"Prepared original baseline IDs differ");';
  const start=body.indexOf("export async function createCore450ProductFreshCheckpoint");if(start<0)throw Error("Product bootstrap missing");
  return body.slice(0,start)+replace(body.slice(start),old,value);
 });
}
