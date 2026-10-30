/** Separate the installed-source receipt from the original five-field TRAIN metric. */
import {join} from 'node:path';
type Replace=(body:string,before:string,after:string)=>string;
export function patchNativeEvolutionConstraint(body:string,replace:Replace){
 return replace(body,'validateCore450ConstraintResidual(residual)',
  'validateCore450ConstraintResidual({quantized_frobenius:residual.quantized_frobenius,quantized_max_abs:residual.quantized_max_abs,quantized_relative_to_delta:residual.quantized_relative_to_delta,master_frobenius:residual.master_frobenius,weight_dtype:residual.weight_dtype})');
}
export function applyNativeEvolutionConstraintPatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(path:string)=>string;replace:Replace;root:string}){
 const path=target(join(root,'.native-migration/research/native-core450-owner-controller/runner.ts'));
 const body=bodies.get(path);if(!body)throw Error('Actual Core450 runner is missing');
 const start=body.indexOf('export async function runCore450ProductFromCheckpoint(');
 if(start<0)throw Error('Actual product checkpoint runner is missing');
 bodies.set(path,body.slice(0,start)+patchNativeEvolutionConstraint(body.slice(start),replace));
}
