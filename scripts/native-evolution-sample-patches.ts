/** The ledger stores provenance JSON text; explicitly authorized cohorts carry the same JSON as an object. */
export function patchNativeEvolutionSampleFields(body:string,replace:(body:string,before:string,after:string)=>string){
 body=replace(body,'(sample.provenance === undefined || typeof sample.provenance === "string"), "学习材料格式无效"',
  '(sample.provenance === undefined || typeof sample.provenance === "string" || record(sample.provenance)), "学习材料格式无效"');
 return replace(body,'if (sample.provenance !== undefined) approved.provenance = sample.provenance;',
  'if (sample.provenance !== undefined) approved.provenance = typeof sample.provenance === "string" ? sample.provenance : encoded(sample.provenance);');
}
