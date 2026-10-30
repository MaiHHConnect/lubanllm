import {test,expect} from 'bun:test';
import {identityImportMaterial} from '../scripts/identity-import-material.ts';
import {readIdentityMaterial} from '../scripts/run-native-identity-round.ts';
import {evaluateGeneration} from '../backend/generation-checks.ts';

test('import scoring preserves original material and accepts only an optional Chinese sentence stop',async()=>{
  const original=await readIdentityMaterial('/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/authorized-identity-material.json');
  const before=JSON.stringify(original),revised=identityImportMaterial(original);
  expect(revised.samples).toEqual(original.samples);
  expect(revised.directedTeaching).toEqual(original.directedTeaching);
  for(const[index,row]of revised.retentionRegressions!.entries()){
    const old=original.retentionRegressions![index]!;
    const p=typeof row.provenance==='string'?JSON.parse(row.provenance):row.provenance;
    const previous=typeof old.provenance==='string'?JSON.parse(old.provenance):old.provenance;
    expect({...row,provenance:undefined}).toEqual({...old,provenance:undefined});
    expect(p.generation_check_revision.previous_check).toEqual(previous.generation_check);
    const expected=previous.generation_check.expected;
    for(const answer of [expected,expected+'。',' \n'+expected+'。\n'])expect(evaluateGeneration(answer,p.generation_check,true).passed).toBe(true);
    for(const answer of [expected+'。。','不是'+expected,expected+'和别人','不知道'+expected,'别的名字'])expect(evaluateGeneration(answer,p.generation_check,true).passed).toBe(false);
    expect(evaluateGeneration(expected,p.generation_check,false).passed).toBe(false);
    expect(row.training_allowed).toBe(false);
  }
  expect(JSON.stringify(original)).toBe(before);
  const bad=structuredClone(original);bad.retentionRegressions![0]!.training_allowed=true;
  expect(()=>identityImportMaterial(bad)).toThrow();
});
