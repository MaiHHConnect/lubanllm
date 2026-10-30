/** A new read-only scoring view; original training/validation material is never overwritten. */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {readIdentityMaterial} from './run-native-identity-round.ts';
import type {NativeEvolutionMaterial} from '../backend/native-evolution-cohort.ts';

export function identityImportMaterial(original:NativeEvolutionMaterial):NativeEvolutionMaterial {
  assert.equal(original.samples.length,50,'Import must retain the candidate’s original training cohort');
  const result=structuredClone(original),rows=result.retentionRegressions!;
  assert.deepEqual(rows.map(row=>row.id),[0,1,2].map(n=>'q4-id-user-boundary-'+n));
  for(const row of rows){
    assert(row.training_allowed===false&&row.purpose==='validation');
    const p=typeof row.provenance==='string'?JSON.parse(row.provenance):row.provenance;
    const check=p.generation_check;
    assert(check.type==='exact'&&typeof check.expected==='string'&&check.expected.trim()===check.expected&&check.expected.length>0);
    p.generation_check_revision={policy:'identity-boundary-punctuation-v1',previous_check:structuredClone(check),training:false};
    const {expected,type:_type,...rest}=check;
    p.generation_check={...rest,type:'semantic',validator:'one_of',accepted:[expected,expected+'。']};
    row.provenance=typeof row.provenance==='string'?JSON.stringify(p):p;
  }
  return result;
}

if(import.meta.main){
  assert.equal(Bun.argv.length,4,'Usage: bun scripts/identity-import-material.ts ORIGINAL.json NEW.json');
  const material=identityImportMaterial(await readIdentityMaterial(resolve(Bun.argv[2]!)));
  await writeFile(resolve(Bun.argv[3]!),JSON.stringify(material,null,2)+'\n',{flag:'wx',mode:0o600});
  console.log(JSON.stringify({training_rows_unchanged:true,validation_rows_unchanged:true,readonly_boundary_checks_revised:3,model_loaded:false,training_updates:0}));
}
