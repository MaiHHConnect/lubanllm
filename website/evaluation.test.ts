import { expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { canonical, createEvaluations } from "./evaluation";

const policy = {temperature:0,seed:42,max_output_tokens:512,route:"local_core",teacher_assistance:false,thinking:"off",shots:0,system_prompt:"Fixed synthetic test policy."};
const questions = Array.from({length:48},(_,i)=>({id:`q${i}`,kind:i<24?"benchmark":"generalization",prompt:`Isolated task ${i}`,scoring:i<24?"choice":i<36?"exact_number":"json",answer:i<24?"B":i<36?10:{a:1,b:[2,3]}}));
const owner="a".repeat(64), model="b".repeat(64);
function fixture(run:(db:Database,path:string)=>void) {
  const root=mkdtempSync(join(tmpdir(),"luban-ranking-evaluation-isolated-"));const path=join(root,"suite.json");
  writeFileSync(path,JSON.stringify({suite_id:"luban-ranking-v1",policy,questions}));
  const db=new Database(":memory:");
  try {run(db,path);}finally {db.close();rmSync(root,{recursive:true,force:true});}
}
const answers=()=>questions.map(q=>({id:q.id,text:typeof q.answer==="object"?JSON.stringify(q.answer):String(q.answer),finish_reason:"stop"}));
test("objective graders preserve exact JSON types and full-answer constraints",()=>fixture((db,path)=>{
  const evaluation=createEvaluations(db,path);const challenge=evaluation.challenge(owner,{model_version:model}) as any;
  const sheet=answers();sheet[0].text="b";sheet[1].text="The answer is B";sheet[2].finish_reason="length";
  sheet[24].text="10.0";sheet[25].text="1e1";sheet[36].text='{"b":[2,3],"a":1}';sheet[37].text='{"a":true,"b":[2,3]}';sheet[38].text='{"a":1,"b":[2,3],"extra":0}';
  const receipt=evaluation.submit(owner,{challenge_id:challenge.challenge_id,model_version:model,answers:sheet});
  expect(receipt.benchmark).toBe(91.67);expect(receipt.generalization).toBe(87.5);
  expect(canonical({b:[3,2],a:"中文"})).toBe('{"a":"中文","b":[3,2]}');
}));
test("a changed suite byte fingerprint invalidates old challenge and public receipt binding",()=>fixture((db,path)=>{
  const first=createEvaluations(db,path);const c=first.challenge(owner,{model_version:model}) as any;
  const receipt=first.submit(owner,{challenge_id:c.challenge_id,model_version:model,answers:answers()});
  expect(first.validatePublication(owner,model,receipt.receipt_id).benchmark).toBe(100);
  writeFileSync(path,JSON.stringify({suite_id:"luban-ranking-v1",policy,questions})+"\n");
  const changed=createEvaluations(db,path);
  expect(changed.metadata.suite_sha256).not.toBe(first.metadata.suite_sha256);
  expect(()=>changed.validatePublication(owner,model,receipt.receipt_id)).toThrow("评测凭据不属于");
  expect(()=>changed.submit(owner,{challenge_id:c.challenge_id,model_version:model,answers:answers()})).toThrow("评测题库已更新");
  expect((changed.challenge(owner,{model_version:model}) as any).receipt).toBeUndefined();
}));
test("same-version concurrent nonces cannot replace a receipt and unrelated policies do not make deltas",()=>fixture((db,path)=>{
  const first=createEvaluations(db,path);const c=first.challenge(owner,{model_version:model}) as any;
  db.query("UPDATE evaluation_challenges SET expires_at=0 WHERE id=?").run(c.challenge_id);
  const next=first.challenge(owner,{model_version:model}) as any;
  expect(next.challenge_id).not.toBe(c.challenge_id);
  db.query("UPDATE evaluation_challenges SET expires_at=? WHERE id=?").run(Date.now()+10000,c.challenge_id);
  first.submit(owner,{challenge_id:next.challenge_id,model_version:model,answers:answers()});
  expect(()=>first.submit(owner,{challenge_id:c.challenge_id,model_version:model,answers:answers()})).toThrow("不能重复刷分");
  writeFileSync(path,JSON.stringify({suite_id:"luban-ranking-v1",policy:{...policy,system_prompt:"Other fixed prompt."},questions}));
  const changed=createEvaluations(db,path);const different=changed.challenge(owner,{model_version:"c".repeat(64)}) as any;
  const receipt=changed.submit(owner,{challenge_id:different.challenge_id,model_version:"c".repeat(64),answers:answers()});
  expect(receipt.benchmark_delta).toBeNull();expect(receipt.generalization_delta).toBeNull();
}));
