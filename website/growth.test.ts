import { expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { createGrowth, generateGrowthQuestions, GROWTH_POLICY, GROWTH_SKILLS } from "./growth";

const owner="a".repeat(64),other="b".repeat(64),base="c".repeat(64),beforeModel="d".repeat(64),afterModel="e".repeat(64);
function fixture(run:(db:Database,growth:ReturnType<typeof createGrowth>,clock:{value:number})=>void) {
  const db=new Database(":memory:"),clock={value:Date.now()};
  try {run(db,createGrowth(db,()=>clock.value),clock);}finally {db.close();}
}
function challenge(growth:ReturnType<typeof createGrowth>,skills:string[]=["arithmetic_add"],model=beforeModel) {
  return growth.challenge(owner,{base_identity:base,model_version:model,skills});
}
function sheet(db:Database,id:string,correct:(index:number)=>boolean=()=>true) {
  const row=db.query<{questions:string},[string]>("SELECT questions FROM growth_challenges WHERE id=?").get(id)!;
  return JSON.parse(row.questions).map((q:any,i:number)=>({id:q.id,text:correct(i)?q.scoring==="json"?JSON.stringify(q.answer):String(q.answer):"not the answer",finish_reason:"stop"}));
}
const submission=(id:string,model:string,answers:unknown)=>({challenge_id:id,model_version:model,answers});

test("program families generate unique layered public tasks with objectively verifiable gold",()=>{
  for(const skill of Object.keys(GROWTH_SKILLS)) {
    const questions=generateGrowthQuestions([skill as keyof typeof GROWTH_SKILLS]);
    expect(questions).toHaveLength(24);expect(new Set(questions.map(q=>q.prompt)).size).toBe(24);
    for(const q of questions) {
      if(skill.startsWith("arithmetic_")) {
        const [a,b]=q.prompt.match(/\d+/g)!.map(Number);
        expect(q.answer).toBe(skill==="arithmetic_add"?a+b:skill==="arithmetic_subtract"?a-b:a*b);
      } else if(skill==="uppercase") {
        expect(q.answer).toBe(JSON.parse(q.prompt.match(/"[a-z]+"/)![0]).toUpperCase());
      } else {
        const object=JSON.parse(q.prompt.slice("Given this JSON: ".length).split("\n")[0]);
        const path=q.prompt.match(/Extract the value at (.*?)\./)![1];
        const key=q.prompt.match(/(?:at |record\.|rows\[1\]\.)([a-z]{5})\. Output/)?.[1];
        const expected=path==="record"?object.record[key!]:path==="rows[1]"?object.rows[1][key!]:object[path];
        expect(q.answer).toEqual(expected);
      }
    }
  }
});
test("one server-selected domain and first model scope are fixed with no private original material",()=>fixture((db,growth,clock)=>{
  const request={base_identity:base,model_version:beforeModel,skills:["json_extract","uppercase","arithmetic_add"]};
  const c=growth.challenge(owner,request);
  expect(c.questions).toHaveLength(24);expect(c.policy).toEqual(GROWTH_POLICY);
  expect(c.policy.max_output_tokens).toBe(64);expect(growth.metadata.template_version).toBe("growth-programs-v2");
  expect(c.skills.every((skill:keyof typeof GROWTH_SKILLS)=>GROWTH_SKILLS[skill]===c.domain)).toBe(true);
  expect(c.skills).toEqual(Object.keys(GROWTH_SKILLS).filter(key=>GROWTH_SKILLS[key as keyof typeof GROWTH_SKILLS]===c.domain).sort());
  expect(c.questions.every((q:any)=>Object.keys(q).sort().join(",")==="id,prompt,skill")).toBe(true);
  expect(c.baseline).toBeNull();expect(growth.challenge(owner,request)).toEqual(c);
  expect(()=>growth.challenge(owner,{...request,base_identity:"f".repeat(64)})).toThrow("改写基座身份");
  expect(()=>growth.challenge(owner,{...request,skills:["arithmetic_multiply"]})).toThrow("不能反复挑选");
  for(const skills of [[],["bad"],["arithmetic_add","arithmetic_add"]]) expect(()=>growth.challenge(other,{...request,skills})).toThrow();
  expect(()=>growth.challenge(other,{...request,private_questions:["secret"]})).toThrow();
  clock.value+=3*3600000;const retry=growth.challenge(owner,request);
  expect(retry.challenge_id).toBe(c.challenge_id);expect(retry.questions).toEqual(c.questions);expect(new Date(retry.expires_at).getTime()).toBeGreaterThan(clock.value);
}));
test("legacy 512-token conditions preserve all rows and permit only a new frozen 64-token scope",()=>fixture((db,growth,clock)=>{
  const c=challenge(growth);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  const r=growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)));growth.publish(owner,r);
  const oldPolicy="3aab8b0a76713638278bebe7ce561aad14ed86b1b459477599086d6e8437ec65",oldTemplate="1".repeat(64);
  db.query("UPDATE growth_challenges SET policy_hash=?,template_sha256=?").run(oldPolicy,oldTemplate);
  db.query("UPDATE growth_receipts SET summary=? WHERE id=?").run(JSON.stringify({...r,policy_hash:oldPolicy,template_sha256:oldTemplate}),r.receipt_id);
  db.exec("DROP INDEX growth_receipt_version_scope");
  for(const table of ["growth_challenges","growth_receipts"]) {
    let schema=db.query<{sql:string},[string]>("SELECT sql FROM sqlite_master WHERE name=?").get(table)!.sql;
    schema=table==="growth_challenges"?schema.replace("UNIQUE(owner,before_model_version,policy_hash,template_sha256)","UNIQUE(owner,before_model_version)"):schema.replace(/\)\s*$/,", UNIQUE(owner,model_version,domain))");
    db.exec(`ALTER TABLE ${table} RENAME TO legacy_fixture; ${schema}; INSERT INTO ${table} SELECT * FROM legacy_fixture; DROP TABLE legacy_fixture`);
  }
  const rows=()=>["growth_challenges","growth_receipts","growth_retention","growth_publications"].map(table=>db.query(`SELECT * FROM ${table} ORDER BY 1`).all());
  const original=rows(),current=createGrowth(db,()=>clock.value);expect(rows()).toEqual(original);
  expect(current.profile(owner,afterModel).every(p=>p.growth===null)).toBe(true);
  expect(()=>current.validatePublication(owner,afterModel,r.receipt_id)).toThrow("条件不兼容");
  expect(()=>current.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)))).toThrow("条件已更新");
  const fresh=challenge(current);expect(fresh.challenge_id).not.toBe(c.challenge_id);expect(fresh.policy.max_output_tokens).toBe(64);
  current.before(owner,submission(fresh.challenge_id,beforeModel,sheet(db,fresh.challenge_id,i=>i<12)));
  const freshResult=current.after(owner,submission(fresh.challenge_id,afterModel,sheet(db,fresh.challenge_id)));
  expect(freshResult.policy_hash).toBe(current.metadata.policy_hash);expect(freshResult.receipt_id).not.toBe(r.receipt_id);
  expect(challenge(current).challenge_id).toBe(fresh.challenge_id);expect(db.query<{n:number},[]>("SELECT COUNT(*) AS n FROM growth_receipts").get()!.n).toBe(2);
  const prior=rows();createGrowth(db,()=>clock.value);expect(rows()).toEqual(prior);
}));
test("changed program or policy scope cannot reuse an old baseline or mix an old public profile",()=>fixture((db,growth)=>{
  const c=challenge(growth);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  db.query("UPDATE growth_challenges SET policy_hash=? WHERE id=?").run("0".repeat(64),c.challenge_id);
  expect(()=>growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)))).toThrow("程序或条件已更新");
  const valid=growth.challenge(other,{base_identity:base,model_version:beforeModel,skills:["uppercase"]});
  growth.before(other,submission(valid.challenge_id,beforeModel,sheet(db,valid.challenge_id,i=>i<12)));
  const r=growth.after(other,submission(valid.challenge_id,afterModel,sheet(db,valid.challenge_id)));
  growth.publish(other,r);expect(growth.profile(other,afterModel)[1].growth).not.toBeNull();
  db.query("UPDATE growth_receipts SET summary=? WHERE id=?").run(JSON.stringify({...r,template_sha256:"0".repeat(64)}),r.receipt_id);
  expect(growth.profile(other,afterModel)[1].growth).toBeNull();
  expect(()=>growth.validatePublication(other,afterModel,r.receipt_id)).toThrow("程序或条件不兼容");
}));
test("skills only discover domains and cannot choose an addition-only easier math suite",()=>fixture((db,growth)=>{
  const c=challenge(growth,["arithmetic_add"]);
  expect(c.skills).toEqual(["arithmetic_add","arithmetic_multiply","arithmetic_subtract"]);
  for(const skill of c.skills) expect(c.questions.filter((q:any)=>q.skill===skill)).toHaveLength(8);
  expect(challenge(growth,["arithmetic_subtract"])).toEqual(c);
  expect(challenge(growth,["arithmetic_add","arithmetic_multiply","arithmetic_subtract"])).toEqual(c);
}));
test("full baseline required, owner/version bound and exact answer retries cannot rewrite it",()=>fixture((db,growth)=>{
  const c=challenge(growth),answers=sheet(db,c.challenge_id,i=>i<12);
  expect(()=>growth.after(owner,submission(c.challenge_id,afterModel,answers))).toThrow("尚未完成");
  expect(()=>growth.before(other,submission(c.challenge_id,beforeModel,answers))).toThrow("不属于");
  expect(()=>growth.before(owner,submission(c.challenge_id,afterModel,answers))).toThrow("指纹");
  expect(()=>growth.before(owner,submission(c.challenge_id,beforeModel,answers.slice(1)))).toThrow("完整24题");
  expect(()=>growth.before(owner,submission(c.challenge_id,beforeModel,[answers[0],...answers.slice(0,-1)]))).toThrow("重题");
  for(const finish_reason of ["error",true,["stop"]]) expect(()=>growth.before(owner,submission(c.challenge_id,beforeModel,answers.map((a:any,i:number)=>i?a:{...a,finish_reason})))).toThrow("结束状态");
  expect(()=>growth.before(owner,{...submission(c.challenge_id,beforeModel,answers),baseline_score:50})).toThrow("请求无效");
  const result=growth.before(owner,submission(c.challenge_id,beforeModel,answers));expect(result.baseline_correct).toBe(12);expect(result.baseline_score).toBe(50);
  expect(growth.before(owner,submission(c.challenge_id,beforeModel,[...answers].reverse()))).toEqual(result);
  expect(()=>growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id)))).toThrow("已经冻结");
}));
test("objective before-after counts regressions, truncation and abstention without cumulative points",()=>fixture((db,growth)=>{
  const c=challenge(growth);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  const answers=sheet(db,c.challenge_id,i=>i<10||i>=12&&i<20);
  const receipt=growth.after(owner,submission(c.challenge_id,afterModel,answers));
  expect(receipt).toMatchObject({baseline_correct:12,current_correct:18,baseline_score:50,current_score:75,wrong_to_correct:8,correct_to_wrong:2,net_delta:25,gap_recovery:66.67,retention:83.33,eligible:true});
  expect(receipt).not.toHaveProperty("cumulative_points");
  expect(growth.after(owner,submission(c.challenge_id,afterModel,[...answers].reverse()))).toEqual(receipt);
  expect(()=>growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)))).toThrow("不可重复刷分");
  expect(()=>growth.after(owner,submission(c.challenge_id,"f".repeat(64),answers))).toThrow("不可重复刷分");
  const next=challenge(growth,["arithmetic_add"],afterModel);const baseline=sheet(db,next.challenge_id);
  baseline[0].finish_reason="length";baseline[1].finish_reason="abstain";
  expect(growth.before(owner,submission(next.challenge_id,afterModel,baseline)).baseline_correct).toBe(22);
  const current=sheet(db,next.challenge_id);current[2].text+=" explanation";
  expect(growth.after(owner,submission(next.challenge_id,"f".repeat(64),current)).current_correct).toBe(23);
}));
test("same version-domain cannot acquire another chosen receipt, and >14-day baseline cannot score",()=>fixture((db,growth,clock)=>{
  const c=challenge(growth);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)));
  const different=challenge(growth,["arithmetic_multiply"],"f".repeat(64));growth.before(owner,submission(different.challenge_id,"f".repeat(64),sheet(db,different.challenge_id,i=>i<12)));
  expect(()=>growth.after(owner,submission(different.challenge_id,afterModel,sheet(db,different.challenge_id)))).toThrow("同模型同领域");
  clock.value+=14*86400000+1;
  expect(()=>growth.after(owner,submission(different.challenge_id,"9".repeat(64),sheet(db,different.challenge_id)))).toThrow("超过14天");
}));
test("unknown domain remains null, low evidence stays personal and scores do not cross model scopes",()=>fixture((db,growth)=>{
  const c=challenge(growth,["uppercase"]);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<2)));
  const receipt=growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)));
  expect(receipt.eligible).toBe(false);expect(receipt.retention).toBe(100);
  expect(()=>growth.validatePublication(other,afterModel,receipt.receipt_id)).toThrow("不属于");
  expect(()=>growth.validatePublication(owner,beforeModel,receipt.receipt_id)).toThrow("指纹");
  expect(growth.profile(owner,afterModel).every(p=>p.growth===null)).toBe(true);
  growth.publish(owner,growth.validatePublication(owner,afterModel,receipt.receipt_id));
  const profile=growth.profile(owner,afterModel);expect(profile.map(p=>p.domain)).toEqual(["mathematics","programming","data"]);
  expect(profile[0].growth).toBeNull();expect(profile[1].growth?.current_score).toBe(100);expect(profile[2].growth).toBeNull();
  expect(growth.profile(owner,beforeModel).every(p=>p.growth===null)).toBe(true);
  expect(profile[1].growth).not.toHaveProperty("before_model_version");
  expect(profile[1].growth).not.toHaveProperty("questions");
}));
test("24-hour same-model retention is one immutable follow-up, grants no new growth and preserves types",()=>fixture((db,growth,clock)=>{
  const c=challenge(growth,["json_extract"]);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  const receipt=growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)));
  const body={receipt_id:receipt.receipt_id,model_version:afterModel,answers:sheet(db,c.challenge_id,i=>i<18)};
  expect(()=>growth.retention(owner,body)).toThrow("至少24小时");
  clock.value+=86400000;
  expect(()=>growth.retention(other,body)).toThrow("不属于");
  expect(()=>growth.retention(owner,{...body,model_version:beforeModel})).toThrow("同一模型");
  const result=growth.retention(owner,body);expect(result).toMatchObject({correct:18,retained_correct:18,retention:75});
  expect(growth.retention(owner,{...body,answers:[...body.answers].reverse()})).toEqual(result);
  expect(()=>growth.retention(owner,{...body,answers:sheet(db,c.challenge_id)})).toThrow("只允许一次");
  const unchanged=growth.validatePublication(owner,afterModel,receipt.receipt_id);expect(unchanged.net_delta).toBe(receipt.net_delta);expect(unchanged.retention_check?.retention).toBe(75);
  expect(growth.summary(unchanged).retention_check).not.toHaveProperty("model_version");
  const noBaseline=challenge(growth,["arithmetic_subtract"],"8".repeat(64));growth.before(owner,submission(noBaseline.challenge_id,"8".repeat(64),sheet(db,noBaseline.challenge_id,()=>false)));
  const zero=growth.after(owner,submission(noBaseline.challenge_id,"7".repeat(64),sheet(db,noBaseline.challenge_id,()=>false)));
  expect(zero.retention).toBeNull();expect(zero.net_delta).toBe(0);expect(zero.eligible).toBe(false);
}));
test("public growth history includes only published rounds and preserves older rounds without duplicate republishing",()=>fixture((db,growth,clock)=>{
  const first=challenge(growth);growth.before(owner,submission(first.challenge_id,beforeModel,sheet(db,first.challenge_id,i=>i<12)));
  const r=growth.after(owner,submission(first.challenge_id,afterModel,sheet(db,first.challenge_id)));
  expect(growth.history(owner)).toEqual([]);
  growth.publish(owner,r);const publishedAt=clock.value;
  clock.value+=1000;growth.publish(owner,r);
  expect(db.query<{published_at:number},[]>("SELECT published_at FROM growth_public_history").get()!.published_at).toBe(publishedAt);
  expect(growth.history(owner)).toEqual([growth.summary(r)]);expect(growth.history(other)).toEqual([]);
  const second=challenge(growth,["arithmetic_add"],afterModel);growth.before(owner,submission(second.challenge_id,afterModel,sheet(db,second.challenge_id,i=>i<12)));
  const next=growth.after(owner,submission(second.challenge_id,"f".repeat(64),sheet(db,second.challenge_id,i=>i<20)));
  expect(growth.history(owner)).toEqual([growth.summary(r)]);
  growth.publish(owner,next);
  expect(growth.history(owner)).toEqual([growth.summary(next),growth.summary(r)]);
  expect(growth.profile(owner,afterModel)[0].growth).toBeNull();
  createGrowth(db,()=>clock.value);expect(growth.history(owner)).toHaveLength(2);
  expect(()=>growth.publish(other,r)).toThrow("不属于");expect(growth.history(other)).toEqual([]);
}));
test("legacy history migration imports only the current public receipt with matching owner and domain",()=>fixture((db,growth,clock)=>{
  const first=challenge(growth);growth.before(owner,submission(first.challenge_id,beforeModel,sheet(db,first.challenge_id,i=>i<12)));
  const r=growth.after(owner,submission(first.challenge_id,afterModel,sheet(db,first.challenge_id)));growth.publish(owner,r);
  const second=challenge(growth,["arithmetic_add"],afterModel);growth.before(owner,submission(second.challenge_id,afterModel,sheet(db,second.challenge_id,i=>i<12)));
  growth.after(owner,submission(second.challenge_id,"f".repeat(64),sheet(db,second.challenge_id)));
  db.exec("DROP TABLE growth_public_history");
  db.query("INSERT INTO growth_publications VALUES (?,?,?)").run(other,r.domain,r.receipt_id);
  db.query("INSERT INTO growth_publications VALUES (?,?,?)").run(owner,"data",r.receipt_id);
  const migrated=createGrowth(db,()=>clock.value);
  expect(migrated.history(owner)).toEqual([migrated.summary(r)]);expect(migrated.history(other)).toEqual([]);
  const rows=db.query("SELECT * FROM growth_public_history").all();expect(rows).toHaveLength(1);
  createGrowth(db,()=>clock.value);expect(db.query("SELECT * FROM growth_public_history").all()).toEqual(rows);
}));
test("public growth history isolates current suite, policy and template versions",()=>fixture((db,growth)=>{
  const receipts=["suite_id","policy_hash","template_sha256",null].map((field,i)=>{
    const before=String(i+1).repeat(64),after=String(i+5).repeat(64),c=challenge(growth,["arithmetic_add"],before);
    growth.before(owner,submission(c.challenge_id,before,sheet(db,c.challenge_id,i=>i<12)));
    const r=growth.after(owner,submission(c.challenge_id,after,sheet(db,c.challenge_id)));growth.publish(owner,r);
    if(field) db.query("UPDATE growth_receipts SET summary=? WHERE id=?").run(JSON.stringify({...r,[field]:"old-version"}),r.receipt_id);
    return r;
  });
  expect(growth.history(owner)).toEqual([growth.summary(receipts[3]!)]);
  expect(db.query<{n:number},[]>("SELECT COUNT(*) AS n FROM growth_public_history").get()!.n).toBe(4);
}));
test("public growth history exposes count changes and limited retention statistics without any raw or private fields",()=>fixture((db,growth,clock)=>{
  const c=challenge(growth,["json_extract"]);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  const r=growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)));growth.publish(owner,r);
  clock.value+=86400000;
  growth.retention(owner,{receipt_id:r.receipt_id,model_version:afterModel,answers:sheet(db,c.challenge_id,i=>i<18)});
  const secret="PRIVATE_CHAT_QUESTION_ANSWER_CASE";
  db.query("UPDATE growth_receipts SET summary=? WHERE id=?").run(JSON.stringify({...r,owner,questions:[secret],answers:[secret],chat:secret,case:secret}),r.receipt_id);
  const retention=db.query<{summary:string},[string]>("SELECT summary FROM growth_retention WHERE receipt_id=?").get(r.receipt_id)!;
  db.query("UPDATE growth_retention SET summary=? WHERE receipt_id=?").run(JSON.stringify({...JSON.parse(retention.summary),owner,questions:[secret],answers:[secret],skills:r.skills}),r.receipt_id);
  const result=growth.history(owner);
  expect(result).toEqual([growth.summary(growth.validatePublication(owner,afterModel,r.receipt_id))]);
  expect(result[0]!.retention_check).toEqual({checked_at:new Date(clock.value).toISOString(),correct:18,retained_correct:18,retention:75});
  expect(Object.keys(result[0]!).sort()).toEqual(["domain","question_count","baseline_correct","current_correct","baseline_score","current_score","wrong_to_correct","correct_to_wrong","net_delta","gap_recovery","retention","eligible","completed_at","retention_check"].sort());
  const encoded=JSON.stringify(result);
  for(const privateValue of [secret,owner,base,beforeModel,afterModel,r.receipt_id,r.challenge_id,r.suite_sha256,r.policy_hash,r.template_sha256,...r.skills,...c.questions.map(q=>q.prompt)]) expect(encoded).not.toContain(privateValue);
}));
test("public growth history returns the twelve most recently completed published rounds regardless of publication order",()=>fixture((db,growth,clock)=>{
  const receipts=Array.from({length:13},(_,i)=>{
    const before=(100+i).toString(16).padStart(64,"0"),after=(200+i).toString(16).padStart(64,"0");
    const c=challenge(growth,["uppercase"],before);growth.before(owner,submission(c.challenge_id,before,sheet(db,c.challenge_id,i=>i<12)));
    const r=growth.after(owner,submission(c.challenge_id,after,sheet(db,c.challenge_id)));
    clock.value+=1000;return r;
  });
  for(const r of [...receipts].reverse()) {clock.value+=1000;growth.publish(owner,r);}
  expect(growth.history(owner)).toEqual(receipts.slice(1).reverse().map(growth.summary));
  const last=growth.history(owner);clock.value+=1000;growth.publish(owner,receipts[0]!);
  expect(growth.history(owner)).toEqual(last);
}));
test("publication and public history commit atomically",()=>fixture((db,growth)=>{
  const c=challenge(growth);growth.before(owner,submission(c.challenge_id,beforeModel,sheet(db,c.challenge_id,i=>i<12)));
  const r=growth.after(owner,submission(c.challenge_id,afterModel,sheet(db,c.challenge_id)));
  db.exec("CREATE TRIGGER reject_history BEFORE INSERT ON growth_public_history BEGIN SELECT RAISE(ABORT,'history failed'); END");
  expect(()=>growth.publish(owner,r)).toThrow("history failed");
  expect(growth.profile(owner,afterModel).every(p=>p.growth===null)).toBe(true);expect(growth.history(owner)).toEqual([]);
}));
