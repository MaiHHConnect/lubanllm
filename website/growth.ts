import { Database } from "bun:sqlite";
import { randomInt, randomBytes } from "node:crypto";
import { canonical, EvaluationError } from "./evaluation";

export const GROWTH_DOMAINS = ["mathematics", "programming", "data"] as const;
export type GrowthDomain = typeof GROWTH_DOMAINS[number];
export const GROWTH_SKILLS = { arithmetic_add: "mathematics", arithmetic_subtract: "mathematics", arithmetic_multiply: "mathematics", uppercase: "programming", json_extract: "data" } as const;
type Skill = keyof typeof GROWTH_SKILLS;
type Question = { id: string; skill: Skill; prompt: string; scoring: "number" | "text" | "json"; answer: unknown };
type Answer = { id: string; text: string; finish_reason: "stop" | "length" | "abstain" };
export const GROWTH_POLICY = { temperature: 0, seed: 42, max_output_tokens: 64, route: "local_neural", teacher_assistance: false, reference: false, thinking: "off", shots: 0,
  system_prompt: "Answer each task directly without references or assistance. For arithmetic tasks, output only the number. For uppercase tasks, output only the transformed text. For JSON extraction tasks, output only the requested JSON value. Do not explain your answer." };
const sha = (value: string) => new Bun.CryptoHasher("sha256").update(value).digest("hex");
const fingerprint = (value: unknown): value is string => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
const fields = (value: unknown, keys: string[]): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === keys.length && keys.every(k => Object.hasOwn(value,k));
const percent = (value: number, total: number) => total === 0 ? null : Math.round(10000 * value / total) / 100;
const fail = (message: string, status = 400): never => { throw new EvaluationError(message,status); };

// Public program families generate fresh parameters and contexts, never private materials.
export function generateGrowthQuestions(skills: Skill[]): Question[] {
  const word = (length: number) => Array.from({length},()=>String.fromCharCode(97+randomInt(26))).join("");
  const used = new Set<string>();
  return Array.from({length:24},(_,index) => {
    const skill=skills[index%skills.length], tier=Math.floor(index/skills.length)%3;
    let prompt="", answer:unknown, scoring:Question["scoring"];
    do {
      if (skill.startsWith("arithmetic_")) {
        const range=tier===0?20:tier===1?200:999;
        const a=randomInt(1,range+1),b=randomInt(1,range+1),op=skill==="arithmetic_add"?"+":skill==="arithmetic_subtract"?"-":"*";
        answer=op==="+"?a+b:op==="-"?a-b:a*b;scoring="number";
        prompt=tier===0?`Calculate ${a} ${op} ${b}. Output only the number.`:tier===1?`The values x=${a} and y=${b} are given. Compute x ${op} y. Output only the number.`:op==="+"?`A warehouse has ${a} items and receives ${b} more. How many items now? Output only the number.`:op==="-"?`A signed account balance is ${a} credits. A charge of ${b} credits is applied. What is the new signed balance? Output only the number.`:`There are ${a} boxes with ${b} items per box. How many items in total? Output only the number.`;
      } else if(skill==="uppercase") {
        const token=word(4+tier*3);answer=token.toUpperCase();scoring="text";
        prompt=tier===0?`Convert ${JSON.stringify(token)} to uppercase. Output only the transformed text.`:tier===1?`A code field contains ${JSON.stringify(token)}. Return the code after uppercasing every ASCII letter; output only the text.`:`Normalize the identifier ${JSON.stringify(token)} by replacing each a-z letter with A-Z. Output only the identifier.`;
      } else {
        scoring="json";const key=word(5);answer=tier===0?randomInt(1,1000):word(6);
        const object=tier===0?{[key]:answer,other:word(6)}:tier===1?{record:{[key]:answer,other:randomInt(100)},ignored:word(5)}:{rows:[{[key]:word(6)},{[key]:answer}]};
        const path=tier===0?key:tier===1?`record.${key}`:`rows[1].${key}`;
        prompt=`Given this JSON: ${JSON.stringify(object)}\nExtract the value at ${path}. Output only the extracted JSON value (including quotes for strings).`;
      }
    } while(used.has(prompt));
    used.add(prompt);return {id:`growth-${String(index+1).padStart(2,"0")}`,skill,prompt,scoring:scoring!,answer};
  });
}

type ChallengeRow = { id:string;owner:string;base_identity:string;before_model_version:string;scope_key:string;domain:GrowthDomain;skills:string;questions:string;suite_sha256:string;policy_hash:string;template_sha256:string;created_at:number;expires_at:number;before_hash:string|null;before_results:string|null;baseline_at:number|null; };
export type GrowthSummary = { domain:GrowthDomain; question_count:number; baseline_correct:number; current_correct:number; baseline_score:number; current_score:number; wrong_to_correct:number; correct_to_wrong:number; net_delta:number; gap_recovery:number|null; retention:number|null; eligible:boolean; completed_at:string; retention_check:Record<string,unknown>|null };
type Receipt = GrowthSummary & {receipt_id:string;challenge_id:string;suite_id:string;policy_hash:string;suite_sha256:string;template_sha256:string;base_identity:string;before_model_version:string;model_version:string;skills:Skill[];baseline_at:string};
type ReceiptRow = {id:string;owner:string;challenge_id:string;model_version:string;domain:GrowthDomain;after_hash:string;after_results:string;summary:string;created_at:number};

export function createGrowth(db: Database, now = () => Date.now()) {
  const metadata = {suite_id:"growth-v1",policy_hash:sha(canonical(GROWTH_POLICY)),supports:GROWTH_SKILLS,
    template_version:"growth-programs-v2",template_sha256:sha(canonical({generator:generateGrowthQuestions.toString(),grading:grade.toString(),domain_suites:GROWTH_SKILLS,scope:"domains-set-first-challenge",version:"growth-programs-v2",policy:GROWTH_POLICY})),question_count:24};
  const challengeSchema=`CREATE TABLE IF NOT EXISTS growth_challenges (
    id TEXT PRIMARY KEY,owner TEXT NOT NULL,base_identity TEXT NOT NULL,before_model_version TEXT NOT NULL,
    scope_key TEXT NOT NULL,domain TEXT NOT NULL,skills TEXT NOT NULL,questions TEXT NOT NULL,suite_sha256 TEXT NOT NULL,policy_hash TEXT NOT NULL,template_sha256 TEXT NOT NULL,
    created_at INTEGER NOT NULL,expires_at INTEGER NOT NULL,before_hash TEXT,before_results TEXT,baseline_at INTEGER,
    UNIQUE(owner,before_model_version,policy_hash,template_sha256)
  );`;
  const receiptSchema=`CREATE TABLE IF NOT EXISTS growth_receipts (
    id TEXT PRIMARY KEY,owner TEXT NOT NULL,challenge_id TEXT NOT NULL UNIQUE,model_version TEXT NOT NULL,
    domain TEXT NOT NULL,after_hash TEXT NOT NULL,after_results TEXT NOT NULL,summary TEXT NOT NULL,created_at INTEGER NOT NULL
  );`;
  db.exec(challengeSchema+receiptSchema+`
  CREATE TABLE IF NOT EXISTS growth_retention (
    receipt_id TEXT PRIMARY KEY,answer_hash TEXT NOT NULL,summary TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS growth_publications (
    owner TEXT NOT NULL,domain TEXT NOT NULL,receipt_id TEXT NOT NULL,PRIMARY KEY(owner,domain)
  );
  CREATE TABLE IF NOT EXISTS growth_public_history (
    owner TEXT NOT NULL,receipt_id TEXT NOT NULL,published_at INTEGER NOT NULL,PRIMARY KEY(owner,receipt_id)
  );`);
  // New conditions get their own frozen round; copy legacy rows without rewriting history.
  db.transaction(()=> {
    for(const [table,schema,legacyUnique] of [
      ["growth_challenges",challengeSchema,/UNIQUE\s*\(\s*owner\s*,\s*before_model_version\s*\)/i],
      ["growth_receipts",receiptSchema,/UNIQUE\s*\(\s*owner\s*,\s*model_version\s*,\s*domain\s*\)/i],
    ] as const) {
      const previous=db.query<{sql:string},[string]>("SELECT sql FROM sqlite_master WHERE type='table' AND name=?").get(table)!.sql;
      if(!legacyUnique.test(previous)) continue;
      db.exec(`ALTER TABLE ${table} RENAME TO _growth_policy_migration`);
      db.exec(schema);
      db.exec(`INSERT INTO ${table} SELECT * FROM _growth_policy_migration; DROP TABLE _growth_policy_migration`);
    }
    db.exec("CREATE UNIQUE INDEX IF NOT EXISTS growth_receipt_version_scope ON growth_receipts(owner,model_version,domain,json_extract(summary,'$.policy_hash'),json_extract(summary,'$.template_sha256'))");
    // Legacy publications have no publication timestamp; retain only their known public receipt.
    db.exec("INSERT OR IGNORE INTO growth_public_history SELECT p.owner,r.id,r.created_at FROM growth_publications p JOIN growth_receipts r ON r.id=p.receipt_id AND r.owner=p.owner AND r.domain=p.domain");
  }).immediate();
  function row(owner:string,id:unknown) {
    if(!fingerprint(id)) fail("成长挑战无效");
    const result=db.query<ChallengeRow,[string]>("SELECT * FROM growth_challenges WHERE id=?").get(id);
    if(!result||result.owner!==owner) fail("成长挑战不属于此安装",409);
    if(result.policy_hash!==metadata.policy_hash||result.template_sha256!==metadata.template_sha256) fail("成长评测程序或条件已更新，请使用新模型版本建立基线",409);
    return result;
  }
  function view(challenge:ChallengeRow) {
    return {challenge_id:challenge.id,suite_id:metadata.suite_id,domain:challenge.domain,skills:JSON.parse(challenge.skills),base_identity:challenge.base_identity,model_version:challenge.before_model_version,
      policy:GROWTH_POLICY,policy_hash:metadata.policy_hash,suite_sha256:challenge.suite_sha256,expires_at:new Date(challenge.baseline_at===null?challenge.expires_at:challenge.baseline_at+14*86400000).toISOString(),
      questions:(JSON.parse(challenge.questions) as Question[]).map(({id,skill,prompt})=>({id,skill,prompt})),baseline:challenge.before_results===null?null:baselineView(challenge)};
  }
  function baselineView(challenge:ChallengeRow) {
    const correct=(JSON.parse(challenge.before_results!) as boolean[]).filter(Boolean).length;
    return {challenge_id:challenge.id,domain:challenge.domain,question_count:24,baseline_correct:correct,baseline_score:percent(correct,24)!,before_model_version:challenge.before_model_version,baseline_at:new Date(challenge.baseline_at!).toISOString(),expires_at:new Date(challenge.baseline_at!+14*86400000).toISOString()};
  }
  const challenge = db.transaction((owner:string,body:unknown) => {
    if(!fields(body,["base_identity","model_version","skills"])||!fingerprint(body.base_identity)||!fingerprint(body.model_version)||!Array.isArray(body.skills)||!body.skills.length||body.skills.length>5||body.skills.some(skill=>typeof skill!=="string"||!Object.hasOwn(GROWTH_SKILLS,skill))||new Set(body.skills).size!==body.skills.length) fail("成长评测只接受基座、模型指纹及支持的技能方向");
    const skills=(body.skills as Skill[]).toSorted(),domains=[...new Set(skills.map(skill=>GROWTH_SKILLS[skill]))].toSorted(),scope=sha(canonical({domains}));
    const priorBase=db.query<{base_identity:string},[string,string]>("SELECT base_identity FROM growth_challenges WHERE owner=? AND before_model_version=? LIMIT 1").get(owner,body.model_version);
    if(priorBase&&priorBase.base_identity!==body.base_identity) fail("同一完整模型指纹不能改写基座身份",409);
    let existing=db.query<ChallengeRow,string[]>("SELECT * FROM growth_challenges WHERE owner=? AND before_model_version=? AND policy_hash=? AND template_sha256=?").get(owner,body.model_version,metadata.policy_hash,metadata.template_sha256);
    if(existing) {
      if(existing.base_identity!==body.base_identity) fail("同一完整模型指纹不能改写基座身份",409);
      if(existing.scope_key!==scope) fail("同一模型已固定成长领域，不能反复挑选测试",409);
      if(existing.before_hash===null&&existing.expires_at<=now()) db.query("UPDATE growth_challenges SET expires_at=? WHERE id=?").run(now()+2*3600000,existing.id);
      return view(row(owner,existing.id));
    }
    const domain=domains[randomInt(domains.length)];
    const selected=(Object.keys(GROWTH_SKILLS) as Skill[]).filter(skill=>GROWTH_SKILLS[skill]===domain).toSorted(),questions=generateGrowthQuestions(selected),time=now();
    const id=randomBytes(32).toString("hex"),suite_sha256=sha(canonical({suite_id:metadata.suite_id,template_sha256:metadata.template_sha256,domain,skills:selected,questions}));
    db.query("INSERT INTO growth_challenges(id,owner,base_identity,before_model_version,scope_key,domain,skills,questions,suite_sha256,policy_hash,template_sha256,created_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)").run(id,owner,body.base_identity,body.model_version,scope,domain,JSON.stringify(selected),JSON.stringify(questions),suite_sha256,metadata.policy_hash,metadata.template_sha256,time,time+2*3600000);
    return view(row(owner,id));
  });
  function grade(challenge:ChallengeRow,input:unknown) {
    if(!Array.isArray(input)||input.length!==24) fail("成长答卷必须包含完整24题，包括截断或放弃的题目");
    const questions=JSON.parse(challenge.questions) as Question[],ids=new Set(questions.map(q=>q.id));
    const answers=input.map(value=> {
      if(!fields(value,["id","text","finish_reason"])||typeof value.id!=="string"||!ids.delete(value.id)||typeof value.text!=="string"||new TextEncoder().encode(value.text).length>8192||typeof value.finish_reason!=="string"||!["stop","length","abstain"].includes(value.finish_reason)) fail("成长答卷有未知题、重题或非法结束状态");
      return value as Answer;
    });
    const byId=new Map(answers.map(a=>[a.id,a]));
    const results=questions.map(q=> {
      const answer=byId.get(q.id)!;if(answer.finish_reason!=="stop") return false;
      const text=answer.text.trim();
      if(q.scoring==="text") return text===q.answer;
      if(q.scoring==="number") return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)&&Number.isFinite(Number(text))&&Number(text)===q.answer;
      try{return canonical(JSON.parse(text))===canonical(q.answer);}catch{return false;}
    });
    return {results,answer_hash:sha(canonical(answers.toSorted((a,b)=>a.id.localeCompare(b.id))))};
  }
  const before = db.transaction((owner:string,body:unknown) => {
    if(!fields(body,["challenge_id","model_version","answers"])||!fingerprint(body.model_version)) fail("成长基线请求无效");
    const challenge=row(owner,body.challenge_id);
    if(challenge.before_model_version!==body.model_version) fail("基线模型指纹与挑战不一致",409);
    const graded=grade(challenge,body.answers);
    if(challenge.before_hash!==null) {
      if(challenge.before_hash!==graded.answer_hash) fail("成长基线已经冻结，不能改答卷",409);
      return baselineView(challenge);
    }
    if(challenge.expires_at<=now()) fail("成长基线挑战已过期",409);
    db.query("UPDATE growth_challenges SET before_hash=?,before_results=?,baseline_at=? WHERE id=?").run(graded.answer_hash,JSON.stringify(graded.results),now(),challenge.id);
    return baselineView(row(owner,challenge.id));
  });
  function receipt(owner:string,id:unknown) {
    if(!fingerprint(id)) fail("成长凭据无效");
    const record=db.query<ReceiptRow,[string]>("SELECT * FROM growth_receipts WHERE id=?").get(id);
    if(!record||record.owner!==owner) fail("成长凭据不属于此安装",409);
    const result=JSON.parse(record.summary) as Receipt;
    const check=db.query<{summary:string},[string]>("SELECT summary FROM growth_retention WHERE receipt_id=?").get(record.id);
    return {record,result:{...result,retention_check:check?JSON.parse(check.summary):null} as Receipt};
  }
  const after = db.transaction((owner:string,body:unknown) => {
    if(!fields(body,["challenge_id","model_version","answers"])||!fingerprint(body.model_version)) fail("成长测评请求无效");
    const challenge=row(owner,body.challenge_id);
    if(challenge.before_hash===null) fail("尚未完成进化前基线，不能上报成长",409);
    if(challenge.before_model_version===body.model_version) fail("成长评测需要不同的进化后模型指纹",409);
    const graded=grade(challenge,body.answers),existing=db.query<ReceiptRow,[string]>("SELECT * FROM growth_receipts WHERE challenge_id=?").get(challenge.id);
    if(existing) {
      if(existing.model_version!==body.model_version||existing.after_hash!==graded.answer_hash) fail("此轮成长已冻结，不可重复刷分",409);
      return receipt(owner,existing.id).result;
    }
    if(challenge.baseline_at!+14*86400000<now()) fail("成长基线已超过14天，请重新建立基线",409);
    if(db.query("SELECT id FROM growth_receipts WHERE owner=? AND model_version=? AND domain=? AND json_extract(summary,'$.policy_hash')=? AND json_extract(summary,'$.template_sha256')=?").get(owner,body.model_version,challenge.domain,metadata.policy_hash,metadata.template_sha256)) fail("同模型同领域已经完成成长评测，不能挑选最高轮次",409);
    const baseline=JSON.parse(challenge.before_results!) as boolean[],baseline_correct=baseline.filter(Boolean).length,current_correct=graded.results.filter(Boolean).length;
    const improved=baseline.filter((value,i)=>!value&&graded.results[i]).length,regressed=baseline.filter((value,i)=>value&&!graded.results[i]).length;
    const id=sha(`${challenge.id}:${body.model_version}:${graded.answer_hash}`),time=now();
    const result:Receipt={receipt_id:id,challenge_id:challenge.id,suite_id:metadata.suite_id,policy_hash:metadata.policy_hash,suite_sha256:challenge.suite_sha256,template_sha256:metadata.template_sha256,domain:challenge.domain,skills:JSON.parse(challenge.skills),base_identity:challenge.base_identity,before_model_version:challenge.before_model_version,model_version:body.model_version,question_count:24,baseline_correct,current_correct,baseline_score:percent(baseline_correct,24)!,current_score:percent(current_correct,24)!,wrong_to_correct:improved,correct_to_wrong:regressed,net_delta:percent(improved-regressed,24)!,gap_recovery:percent(improved,24-baseline_correct),retention:percent(baseline_correct-regressed,baseline_correct),eligible:baseline_correct>=4&&24-baseline_correct>=4,baseline_at:new Date(challenge.baseline_at!).toISOString(),completed_at:new Date(time).toISOString(),retention_check:null};
    db.query("INSERT INTO growth_receipts VALUES (?,?,?,?,?,?,?,?,?)").run(id,owner,challenge.id,body.model_version,challenge.domain,graded.answer_hash,JSON.stringify(graded.results),JSON.stringify(result),time);
    return result;
  });
  const retention = db.transaction((owner:string,body:unknown) => {
    if(!fields(body,["receipt_id","model_version","answers"])||!fingerprint(body.model_version)) fail("保持复查请求无效");
    const {record,result}=receipt(owner,body.receipt_id);
    if(record.model_version!==body.model_version) fail("保持复查必须使用此轮进化后同一模型指纹",409);
    const challenge=row(owner,record.challenge_id),graded=grade(challenge,body.answers);
    const existing=db.query<{answer_hash:string;summary:string},[string]>("SELECT * FROM growth_retention WHERE receipt_id=?").get(record.id);
    if(existing) {
      if(existing.answer_hash!==graded.answer_hash) fail("保持复查已完成，只允许一次不可变结果",409);
      return {receipt_id:record.id,...JSON.parse(existing.summary)};
    }
    if(now()<record.created_at+86400000) fail("保持复查须距完成成长评测至少24小时",409);
    const previous=JSON.parse(record.after_results) as boolean[],retained=previous.filter((value,i)=>value&&graded.results[i]).length;
    const check={model_version:record.model_version,checked_at:new Date(now()).toISOString(),correct:graded.results.filter(Boolean).length,retained_correct:retained,retention:percent(retained,previous.filter(Boolean).length)};
    db.query("INSERT INTO growth_retention VALUES (?,?,?)").run(record.id,graded.answer_hash,JSON.stringify(check));
    return {receipt_id:record.id,...check};
  });
  function validatePublication(owner:string,model:unknown,id:unknown) {
    if(!fingerprint(model)) fail("发布成长需要完整本地模型指纹");
    const value=receipt(owner,id).result;if(value.model_version!==model) fail("成长凭据与当前完整模型指纹不匹配",409);
    if(value.policy_hash!==metadata.policy_hash||value.template_sha256!==metadata.template_sha256) fail("成长评测程序或条件不兼容",409);
    return value;
  }
  const publish = db.transaction((owner:string,value:Receipt) => {
    const {record}=receipt(owner,value.receipt_id);
    db.query("INSERT INTO growth_publications VALUES (?,?,?) ON CONFLICT(owner,domain) DO UPDATE SET receipt_id=excluded.receipt_id").run(owner,record.domain,record.id);
    db.query("INSERT OR IGNORE INTO growth_public_history VALUES (?,?,?)").run(owner,record.id,now());
  });
  function summary(value:Receipt):GrowthSummary {
    const {domain,question_count,baseline_correct,current_correct,baseline_score,current_score,wrong_to_correct,correct_to_wrong,net_delta,gap_recovery,retention,eligible,completed_at,retention_check}=value;
    const check=retention_check?{checked_at:retention_check.checked_at,correct:retention_check.correct,retained_correct:retention_check.retained_correct,retention:retention_check.retention}:null;
    return {domain,question_count,baseline_correct,current_correct,baseline_score,current_score,wrong_to_correct,correct_to_wrong,net_delta,gap_recovery,retention,eligible,completed_at,retention_check:check};
  }
  function profile(owner:string,model:string|null) {
    return GROWTH_DOMAINS.map(domain=> {
      const published=model?db.query<{id:string},string[]>("SELECT r.id FROM growth_publications p JOIN growth_receipts r ON r.id=p.receipt_id WHERE p.owner=? AND p.domain=? AND r.model_version=? AND json_extract(r.summary,'$.policy_hash')=? AND json_extract(r.summary,'$.template_sha256')=?").get(owner,domain,model,metadata.policy_hash,metadata.template_sha256):null;
      return {domain,growth:published?summary(receipt(owner,published.id).result):null};
    });
  }
  function history(owner:string):GrowthSummary[] {
    return db.query<{id:string},string[]>("SELECT r.id FROM growth_public_history h JOIN growth_receipts r ON r.id=h.receipt_id AND r.owner=h.owner WHERE h.owner=? AND json_extract(r.summary,'$.suite_id')=? AND json_extract(r.summary,'$.policy_hash')=? AND json_extract(r.summary,'$.template_sha256')=? ORDER BY r.created_at DESC,r.id DESC LIMIT 12").all(owner,metadata.suite_id,metadata.policy_hash,metadata.template_sha256).map(r=>summary(receipt(owner,r.id).result));
  }
  return {metadata,challenge:(owner:string,body:unknown)=>challenge.immediate(owner,body),before:(owner:string,body:unknown)=>before.immediate(owner,body),after:(owner:string,body:unknown)=>after.immediate(owner,body),retention:(owner:string,body:unknown)=>retention.immediate(owner,body),validatePublication,publish:(owner:string,value:Receipt)=>publish.immediate(owner,value),profile,history,summary};
}
