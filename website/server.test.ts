import { afterAll, beforeAll, expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { maskName } from "./name-filter";
import { readFileSync, statSync } from "node:fs";
import { displayEntries, seedDisplayEntries } from "./seed-display";

// Existing Chinese UI regressions use an explicit browser language.
const fetch:typeof globalThis.fetch=(input,init)=>{const headers=new Headers(init?.headers);if(!headers.has("Accept-Language"))headers.set("Accept-Language","zh-CN");return globalThis.fetch(input,{...init,headers})};

const temporary = mkdtempSync(join(tmpdir(), "luban-ranking-v2-test-"));
const tokenA = "a".repeat(64), tokenB = "b".repeat(64), tokenC = "c".repeat(64);
let server: Awaited<typeof import("./server")>["server"];
let base: string;
const proof = (number: number) => ({
  base_sha256: "d".repeat(64), weights_sha256: number.toString(16).padStart(64, "0"),
  baseline_validation_loss: 2, validation_loss: 1, replay_before: 1, replay_loss: 1,
  training_samples: 5, validation_samples: 2, completed_at: 1700000000 + number,
});
beforeAll(async () => {
  process.env.PORT = "0";
  process.env.LEADERBOARD_DB = join(temporary, "test.sqlite");
  // A v1 database is preserved; old editable scores must never enter v2 ranking.
  const legacy = new Database(process.env.LEADERBOARD_DB);
  legacy.exec("CREATE TABLE entries (owner TEXT PRIMARY KEY,username TEXT NOT NULL,name_key TEXT NOT NULL UNIQUE,evolution_count INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)");
  legacy.query("INSERT INTO entries VALUES (?,?,?,?,?,?)").run("legacy", "EditableScore", "editablescore", 999999, "old", "old");
  legacy.close();
  server = (await import("./server")).server;
  base = `http://127.0.0.1:${server.port}`;
});
afterAll(() => { server.stop(true); rmSync(temporary, { recursive: true, force: true }); });
const request = (token: string, body: unknown, origin?: string) => fetch(`${base}/api/leaderboard/me`, {
  method: "PUT", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json", ...(origin ? { Origin: origin } : {}) }, body: JSON.stringify(body),
});
const publish = (token: string, username: unknown, upgrades: unknown = [], origin?: string) => request(token, { username, upgrades }, origin);
const list = async () => (await fetch(`${base}/api/leaderboard?sort=evolution_count`)).json();

test("enterprise server entry exposes only a backend and authenticated voluntary accounting", async () => {
  const entry = await fetch(`${base}/qy`);
  expect(await entry.json()).toEqual({ service: "enterprise_authorization", ok: true });
  expect(entry.headers.get("cache-control")).toBe("no-store");
  expect(entry.headers.get("x-robots-tag")).toContain("noindex");
  expect(await (await fetch(`${base}/qy/`, { method: "HEAD" })).text()).toBe("");
  expect((await fetch(`${base}/qy.js`)).status).toBe(404);
  expect((await fetch(`${base}/api/enterprise/admin/companies`)).status).toBe(401);
  const keyPath = join(temporary, "enterprise-admin-key");
  expect(statSync(keyPath).mode & 0o777).toBe(0o600);
  const login = await fetch(`${base}/api/enterprise/admin/login`, { method: "POST", headers: { "Content-Type": "application/json", Origin: "https://ai.maihh.net" }, body: JSON.stringify({ key: readFileSync(keyPath, "utf8").trim() }) });
  expect(login.status).toBe(200);
  const cookie = login.headers.get("set-cookie")!.split(";")[0];
  const headers = { "Content-Type": "application/json", Origin: "https://ai.maihh.net", Cookie: cookie };
  const create = await fetch(`${base}/api/enterprise/admin/companies`, { method: "POST", headers, body: JSON.stringify({ request_id: crypto.randomUUID(), name: "接口隔离测试企业", annual_tokens: 1, billing_policy: "inference_input_output_v1" }) });
  expect(create.status).toBe(201);
  const company = await create.json();
  const activate = await fetch(`${base}/api/enterprise/activate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ license_code: company.license_code, device_id: crypto.randomUUID(), device_name: "隔离设备" }) });
  expect(activate.status).toBe(201);
  const device = await activate.json();
  const used = await fetch(`${base}/api/enterprise/usage`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${device.device_token}` }, body: JSON.stringify({ request_id: crypto.randomUUID(), used_tokens: 99 }) });
  expect(used.status).toBe(200);
  expect((await used.json()).company.overage_tokens).toBe(98);
  const followups = await fetch(`${base}/api/enterprise/admin/followups`, { headers: { Cookie: cookie } });
  expect(followups.status).toBe(200);
  const data = await followups.json();
  expect(data.companies[0].followup_reasons).toEqual(["quota_overage"]);
  expect(JSON.stringify(data)).not.toContain(device.device_token);
});

test("quarantines old manual scores and announces the new counting protocol", async () => {
  const response = await list();
  expect(response.total).toBe(0);
  expect(response.entries).toEqual([]);
  expect(response.protocol_version).toBe(2);
});
test("protects publication identity and origin", async () => {
  expect((await publish("bad", "A")).status).toBe(401);
  expect((await publish(tokenA, "A", [], "https://attacker.example")).status).toBe(403);
});
test("rejects hand-entered counts even alongside valid upgrade evidence", async () => {
  for (const evolution_count of [0, 7, true, 99999999]) {
    expect((await request(tokenA, { username: "Alice", evolution_count })).status).toBe(400);
    expect((await request(tokenA, { username: "Alice", evolution_count, upgrades: [proof(1)] })).status).toBe(400);
  }
  expect((await list()).total).toBe(0);
});
test("rejects malformed, non-improving, missing-validation, replay-damaging or future proofs", async () => {
  for (const username of ["", "a\u0000", "a\u200b", "a".repeat(33)]) expect((await publish(tokenA, username)).status).toBe(400);
  for (const patch of [{ weights_sha256: "fake" }, { validation_loss: 1.99 }, { validation_loss: -1 },
    { baseline_validation_loss: 0 }, { training_samples: true }, { validation_samples: 0 },
    { replay_loss: 1.07 }, { replay_before: null }, { completed_at: Date.now() / 1000 + 1000 },
    { evolution_count: 999 }, { approved: true }]) {
    expect((await publish(tokenA, "Alice", [{ ...proof(1), ...patch }])).status).toBe(400);
  }
  expect((await publish(tokenA, "Alice", [null])).status).toBe(400);
  expect((await publish(tokenA, "Alice", Array.from({length:51},(_,i)=>proof(i+1)))).status).toBe(400);
  expect((await list()).total).toBe(0);
});
test("counts upgrades on the server, normalizes usernames and deduplicates within a batch", async () => {
  const response = await publish(tokenA, "  Ａｌｉｃｅ  ", [proof(1), proof(1)]);
  expect(response.status).toBe(200);
  expect((await response.json()).entry).toMatchObject({ username: "Alice", country_code: null, evolution_count: 1, rank: 1 });
  expect((await publish(tokenB, "alice", [proof(2)])).status).toBe(409);
  expect((await list()).total).toBe(1);
});
test("retries and empty/subset reports do not increase or reset accepted totals", async () => {
  for (const upgrades of [[proof(1)], [], [proof(1),proof(3)], [proof(1),proof(3)]]) {
    expect((await publish(tokenA, "Alice", upgrades)).status).toBe(200);
  }
  expect((await list()).entries[0].evolution_count).toBe(2);
});
test("rejects copying another installation's upgrade or modifying an accepted proof atomically", async () => {
  expect((await publish(tokenC, "Copy", [proof(1)])).status).toBe(409);
  expect((await list()).total).toBe(1);
  expect((await publish(tokenA, "Renamed", [proof(4),{...proof(1),validation_loss:0.9}])).status).toBe(409);
  const data = await list();
  expect(data.entries[0].username).toBe("Alice");
  expect(data.entries[0].evolution_count).toBe(2);
});
test("supports validated legacy metrics and descending ranking", async () => {
  const historical = {...proof(2),replay_before:null,replay_loss:null};
  expect((await publish(tokenB, "Bob", [historical,proof(5),proof(6)])).status).toBe(200);
  const data = await list();
  expect(data.entries.map((entry: any)=>[entry.username,entry.evolution_count,entry.rank])).toEqual([["Bob",3,1],["Alice",2,2]]);
});
test("withdrawal keeps receipt ownership; rejoining and copying cannot replay an upgrade", async () => {
  for (let i=0;i<2;i++) {
    const response=await fetch(`${base}/api/leaderboard/me`,{method:"DELETE",headers:{Authorization:`Bearer ${tokenB}`}});
    expect(await response.json()).toEqual({ok:true});
  }
  expect((await list()).total).toBe(1);
  expect((await publish(tokenC,"Copy",[{...proof(2),replay_before:null,replay_loss:null}])).status).toBe(409);
  expect((await publish(tokenB,"Bob",[])).status).toBe(200);
  expect((await list()).entries[0].evolution_count).toBe(3);
});
test("public responses escape usernames and contain neither proofs nor ownership", async () => {
  expect((await publish(tokenB,'<b>x</b>',[])).status).toBe(200);
  const response=await fetch(`${base}/?sort=evolution_count`);
  const html=await response.text();
  expect(html).toContain('&lt;b&gt;x&lt;/b&gt;');
  expect(html).not.toContain('<b>x</b>');
  expect(html).not.toContain('自主登记');
  expect(html).not.toContain('自报');
  expect(html).not.toContain(tokenA);
  expect(response.headers.get("content-security-policy")).toContain("script-src 'none'");
  expect(Object.keys((await list()).entries[0]).sort()).toEqual(["benchmark","benchmark_delta","country_code","evaluated_at","evolution_count","generalization","generalization_delta","growth","growth_history","growth_profile","model_version","rank","suite_id","token_total","username"]);
});
test("persists server-counted receipts without storing credentials",()=>{
  const read=new Database(process.env.LEADERBOARD_DB!,{readonly:true});
  const rows=read.query<{owner:string},[]>("SELECT owner FROM upgrades").all();
  expect(rows.length).toBe(5);
  expect(rows.every(row=>row.owner!==tokenA&&row.owner!==tokenB&&/^[0-9a-f]{64}$/.test(row.owner))).toBe(true);
  read.close();
});
test("limits request bodies and keeps crawler/static routes restricted",async()=>{
  expect((await fetch(`${base}/?page=-1`)).status).toBe(400);
  expect((await fetch(`${base}/api/leaderboard?page=bad`)).status).toBe(400);
  expect((await (await fetch(`${base}/api/leaderboard?sort=evolution_count&page=2`)).json()).entries).toEqual([]);
  expect(await (await fetch(`${base}/robots.txt`)).text()).toContain("Disallow: /api/");
  expect(await (await fetch(`${base}/sitemap.xml`)).text()).toContain("https://ai.maihh.net/");
  expect((await fetch(`${base}/assets/luban-zero-logo.png`)).status).toBe(200);
  for(const path of ["/server.ts","/evaluation.ts","/ranking-v1.json","/data/leaderboard.sqlite","/admin"]) expect((await fetch(`${base}${path}`)).status).toBe(404);
  expect((await fetch(`${base}/api/leaderboard/me`,{method:"PUT",headers:{Authorization:`Bearer ${tokenA}`,"Content-Type":"application/json"},body:"x".repeat(66000)})).status).toBe(413);
});

test("country choices persist, legacy app reports preserve country and invalid codes are rejected",async()=>{
  expect((await request(tokenA,{username:"Alice",country_code:"CN",upgrades:[]})).status).toBe(200);
  expect((await publish(tokenA,"Alice",[])).status).toBe(200);
  expect((await list()).entries.find((e:Entry)=>e.username==="Alice").country_code).toBe("CN");
  expect(await (await fetch(`${base}/?sort=evolution_count`)).text()).toContain("🇨🇳 中国");
  for(const country_code of ["ZZ","CN<script>","cn",true,{},123]) expect((await request(tokenA,{username:"Alice",country_code,upgrades:[]})).status).toBe(400);
  expect((await request(tokenA,{username:"Alice",country_code:null,upgrades:[]})).status).toBe(200);
  expect((await list()).entries.find((e:Entry)=>e.username==="Alice").country_code).toBeNull();
});
test("username limits use Unicode codepoints and mask common comment words before persistence",async()=>{
  const masked=await request(tokenA,{username:"习大大",country_code:"CN",upgrades:[]});
  expect(masked.status).toBe(200);expect((await masked.json()).entry.username).toBe("***");
  expect((await request(tokenA,{username:"最强玩家",country_code:"CN",upgrades:[]})).status).toBe(200);
  expect((await list()).entries.find((e:Entry)=>e.evolution_count===2).username).toBe("***玩家");
  for(const username of ["a".repeat(17),"😀".repeat(17)]) expect((await publish(tokenA,username,[])).status).toBe(400);
  expect((await publish(tokenA,"😀".repeat(16),[])).status).toBe(200);
  expect((await publish(tokenA,"Alice",[])).status).toBe(200);
  for(const [name,expected] of [["习大大","***"],["习-大-大","***"],["最强玩家","***玩家"],["傻-逼用户","***用户"],["ＦＵＣＫ","***"],["İ最强","İ***"],["x".repeat(24),"***"]]) expect(maskName(name)).toBe(expected);
  expect(readFileSync(`${import.meta.dir}/name_words.json`,`utf8`)).toBe(readFileSync(`${import.meta.dir}/../src/luban/name_words.json`,`utf8`));
});
test("only returns the requested 100-entry page and exposes paging after entry100",async()=>{
  const read=new Database(process.env.LEADERBOARD_DB!);
  const insert=read.query("INSERT INTO entries(owner,username,name_key,evolution_count,created_at,updated_at,visible,country_code) VALUES (?,?,?,0,?,?,1,?)");
  for(let i=0;i<201;i++) insert.run(`paging-${i}`,`Player${i}`,`player${i}`,`2026-01-01T00:00:${String(i).padStart(3,"0")}`,"now",i===100?"JP":null);
  const first=await list();expect(first.entries.length).toBe(100);expect(first.total).toBe(203);expect(first.page_size).toBe(100);
  const second=await (await fetch(`${base}/api/leaderboard?sort=evolution_count&page=2`)).json();expect(second.entries.length).toBe(100);expect(second.entries[0].rank).toBe(101);
  const third=await (await fetch(`${base}/api/leaderboard?sort=evolution_count&page=3`)).json();expect(third.entries.length).toBe(3);expect(third.entries[0].rank).toBe(201);
  const html=await (await fetch(`${base}/?sort=evolution_count&page=2`)).text();expect(html).not.toContain("1 / 3");expect(html).toContain("2 / 3");expect(html).toContain("🇯🇵 日本");
  expect(html).toContain('href="/?page=1&amp;sort=evolution_count&amp;domain=mathematics&amp;lang=zh"');expect(html).toContain('href="/?page=3&amp;sort=evolution_count&amp;domain=mathematics&amp;lang=zh"');
  read.query("DELETE FROM entries WHERE owner LIKE 'paging-%'").run();read.close();
});
test("167 display users merge into paginated ranking without modifying real receipts", async () => {
  const db = new Database(process.env.LEADERBOARD_DB!);
  const before = db.query("SELECT * FROM upgrades ORDER BY id").all();
  try {
    expect(seedDisplayEntries(db)).toBe(167);
    expect(seedDisplayEntries(db)).toBe(0);
    expect(displayEntries.filter(e => e.country_code === "US").length).toBe(101);
    expect(new Set(displayEntries.map(e => e.country_code)).size).toBe(21);
    const first = await list();
    const second = await (await fetch(`${base}/api/leaderboard?sort=evolution_count&page=2`)).json();
    expect(first.total).toBe(169);
    expect(first.entries.length).toBe(100);
    expect(second.entries.length).toBe(69);
    const all = [...first.entries, ...second.entries];
    expect(all.map(e => e.rank)).toEqual(Array.from({length:169}, (_,i) => i+1));
    expect(all.every((e,i) => !i || e.evolution_count <= all[i-1].evolution_count)).toBe(true);
    expect(new Set(all.map(e => e.username)).size).toBe(169);
    expect(all.find(e => e.username === "Alice").evolution_count).toBe(2);
    const reply = await (await publish(tokenA,"Alice",[])).json();
    expect(reply.entry.rank).toBe(all.find(e => e.username === "Alice").rank);
    expect(db.query("SELECT * FROM upgrades ORDER BY id").all()).toEqual(before);
    const html = await (await fetch(`${base}/?sort=evolution_count`)).text();
    expect(html).toContain("169<span> 位参与者");
    expect(html).toContain("1 / 2");
    expect(html).not.toContain("模拟");
    expect(html).not.toContain("display:");
    for (const path of ["/seed-display.ts", "/display_entries"]) expect((await fetch(`${base}${path}`)).status).toBe(404);
  } finally { db.query("DELETE FROM display_entries").run(); db.close(); }
});

test("display metrics populate supported domains and sorting without creating evaluation receipts", async () => {
  const db=new Database(process.env.LEADERBOARD_DB!);
  const tables=["entries","upgrades","evaluation_receipts","growth_challenges","growth_receipts","growth_publications","growth_retention"];
  const before=tables.map(name=>db.query(`SELECT * FROM ${name} ORDER BY rowid`).all());
  try {
    seedDisplayEntries(db);
    const seeded=db.query("SELECT * FROM display_entries ORDER BY id").all();
    expect(seedDisplayEntries(db)).toBe(0);
    expect(db.query("SELECT * FROM display_entries ORDER BY id").all()).toEqual(seeded);
    for(const domain of ["mathematics","programming","data"] as const) {
      const expected=displayEntries.filter(e=>e.growth_profile.find(p=>p.domain===domain)?.growth!==null);
      const first=await (await fetch(`${base}/api/leaderboard?domain=${domain}`)).json();
      const second=await (await fetch(`${base}/api/leaderboard?domain=${domain}&page=2`)).json();
      const all=[...first.entries,...second.entries];
      expect(first.total).toBe(expected.length);expect(first.entries.length).toBe(100);expect(all.length).toBe(expected.length);
      expect(all.map(e=>e.rank)).toEqual(Array.from({length:expected.length},(_,i)=>i+1));
      expect(all.every((e,i)=>e.growth.domain===domain&&e.growth.eligible&&e.token_total>0&&(!i||e.growth.net_delta<=all[i-1].growth.net_delta))).toBe(true);
      expect(all.some(e=>e.growth.net_delta<0)).toBe(true);expect(all.some(e=>e.growth.retention<100)).toBe(true);
      expect(all.every(e=>e.growth_profile.length===3&&!("display_profile" in e)&&!("owner" in e))).toBe(true);
    }
    for(const sort of ["benchmark","generalization"] as const) {
      const first=await (await fetch(`${base}/api/leaderboard?sort=${sort}`)).json();
      const second=await (await fetch(`${base}/api/leaderboard?sort=${sort}&page=2`)).json();
      const all=[...first.entries,...second.entries];
      expect(all.length).toBe(displayEntries.filter(e=>e[sort]!==null).length);
      expect(all.every((e,i)=>e[sort]!==null&&(!i||e[sort]<=all[i-1][sort]))).toBe(true);
    }
    expect((await (await fetch(`${base}/api/leaderboard?domain=medicine`)).json()).entries).toEqual([]);
    expect(tables.map(name=>db.query(`SELECT * FROM ${name} ORDER BY rowid`).all())).toEqual(before);
  } finally {db.query("DELETE FROM display_entries").run();db.close();}
});

test("real usernames take priority over display handles without inheriting their counts", async () => {
  const db = new Database(process.env.LEADERBOARD_DB!);
  try {
    seedDisplayEntries(db);
    const display = displayEntries[0];
    const reply = await (await publish(tokenC,display.username,[])).json();
    expect(reply.entry.evolution_count).toBe(0);
    const all = [...(await list()).entries,...(await (await fetch(`${base}/api/leaderboard?sort=evolution_count&page=2`)).json()).entries];
    expect(all.filter(e => e.username === display.username).map(e => e.evolution_count)).toEqual([0]);
    const real=all.find(e=>e.username===display.username);
    expect(real.token_total).toBeNull(); expect(real.benchmark).toBeNull(); expect(real.growth).toBeNull();
  } finally {
    await fetch(`${base}/api/leaderboard/me`,{method:"DELETE",headers:{Authorization:`Bearer ${tokenC}`}});
    db.query("DELETE FROM display_entries").run(); db.close();
  }
});
type Entry={username:string;country_code:string|null;evolution_count:number;rank:number};

const tokenD = "4".repeat(64), tokenE = "5".repeat(64), tokenF = "6".repeat(64);
const model1 = "1".repeat(64), model2 = "2".repeat(64);
const evaluateRequest = (token: string, path: string, body: unknown, origin?: string) => fetch(`${base}/api/leaderboard/evaluation${path}`, {
  method:"POST",headers:{Authorization:`Bearer ${token}`,"Content-Type":"application/json",...(origin?{Origin:origin}:{})},body:JSON.stringify(body),
});
const challenge = async (token: string, model_version = model1) => (await evaluateRequest(token,"/challenge",{model_version})).json();
const suite = () => JSON.parse(readFileSync(`${import.meta.dir}/../benchmarks/ranking-v1.json`,"utf8"));
const answerSheet = () => suite().questions.map((q:any) => ({id:q.id,text:typeof q.answer === "object" ? JSON.stringify(q.answer) : String(q.answer),finish_reason:"stop"}));
let challengeD:any, receiptD:any, receiptE:any;
test("evaluation challenges require identity, omit gold and freeze the complete fixed policy", async()=>{
  expect((await evaluateRequest("bad","/challenge",{model_version:model1})).status).toBe(401);
  expect((await evaluateRequest(tokenD,"/challenge",{model_version:model1},"https://evil.example")).status).toBe(403);
  for(const body of [{model_version:"fake"},{model_version:model1,benchmark:100}]) expect((await evaluateRequest(tokenD,"/challenge",body)).status).toBe(400);
  challengeD = await challenge(tokenD);
  expect(challengeD.questions.length).toBe(48);
  expect(challengeD.questions.filter((q:any)=>q.kind==="benchmark").length).toBe(24);
  expect(challengeD.questions.every((q:any)=>Object.keys(q).sort().join(",")==="id,kind,prompt")).toBe(true);
  expect(challengeD.policy).toMatchObject({temperature:0,seed:42,max_output_tokens:512,route:"local_core",teacher_assistance:false,thinking:"off",shots:0});
  expect(challengeD.policy_hash).toBe(suite().policy_hash);
  expect(challengeD.suite_sha256).toBe(new Bun.CryptoHasher("sha256").update(readFileSync(`${import.meta.dir}/../benchmarks/ranking-v1.json`)).digest("hex"));
  expect(new Date(challengeD.expires_at).getTime()-Date.now()).toBeGreaterThan(30*60*1000);
  expect((await challenge(tokenD)).challenge_id).toBe(challengeD.challenge_id);
});
test("server rejects forged scores, incomplete sheets, duplicate tasks and stolen challenges",async()=>{
  const valid={challenge_id:challengeD.challenge_id,model_version:model1,answers:answerSheet()};
  expect((await evaluateRequest(tokenE,"",valid)).status).toBe(409);
  expect((await evaluateRequest(tokenD,"",{...valid,model_version:model2})).status).toBe(409);
  expect((await evaluateRequest(tokenD,"",{...valid,benchmark:100})).status).toBe(400);
  expect((await evaluateRequest(tokenD,"",{...valid,answers:valid.answers.slice(1)})).status).toBe(400);
  expect((await evaluateRequest(tokenD,"",{...valid,answers:[valid.answers[0],...valid.answers.slice(0,-1)]})).status).toBe(400);
  expect((await evaluateRequest(tokenD,"",{...valid,answers:valid.answers.map((a:any,i:number)=>i? a:{...a,score:1})})).status).toBe(400);
  expect((await evaluateRequest(tokenD,"",{...valid,answers:valid.answers.map((a:any,i:number)=>i? a:{...a,text:true})})).status).toBe(400);
});
test("objective server scoring counts truncation as wrong and immutable retries cannot improve scores",async()=>{
  const answers=answerSheet();
  const benchmark = suite().questions.filter((q:any)=>q.kind==="benchmark");
  const generalization = suite().questions.filter((q:any)=>q.kind==="generalization");
  for(const a of answers) {
    if(a.id===benchmark[0].id) a.finish_reason="length";
    if([generalization[0].id,generalization[1].id].includes(a.id)) a.text="not an answer";
  }
  const body={challenge_id:challengeD.challenge_id,model_version:model1,answers};
  const result=await evaluateRequest(tokenD,"",body);expect(result.status).toBe(200);receiptD=await result.json();
  expect(receiptD).toMatchObject({benchmark:95.83,generalization:91.67,benchmark_delta:null,generalization_delta:null,model_version:model1,suite_id:"luban-ranking-v1"});
  expect(await (await evaluateRequest(tokenD,"",{...body,answers:[...answers].reverse()})).json()).toEqual(receiptD);
  expect((await evaluateRequest(tokenD,"",{...body,answers:answerSheet()})).status).toBe(409);
  expect((await challenge(tokenD)).receipt).toEqual(receiptD);
  const defaultList=await (await fetch(`${base}/api/leaderboard?sort=benchmark`)).json();expect(defaultList.sort).toBe("benchmark");expect(defaultList.entries).toEqual([]);
});
test("publishing requires an owner/version-bound server receipt and preserves cumulative Token maxima",async()=>{
  const body={username:"测评甲",upgrades:[],model_version:model1,evaluation_receipt_id:receiptD.receipt_id,token_total:1000};
  expect((await request(tokenE,body)).status).toBe(409);
  expect((await request(tokenD,{...body,model_version:model2})).status).toBe(409);
  expect((await request(tokenD,{...body,benchmark:999})).status).toBe(400);
  for(const token_total of [-1,true,"1000",1.5,1e15+1]) expect((await request(tokenD,{...body,token_total})).status).toBe(400);
  for(const token_total of [1000,1000,500,null]) {
    const reply=await (await request(tokenD,{...body,token_total})).json();expect(reply.entry.token_total).toBe(1000);expect(reply.entry.benchmark).toBe(95.83);
  }
  const {token_total,...withoutTokens}=body;
  expect((await (await request(tokenD,withoutTokens)).json()).entry.token_total).toBe(1000);
  expect((await (await fetch(`${base}/api/leaderboard?sort=benchmark`)).json()).entries[0]).toMatchObject({username:"测评甲",benchmark:95.83,generalization:91.67,token_total:1000,evolution_count:0});
  expect((await (await request(tokenD,{...body,token_total:1500})).json()).entry.token_total).toBe(1500);
});
test("benchmark and generalization exclude unmeasured display rows",async()=>{
  const c=await challenge(tokenE);const answers=answerSheet().map((a:any)=>suite().questions.find((q:any)=>q.id===a.id).kind==="generalization"?{...a,text:"wrong"}:a);
  receiptE=await (await evaluateRequest(tokenE,"",{challenge_id:c.challenge_id,model_version:model1,answers})).json();
  expect((await request(tokenE,{username:"测评乙",upgrades:[],model_version:model1,evaluation_receipt_id:receiptE.receipt_id})).status).toBe(200);
  const db=new Database(process.env.LEADERBOARD_DB!);
  try {
    seedDisplayEntries(db);
    db.query("UPDATE display_entries SET benchmark=NULL,generalization=NULL,token_total=NULL").run();
    const benchmark=await (await fetch(`${base}/api/leaderboard?sort=benchmark`)).json();
    expect(benchmark.total).toBe(2);expect(benchmark.entries.map((e:any)=>e.username)).toEqual(["测评乙","测评甲"]);
    const general=await (await fetch(`${base}/api/leaderboard?sort=generalization`)).json();
    expect(general.sort).toBe("generalization");expect(general.entries.map((e:any)=>e.username)).toEqual(["测评甲","测评乙"]);
    const counted=[...(await list()).entries,...(await (await fetch(`${base}/api/leaderboard?sort=evolution_count&page=2`)).json()).entries];
    const displayNames=new Set(displayEntries.map(e=>e.username));
    expect(counted.filter((e:any)=>displayNames.has(e.username)).every((e:any)=>e.benchmark===null&&e.generalization===null&&e.token_total===null)).toBe(true);
    const html=await (await fetch(`${base}/?sort=benchmark`)).text();
    expect(html).toContain('href="/?sort=benchmark&amp;domain=mathematics&amp;lang=zh" aria-current="page"');
    expect(html.indexOf("累计 Token")).toBeLessThan(html.indexOf("进化次数</th>"));
    expect(html.indexOf("进化次数</th>")).toBeLessThan(html.indexOf("本次成长</th>"));
    expect(html.indexOf("本次成长</th>")).toBeLessThan(html.indexOf("旧能力保持</th>"));
    expect(html).toContain("MMLU-Pro 24题子集");
    expect((await fetch(`${base}/api/leaderboard?sort=token_total`)).status).toBe(400);
  } finally {db.query("DELETE FROM display_entries").run();db.close();}
});
test("deltas compare preceding different versions only, allow regressions and old clients clear stale scores",async()=>{
  const c=await challenge(tokenD,model2);const questions=suite().questions.filter((q:any)=>q.kind==="generalization");
  const wrong=new Set(questions.slice(12).map((q:any)=>q.id));const answers=answerSheet().map((a:any)=>wrong.has(a.id)?{...a,text:"wrong"}:a);
  const r=await (await evaluateRequest(tokenD,"",{challenge_id:c.challenge_id,model_version:model2,answers})).json();
  expect(r).toMatchObject({benchmark:100,generalization:50,benchmark_delta:4.17,generalization_delta:-41.67});
  expect((await request(tokenD,{username:"测评甲",upgrades:[],model_version:model2,evaluation_receipt_id:r.receipt_id})).status).toBe(200);
  expect((await (await fetch(`${base}/api/leaderboard?sort=benchmark`)).json()).entries.find((e:any)=>e.username==="测评甲").benchmark_delta).toBe(4.17);
  const old=await (await publish(tokenD,"测评甲")).json();expect(old.entry.benchmark).toBeNull();expect(old.entry.token_total).toBe(1500);
  expect((await (await fetch(`${base}/api/leaderboard?sort=benchmark`)).json()).total).toBe(1);
  expect((await request(tokenD,{username:"测评甲",upgrades:[],model_version:model2,evaluation_receipt_id:r.receipt_id})).status).toBe(200);
});
test("expired challenges cannot be submitted and a fresh nonce does not alter immutable receipts",async()=>{
  const c=await challenge(tokenF);
  const db=new Database(process.env.LEADERBOARD_DB!);db.query("UPDATE evaluation_challenges SET expires_at=0 WHERE id=?").run(c.challenge_id);db.close();
  expect((await evaluateRequest(tokenF,"",{challenge_id:c.challenge_id,model_version:model1,answers:answerSheet()})).status).toBe(409);
  expect((await challenge(tokenF)).challenge_id).not.toBe(c.challenge_id);
});
test("core upgrade counts use the same immutable hash identity and validated larger sample limit",async()=>{
  const core={...proof(7000),kind:"core",training_samples:34,validation_samples:8};
  const body={username:"测评丙",upgrades:[core]};
  expect((await request(tokenF,body)).status).toBe(200);
  expect((await (await request(tokenF,body)).json()).entry.evolution_count).toBe(1);
  for(const patch of [{kind:"unknown"},{kind:"branch"},{training_samples:10001},{validation_samples:true},{validation_loss:1.999}]) expect((await request(tokenF,{...body,upgrades:[{...core,...patch}]})).status).toBe(400);
  const {kind,...branch}=core;
  expect((await request(tokenF,{...body,upgrades:[branch]})).status).toBe(400);
  expect((await request(tokenF,{...body,upgrades:[{...core,training_samples:33}]})).status).toBe(409);
  expect((await request(tokenE,{username:"复制",upgrades:[core]})).status).toBe(409);
  expect((await list()).count_scope).toBe("accepted_neural_upgrade_records");
});

const growthRequest=(token:string,path:string,body:unknown)=>fetch(`${base}/api/leaderboard/growth/${path}`,{method:"POST",headers:{Authorization:`Bearer ${token}`,"Content-Type":"application/json"},body:JSON.stringify(body)});
const growthSheet=(id:string,count:number)=>{
  const db=new Database(process.env.LEADERBOARD_DB!);const row=db.query<{questions:string},[string]>("SELECT questions FROM growth_challenges WHERE id=?").get(id)!;db.close();
  return JSON.parse(row.questions).map((q:any,i:number)=>({id:q.id,text:i<count?(q.scoring==="json"?JSON.stringify(q.answer):String(q.answer)):"wrong",finish_reason:"stop"}));
};
let personalChallenge:any,personalReceipt:any;
const personalBefore="7".repeat(64),personalAfter="8".repeat(64);
test("growth HTTP default selects mathematics and rejects private/self-scored challenge bodies",async()=>{
  const current=await (await fetch(`${base}/api/leaderboard`)).json();expect(current).toMatchObject({sort:"growth",domain:"mathematics",total:0});
  expect((await fetch(`${base}/api/leaderboard?sort=growth&domain=biology`)).status).toBe(400);
  const body={base_identity:"c".repeat(64),model_version:personalBefore,skills:["arithmetic_add"]};
  expect((await growthRequest("bad","challenge",body)).status).toBe(401);
  expect((await growthRequest(tokenD,"challenge",{...body,questions:["private example"]})).status).toBe(400);
  expect((await growthRequest(tokenD,"challenge",{...body,skills:["arbitrary_private_task"]})).status).toBe(400);
  const response=await growthRequest(tokenD,"challenge",body);expect(response.status).toBe(200);personalChallenge=await response.json();
  expect(personalChallenge.domain).toBe("mathematics");expect(personalChallenge.questions).toHaveLength(24);expect(JSON.stringify(personalChallenge.questions)).not.toContain('"answer"');
  expect(personalChallenge.policy_hash).toBe("ee4277bcffe85771e6c5d8e30ecd77d5fd841b136ae8529d0f082f3d300e3337");
  expect(personalChallenge.policy.max_output_tokens).toBe(64);
  expect((await growthRequest(tokenD,"after",{challenge_id:personalChallenge.challenge_id,model_version:personalAfter,answers:growthSheet(personalChallenge.challenge_id,18)})).status).toBe(409);
});
test("growth public publication uses the full-model receipt and preserves only matching core benchmark",async()=>{
  const before={challenge_id:personalChallenge.challenge_id,model_version:personalBefore,answers:growthSheet(personalChallenge.challenge_id,12)};
  expect((await growthRequest(tokenD,"before",before)).status).toBe(200);
  const after={...before,model_version:personalAfter,answers:growthSheet(personalChallenge.challenge_id,18)};
  expect((await growthRequest(tokenE,"after",after)).status).toBe(409);
  expect((await growthRequest(tokenD,"after",{...after,net_delta:100})).status).toBe(400);
  personalReceipt=await (await growthRequest(tokenD,"after",after)).json();expect(personalReceipt).toMatchObject({baseline_score:50,current_score:75,net_delta:25,retention:100,eligible:true});
  const body={username:"测评甲",upgrades:[],model_version:model2,growth_model_version:personalAfter,growth_receipt_id:personalReceipt.receipt_id};
  expect((await request(tokenE,body)).status).toBe(409);expect((await request(tokenD,{...body,growth_model_version:personalBefore})).status).toBe(409);
  const published=await (await request(tokenD,body)).json();expect(published.entry.benchmark).toBe(100);expect(published.entry.growth).toMatchObject({net_delta:25,retention:100});
  expect(published.entry.growth_profile.map((row:any)=>[row.domain,row.growth===null])).toEqual([["mathematics",false],["programming",true],["data",true]]);
  const ranking=await (await fetch(`${base}/api/leaderboard`)).json();expect(ranking.entries).toHaveLength(1);expect(ranking.entries[0].username).toBe("测评甲");
  expect(JSON.stringify(ranking.entries)).not.toContain("challenge_id");expect(JSON.stringify(ranking.entries)).not.toContain("receipt_id");expect(JSON.stringify(ranking.entries)).not.toContain(personalAfter);
  expect((await (await fetch(`${base}/api/leaderboard?sort=growth&domain=programming`)).json()).entries).toEqual([]);
  const html=await (await fetch(base)).text();expect(html).toContain("本次成长</th>");expect(html).toContain("旧能力保持</th>");expect(html).toContain("暂定证据门槛");expect(html).toContain("非难度校准或认证");
  expect(html).not.toContain("泛化分</th>");expect(html).toContain('href="/?sort=growth&amp;domain=mathematics&amp;lang=zh" aria-current="page"');
  const different=await (await request(tokenD,{...body,model_version:model1})).json();expect(different.entry.benchmark).toBeNull();expect(different.entry.growth.net_delta).toBe(25);
  expect((await request(tokenD,{...body,evaluation_receipt_id:(await challenge(tokenD,model2)).receipt.receipt_id})).status).toBe(200);
});
test("low growth evidence remains a personal domain result and retention changes no growth rank",async()=>{
  const requestBody={base_identity:"c".repeat(64),model_version:personalBefore,skills:["uppercase"]};
  const c=await (await growthRequest(tokenE,"challenge",requestBody)).json();
  expect((await growthRequest(tokenE,"before",{challenge_id:c.challenge_id,model_version:personalBefore,answers:growthSheet(c.challenge_id,2)})).status).toBe(200);
  const r=await (await growthRequest(tokenE,"after",{challenge_id:c.challenge_id,model_version:personalAfter,answers:growthSheet(c.challenge_id,20)})).json();
  expect(r.eligible).toBe(false);
  const report=await (await request(tokenE,{username:"测评乙",upgrades:[],model_version:model1,growth_model_version:personalAfter,growth_receipt_id:r.receipt_id})).json();
  expect(report.entry.growth).toBeNull();expect(report.entry.growth_profile[1].growth.current_score).toBe(83.33);
  expect((await (await fetch(`${base}/api/leaderboard?sort=growth&domain=programming`)).json()).total).toBe(0);
  const retention={receipt_id:personalReceipt.receipt_id,model_version:personalAfter,answers:growthSheet(personalChallenge.challenge_id,15)};
  expect((await growthRequest(tokenD,"retention",retention)).status).toBe(409);
  const db=new Database(process.env.LEADERBOARD_DB!);db.query("UPDATE growth_receipts SET created_at=created_at-? WHERE id=?").run(86400001,personalReceipt.receipt_id);db.close();
  const response=await growthRequest(tokenD,"retention",retention);expect(response.status).toBe(200);expect(await response.json()).toMatchObject({correct:15,retained_correct:15,retention:83.33});
  const ranking=await (await fetch(`${base}/api/leaderboard`)).json();expect(ranking.entries[0].growth.net_delta).toBe(25);expect(ranking.entries[0].growth.retention_check.retention).toBe(83.33);
  expect((await growthRequest(tokenD,"retention",{...retention,answers:growthSheet(personalChallenge.challenge_id,24)})).status).toBe(409);
  for(const path of ["/growth.ts","/growth_challenges","/growth_receipts"]) expect((await fetch(`${base}${path}`)).status).toBe(404);
});

test("all domains show each user once using the latest current-model domain, never the best score",async()=>{
  const c=await (await growthRequest(tokenD,"challenge",{base_identity:"c".repeat(64),model_version:"9".repeat(64),skills:["uppercase"]})).json();
  expect((await growthRequest(tokenD,"before",{challenge_id:c.challenge_id,model_version:"9".repeat(64),answers:growthSheet(c.challenge_id,12)})).status).toBe(200);
  const r=await (await growthRequest(tokenD,"after",{challenge_id:c.challenge_id,model_version:personalAfter,answers:growthSheet(c.challenge_id,13)})).json();
  expect((await request(tokenD,{username:"测评甲",upgrades:[],growth_model_version:personalAfter,growth_receipt_id:r.receipt_id})).status).toBe(200);
  const all=await (await fetch(`${base}/api/leaderboard?domain=all`)).json();
  expect(all.domain).toBe("all");expect(all.entries).toHaveLength(1);
  expect(all.entries[0].growth).toMatchObject({domain:"programming",net_delta:4.17});
  const math=await (await fetch(`${base}/api/leaderboard?domain=mathematics`)).json();
  expect(math.entries[0].growth.net_delta).toBe(25);
  const html=await (await fetch(`${base}/?domain=all`)).text();
  expect(html).toContain('<option value="all" selected>全部</option>');
  expect(html).toContain('跨领域分数未经难度校准');expect(html).toContain('编程与计算机</span>');
  expect(html).not.toContain("评测待接入");expect(html).not.toContain("<optgroup");
  const db=new Database(process.env.LEADERBOARD_DB!);
  try {
    seedDisplayEntries(db);
    const pages=await Promise.all([1,2].map(async page=>(await fetch(`${base}/api/leaderboard?domain=all&page=${page}`)).json()));
    const entries=pages.flatMap(page=>page.entries);
    expect(entries).toHaveLength(pages[0].total);expect(new Set(entries.map(entry=>entry.username)).size).toBe(entries.length);
    expect(pages[0].entries).toHaveLength(100);
    const paging=await (await fetch(`${base}/?domain=all&page=2`)).text();
    expect(paging).toContain('page=1&amp;sort=growth&amp;domain=all');
  } finally { db.query("DELETE FROM display_entries").run();db.close(); }
});

test("public growth popovers contain only published aggregate history, never private tasks or answers",async()=>{
  const page=await fetch(`${base}/?domain=all&sort=evolution_count`),html=await page.text();
  expect(page.headers.get("content-security-policy")).toContain("script-src 'none'");
  expect(html).toContain('popovertarget="profile-');expect(html).toContain('popover role="dialog"');
  expect(html).toContain("成长概览");expect(html).toContain("成长历程");expect(html).toContain("新答对");
  expect(html).toContain('popovertargetaction="hide"');expect(html).not.toContain("<script");
  const list=await (await fetch(`${base}/api/leaderboard?domain=all&sort=evolution_count`)).json();
  const entry=list.entries.find((value:any)=>value.username==="测评甲");
  expect(entry.growth_history).toHaveLength(2);
  expect(entry.growth_history.map((value:any)=>value.domain)).toEqual(["programming","mathematics"]);
  for(const row of entry.growth_history) {
    expect(Object.keys(row).toSorted()).toEqual(["baseline_correct","baseline_score","completed_at","correct_to_wrong","current_correct","current_score","domain","eligible","gap_recovery","net_delta","question_count","retention","retention_check","wrong_to_correct"].toSorted());
  }
  const db=new Database(process.env.LEADERBOARD_DB!);
  try {
    const taskRows=db.query<{questions:string},[]>("SELECT questions FROM growth_challenges").all();
    for(const row of taskRows) for(const q of JSON.parse(row.questions)) {
      expect(html).not.toContain(q.prompt);expect(JSON.stringify(list)).not.toContain(q.prompt);
    }
  } finally {db.close();}
  for(const privateValue of [personalAfter,personalBefore,personalReceipt.receipt_id,personalChallenge.challenge_id]) {
    expect(html).not.toContain(privateValue);expect(JSON.stringify(list)).not.toContain(privateValue);
  }
  const c=await (await growthRequest(tokenD,"challenge",{base_identity:"c".repeat(64),model_version:"2".repeat(64),skills:["json_extract"]})).json();
  expect((await growthRequest(tokenD,"before",{challenge_id:c.challenge_id,model_version:"2".repeat(64),answers:growthSheet(c.challenge_id,12)})).status).toBe(200);
  const receipt=await (await growthRequest(tokenD,"after",{challenge_id:c.challenge_id,model_version:"3".repeat(64),answers:growthSheet(c.challenge_id,18)})).json();
  const stillPublic=await (await fetch(`${base}/api/leaderboard?domain=all&sort=evolution_count`)).json();
  const stillEntry=stillPublic.entries.find((value:any)=>value.username==="测评甲");
  expect(stillEntry.growth_history).toHaveLength(2);expect(JSON.stringify(stillPublic)).not.toContain(receipt.receipt_id);
});

test("bilingual website localizes profiles, preserves navigation and safely remembers manual language",async()=> {
  const response=await fetch(`${base}/?domain=all&lang=en`),html=await response.text();
  expect(response.headers.get("content-language")).toBe("en-US");
  expect(response.headers.get("cache-control")).toBe("private, no-store");
  expect(response.headers.get("vary")).toBe("Cookie, Accept-Language");
  expect(response.headers.get("set-cookie")).toContain("luban_lang=en;");
  for(const attribute of ["HttpOnly","Secure","SameSite=Lax"])expect(response.headers.get("set-cookie")).toContain(attribute);
  for(const label of ["Global Evolution Ranking","Growth Overview","Growth History","newly correct answers","Programming &amp; Computing"])expect(html).toContain(label);
  expect(html).toContain('domain=all&amp;lang=en');
  expect(html).toContain('name="lang" value="en"');
  expect(html).toContain('hreflang="zh-CN"');expect(html).toContain('hreflang="en"');
  const preferred=await fetch(`${base}/`,{headers:{Cookie:"luban_lang=en","Accept-Language":"zh-CN"}});
  expect(preferred.headers.get("content-language")).toBe("en-US");
  const explicit=await fetch(`${base}/?lang=zh`,{headers:{Cookie:"luban_lang=en","Accept-Language":"en"}});
  expect(explicit.headers.get("content-language")).toBe("zh-CN");
  expect((await fetch(`${base}/?lang=unsupported`)).status).toBe(400);
  const auto=await fetch(`${base}/`,{headers:{"Accept-Language":"en-US"}});
  expect(auto.headers.get("content-language")).toBe("en-US");
  const zhApi=await (await fetch(`${base}/api/leaderboard?domain=all&lang=zh`)).json();
  const enApi=await (await fetch(`${base}/api/leaderboard?domain=all&lang=en`)).json();
  expect(enApi).toEqual(zhApi);
});
